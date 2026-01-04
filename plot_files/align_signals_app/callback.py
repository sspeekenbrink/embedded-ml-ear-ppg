import json
import logging
import os
from typing import Tuple, Any, Dict

from dash import Input, Output, State, no_update, ctx

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
from wfdb.processing import xqrs_detect

from signal_processing.sensor_data import SensorData
from signal_processing.filter import filter_ecg_signal
from signal_processing.align_data import min_max_normalize, refine_peak_alignment
from signal_processing.acceleration_metrics import (
    jerk_based_acceleration,
    sum_absolute_acceleration,
    euclidean_accel,
)
from signal_processing.find_peaks import snap_peaks_to_local_maxima
from file_handling import get_available_directories

def json_file_path(base_path: str, signal_folder: str) -> str:
    return os.path.join(base_path, signal_folder, "data.json")


def save_params(path: str, ppg_hz: Any, time_offset: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"ppg_fs": ppg_hz, "ecg_offset": time_offset}, f, indent=4)


def load_params(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
    return None


def toast_payload(icon: str, header: str, message: str) -> Tuple[bool, str, str, str]:
    """Return (is_open, icon, header, children) for dbc.Toast."""
    return True, icon, header, message


def handle_save(
    path: str, ppg_hz: Any, time_offset: Any
) -> Tuple[Any, Any, Tuple[bool, str, str, str]]:
    try:
        save_params(path, ppg_hz, time_offset)
        logging.info(f"Saved to {path} (fs={ppg_hz}, offset={time_offset})")
        return (
            ppg_hz,
            time_offset,
            toast_payload("success", "Saved", f"Saved to {path}"),
        )
    except Exception as e:
        logging.exception("Save failed")
        return (
            no_update,
            no_update,
            toast_payload("danger", "Error", f"Error while saving: {e}"),
        )


def handle_load(
    path: str, current_ppg: Any, current_off: Any
) -> Tuple[Any, Any, Tuple[bool, str, str, str]]:
    try:
        data = load_params(path)
        new_ppg = data.get("ppg_fs", current_ppg)
        new_off = data.get("ecg_offset", current_off)
        return (
            new_ppg,
            new_off,
            toast_payload("success", "Loaded", f"Loaded from {path}"),
        )
    except FileNotFoundError:
        return (
            no_update,
            no_update,
            toast_payload("warning", "Not found", f"No data.json in {path}"),
        )
    except Exception as e:
        logging.exception("Load failed")
        return (
            no_update,
            no_update,
            toast_payload("danger", "Error", f"Error while loading: {e}"),
        )


def register_callbacks(app, base_path: str):
    """
    Expects in the layout:
      - Buttons:  button-save-json, button-load-json
      - Inputs:   ppg-signal-hz (value), time-offset (value), signal-folder (value)
      - Toast:    save-toast (is_open, icon, header, children) with auto-hide duration set in layout
    """

    # Cache for directory scanning to prevent repeated file system operations
    _directory_cache = {}

    @app.callback(
        Output("signal-folder", "options"),
        Output("signal-folder", "value"),
        Input("signal-folder", "id"),  # Trigger on component mount
        prevent_initial_call=False,
    )
    def populate_signal_folders(_):
        """Populate the signal folder dropdown with available directories."""
        # Use caching to avoid repeated file system operations
        cache_key = base_path
        if cache_key in _directory_cache:
            directories = _directory_cache[cache_key]
        else:
            directories = get_available_directories(base_path)
            _directory_cache[cache_key] = directories

        options = [{"label": folder, "value": folder} for folder in directories]
        default_value = directories[0] if directories else ""

        return options, default_value

    @app.callback(
        Output("ppg-signal-hz", "value"),
        Output("time-offset", "value"),
        Output("save-toast", "is_open"),
        Output("save-toast", "icon"),
        Output("save-toast", "header"),
        Output("save-toast", "children"),
        Input("button-save-json", "n_clicks"),
        Input("button-load-json", "n_clicks"),
        State("ppg-signal-hz", "value"),
        State("time-offset", "value"),
        State("signal-folder", "value"),
        prevent_initial_call=True,
    )
    def save_or_load(save_clicks, load_clicks, ppg_hz, time_offset, signal_folder):
        which = ctx.triggered_id  # 'button-save-json' or 'button-load-json'
        path = json_file_path(base_path, signal_folder)

        if which == "button-save-json":
            new_ppg, new_off, toast = handle_save(path, ppg_hz, time_offset)
            return new_ppg, new_off, *toast

        if which == "button-load-json":
            new_ppg, new_off, toast = handle_load(path, ppg_hz, time_offset)
            return new_ppg, new_off, *toast

        # Fallback (shouldn't occur)
        return no_update, no_update, False, no_update, no_update, no_update

    @app.callback(
        Output("aligned-signals", "figure"),
        Output("snapped-peak-times", "data"),
        Output("accel-toast", "is_open"),
        Output("accel-toast", "icon"),
        Output("accel-toast", "header"),
        Output("accel-toast", "children"),
        Input("button-plot-graphs", "n_clicks"),
        State("signal-folder", "value"),
        State("accel-method", "value"),
        State("ppg-signal-hz", "value"),
        State("time-offset", "value"),
        State("ecg-vertical-offset", "value"),
        State("normalize-signals", "value"),
        State("snap-peaks-enabled", "value"),
        State("snap-window-seconds", "value"),
        State("show-ir-signal", "value"),
        State("show-red-signal", "value"),
        prevent_initial_call=True,
    )
    def plot_on_demand(
        n_clicks,
        signal_folder,
        accel_method,
        ppg_hz,
        time_offset,
        ecg_vertical_offset,
        normalize,
        snap_enabled,
        snap_window_s,
        show_ir,
        show_red,
    ):
        if not n_clicks:
            raise PreventUpdate

        try:
            # ---------------- I/O paths ----------------
            folder = os.path.join(base_path, signal_folder)
            ppg_path = os.path.join(folder, "ppg.txt")
            ecg_path = os.path.join(folder, "ecg.csv")
            acc_path = os.path.join(folder, "acc.csv")
            acc_present = os.path.exists(acc_path)

            # ---------------- Load PPG via your library ----------------
            ear = SensorData(file_location=ppg_path, sample_rate=float(ppg_hz))
            ear_time = np.arange(len(ear.led_green_values)) / float(ppg_hz)
            ear_signal_raw = ear.led_green_values
            ear_ir_raw = ear.led_ir_values
            ear_red_raw = ear.led_red_values
            time_ear_accel = ear_time

            df_ecg = pd.read_csv(ecg_path)

            if "time" in df_ecg.columns:
                ecg_time = df_ecg["time"].to_numpy()
            else:
                ECG_FS_DEFAULT = 250.0
                ecg_time = np.arange(len(df_ecg)) / ECG_FS_DEFAULT

            if "channel" in df_ecg.columns:
                ecg_raw = df_ecg["channel"].to_numpy()
            else:
                num_cols = [
                    c
                    for c in df_ecg.columns
                    if np.issubdtype(df_ecg[c].dtype, np.number)
                ]
                if not num_cols:
                    raise ValueError(
                        "ecg.csv has no numeric column for the ECG channel."
                    )
                ecg_raw = df_ecg[num_cols[0]].to_numpy()

            # Derive ECG fs if possible
            if "time" in df_ecg.columns and len(ecg_time) > 1:
                dt = np.diff(ecg_time)
                dt = dt[np.isfinite(dt) & (dt > 0)]
                ecg_fs = float(np.round(1.0 / np.median(dt))) if len(dt) else 250.0
            else:
                ecg_fs = 250.0

            ecg_filtered = filter_ecg_signal(ecg_raw, fs=ecg_fs)
            ecg_for_detection = min_max_normalize(ecg_filtered)  # for xqrs
            ecg_for_detection = ecg_filtered - np.mean(ecg_filtered)  # for xqrs
            ecg_for_detection = ecg_filtered
            rpeaks_idx = xqrs_detect(ecg_for_detection, fs=ecg_fs, verbose=False)

            time_offset = float(time_offset or 0.0)
            rpeaks_times_after_global = ecg_time[rpeaks_idx] + time_offset
            # refined = refine_peak_alignment(
            #     rpeaks_times=rpeaks_times_after_global,
            #     ear_signal=ear_signal_raw,
            #     ear_time=ear_time
            # )
            refined = 0
            total_offset = time_offset + refined

            # Shift ECG time by the total offset (chest-side timeline → ear timeline)
            ecg_time_aligned = ecg_time + total_offset

            # Prepare acceleration data if present
            accel_method = (accel_method or "Euclidean").strip()
            ear_metrics = {
                "Euclidean": ear.accel_euclidean,
                "SumAbs": ear.absolute_accel,
                "Jerk": ear.jerk_accel,
            }
            if accel_method not in ear_metrics:
                raise ValueError(f"Unknown acceleration method '{accel_method}'.")

            acc_ear_raw = np.asarray(ear_metrics[accel_method])

            # Defaults if acceleration missing
            acc_chest_raw = None
            time_chest_accel_aligned = None

            if acc_present:
                df_acc = pd.read_csv(acc_path)

                # time column must exist for chest accel
                if "time" not in df_acc.columns:
                    raise ValueError("acc.csv must contain a 'time' column.")
                time_chest_accel = df_acc["time"].to_numpy()

                def _three_channels(df):
                    if all(
                        c in df.columns for c in ["channel1", "channel2", "channel3"]
                    ):
                        return df["channel1"], df["channel2"], df["channel3"]
                    numeric = [
                        c
                        for c in df.columns
                        if c != "time" and np.issubdtype(df[c].dtype, np.number)
                    ]
                    if len(numeric) < 3:
                        raise ValueError(
                            "acc.csv needs channel1..3 or at least three numeric accel columns."
                        )
                    return df[numeric[0]], df[numeric[1]], df[numeric[2]]

                c1, c2, c3 = _three_channels(df_acc)

                chest_metrics = {
                    "Euclidean": euclidean_accel(c1, c2, c3),
                    "SumAbs": sum_absolute_acceleration(c1, c2, c3),
                    "Jerk": jerk_based_acceleration(c1, c2, c3),
                }
                acc_chest_raw = np.asarray(chest_metrics[accel_method])

                # Align chest accel timeline like ECG (same device) → add total_offset
                time_chest_accel_aligned = time_chest_accel + total_offset

            # (ECG detection uses normalized; plotting respects checkbox)
            if normalize:
                ear_signal_plot = min_max_normalize(ear_signal_raw)
                ear_ir_plot = min_max_normalize(ear_ir_raw)
                ear_red_plot = min_max_normalize(ear_red_raw)
                ecg_plot = min_max_normalize(ecg_filtered)
                acc_ear_plot = min_max_normalize(acc_ear_raw)
                acc_chest_plot = (
                    min_max_normalize(acc_chest_raw)
                    if acc_chest_raw is not None
                    else None
                )
            else:
                ear_signal_plot = ear_signal_raw
                ear_ir_plot = ear_ir_raw
                ear_red_plot = ear_red_raw
                ecg_plot = ecg_filtered
                acc_ear_plot = acc_ear_raw
                acc_chest_plot = acc_chest_raw

            # R-peaks markers projected on ear signal (times in ear/PPG timeline)
            rpeaks_times_final = ecg_time_aligned[rpeaks_idx]

            snapped_data = None
            use_times_for_markers = rpeaks_times_final
            marker_name = "ECG R-Peaks (on PPG)"
            marker_color = None

            if bool(snap_enabled):
                try:
                    window_val = (
                        float(snap_window_s) if snap_window_s is not None else 0.12
                    )
                    snapped_times, snapped_idx = snap_peaks_to_local_maxima(
                        ppg_signal=ear_signal_raw,
                        ppg_time=ear_time,
                        candidate_peak_times=rpeaks_times_final,
                        window_seconds=window_val,
                    )
                    # Filter out any -1 indices just in case (should not occur when window contains samples)
                    valid = snapped_idx >= 0
                    if np.any(valid):
                        use_times_for_markers = snapped_times[valid]
                        snapped_data = {
                            "times_s": use_times_for_markers.tolist(),
                            "window_s": window_val,
                            "source": "ecg_to_ppg_snap",
                        }
                        marker_name = "Snapped Peaks (PPG maxima)"
                        marker_color = "red"
                    else:
                        snapped_data = {
                            "times_s": [],
                            "window_s": window_val,
                            "source": "ecg_to_ppg_snap",
                        }
                except Exception as e:
                    # In case of any snapping error, fall back to original times
                    logging.exception("Snapping peaks failed")
                    snapped_data = {"error": str(e)}

            # Calculate ECG peaks on all enabled PPG signals
            rpeaks_on_ear = np.interp(
                use_times_for_markers, time_ear_accel, ear_signal_plot
            )
            rpeaks_on_ear_original = np.interp(
                rpeaks_times_final, time_ear_accel, ear_signal_plot
            )

            # Calculate peaks on IR signal if enabled
            rpeaks_on_ir = None
            rpeaks_on_ir_original = None
            if bool(show_ir):
                rpeaks_on_ir = np.interp(
                    use_times_for_markers, time_ear_accel, ear_ir_plot
                )
                rpeaks_on_ir_original = np.interp(
                    rpeaks_times_final, time_ear_accel, ear_ir_plot
                )

            # Calculate peaks on Red signal if enabled
            rpeaks_on_red = None
            rpeaks_on_red_original = None
            if bool(show_red):
                rpeaks_on_red = np.interp(
                    use_times_for_markers, time_ear_accel, ear_red_plot
                )
                rpeaks_on_red_original = np.interp(
                    rpeaks_times_final, time_ear_accel, ear_red_plot
                )

            fig = go.Figure()

            # PPG (ear)
            fig.add_trace(
                go.Scatter(
                    x=time_ear_accel,
                    y=ear_signal_plot,
                    mode="lines",
                    name="LED Green (PPG)",
                )
            )

            # IR Signal (if enabled)
            if bool(show_ir):
                fig.add_trace(
                    go.Scatter(
                        x=time_ear_accel, y=ear_ir_plot, mode="lines", name="LED IR"
                    )
                )

            # Red Signal (if enabled)
            if bool(show_red):
                fig.add_trace(
                    go.Scatter(
                        x=time_ear_accel, y=ear_red_plot, mode="lines", name="LED Red"
                    )
                )

            # ECG (plot, with vertical offset)
            fig.add_trace(
                go.Scatter(
                    x=ecg_time_aligned,
                    y=ecg_plot + float(ecg_vertical_offset or 0.0),
                    mode="lines",
                    name="ECG",
                )
            )

            # R-peaks on all enabled PPG signals
            # If snapping is enabled, plot both: snapped (highlighted) and original for comparison
            if bool(snap_enabled):
                # Original peaks on Green PPG
                fig.add_trace(
                    go.Scatter(
                        x=rpeaks_times_final,
                        y=rpeaks_on_ear_original,
                        mode="markers",
                        marker=dict(size=6, symbol="x", color="rgba(0,0,0,0.5)"),
                        name="ECG R-Peaks (original) - Green",
                    )
                )
                # Snapped peaks on Green PPG
                fig.add_trace(
                    go.Scatter(
                        x=use_times_for_markers,
                        y=rpeaks_on_ear,
                        mode="markers",
                        marker=dict(size=8, symbol="x", color=marker_color or "red"),
                        name=marker_name + " - Green",
                    )
                )

                # Original and snapped peaks on IR signal (if enabled)
                if bool(show_ir):
                    fig.add_trace(
                        go.Scatter(
                            x=rpeaks_times_final,
                            y=rpeaks_on_ir_original,
                            mode="markers",
                            marker=dict(size=6, symbol="x", color="rgba(0,0,0,0.5)"),
                            name="ECG R-Peaks (original) - IR",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=use_times_for_markers,
                            y=rpeaks_on_ir,
                            mode="markers",
                            marker=dict(
                                size=8, symbol="x", color=marker_color or "red"
                            ),
                            name=marker_name + " - IR",
                        )
                    )

                # Original and snapped peaks on Red signal (if enabled)
                if bool(show_red):
                    fig.add_trace(
                        go.Scatter(
                            x=rpeaks_times_final,
                            y=rpeaks_on_red_original,
                            mode="markers",
                            marker=dict(size=6, symbol="x", color="rgba(0,0,0,0.5)"),
                            name="ECG R-Peaks (original) - Red",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=use_times_for_markers,
                            y=rpeaks_on_red,
                            mode="markers",
                            marker=dict(
                                size=8, symbol="x", color=marker_color or "red"
                            ),
                            name=marker_name + " - Red",
                        )
                    )
            else:
                # No snapping - show peaks on all enabled signals
                fig.add_trace(
                    go.Scatter(
                        x=use_times_for_markers,
                        y=rpeaks_on_ear,
                        mode="markers",
                        marker=dict(size=12, symbol="x"),
                        name=marker_name + " - Green",
                    )
                )

                # Peaks on IR signal (if enabled)
                if bool(show_ir):
                    fig.add_trace(
                        go.Scatter(
                            x=use_times_for_markers,
                            y=rpeaks_on_ir,
                            mode="markers",
                            marker=dict(size=7, symbol="x"),
                            name=marker_name + " - IR",
                        )
                    )

                # Peaks on Red signal (if enabled)
                if bool(show_red):
                    fig.add_trace(
                        go.Scatter(
                            x=use_times_for_markers,
                            y=rpeaks_on_red,
                            mode="markers",
                            marker=dict(size=7, symbol="x"),
                            name=marker_name + " - Red",
                        )
                    )

            # Acceleration traces only if acc.csv present
            if acc_present:
                # Ear acceleration
                fig.add_trace(
                    go.Scatter(
                        x=ear_time,
                        y=acc_ear_plot,
                        mode="lines",
                        name=f"Accel {accel_method} (ear)",
                    )
                )

                # Chest acceleration (aligned)
                fig.add_trace(
                    go.Scatter(
                        x=time_chest_accel_aligned,
                        y=acc_chest_plot,
                        mode="lines",
                        name=f"Accel {accel_method} (chest +offset)",
                    )
                )

            # Create subtitle with enabled signals
            signal_names = ["PPG"]
            if bool(show_ir):
                signal_names.append("IR")
            if bool(show_red):
                signal_names.append("Red")
            signal_subtitle = ", ".join(signal_names)
            subtitle = f"(Signals: {signal_subtitle}, Accel: {accel_method})"
            fig.update_layout(
                title=f"Aligned PPG Signals, ECG, and Acceleration {subtitle}",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                legend_title="Signals",
                margin=dict(l=40, r=20, t=60, b=40),
            )

            # fig.add_annotation(
            #     xref="paper", yref="paper", x=0, y=1.08, showarrow=False,
            #     text=f"Applied offsets → global: {time_offset:.3f}s, refined: {refined:.3f}s, total: {total_offset:.3f}s"
            # )

            if acc_present:
                return fig, snapped_data, False, no_update, no_update, no_update
            else:
                warn = toast_payload(
                    "warning",
                    "Acceleration Missing",
                    f"No acc.csv found in '{folder}'. Acceleration signals were not plotted.",
                )
                return fig, snapped_data, *warn

        except Exception as e:
            logging.exception("Plotting failed")
            err = go.Figure()
            err.update_layout(
                title=f"Error while plotting: {e}",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
            )
            toast = toast_payload("danger", "Error", f"Error while plotting: {e}")
            return err, None, *toast

    @app.callback(
        Output("peaks-toast", "is_open"),
        Output("peaks-toast", "icon"),
        Output("peaks-toast", "header"),
        Output("peaks-toast", "children"),
        Input("button-save-peaks", "n_clicks"),
        State("signal-folder", "value"),
        State("snapped-peak-times", "data"),
        prevent_initial_call=True,
    )
    def save_snapped_peaks(n_clicks, signal_folder, snapped_data):
        if not n_clicks:
            raise PreventUpdate
        try:
            if (
                not snapped_data
                or not isinstance(snapped_data, dict)
                or "times_s" not in snapped_data
            ):
                raise ValueError(
                    "No snapped peaks available. Generate plots with snapping enabled first."
                )

            folder = os.path.join(base_path, signal_folder)
            out_path = os.path.join(folder, "peaks_snapped.json")

            payload = {
                "version": 1,
                "times_s": snapped_data.get("times_s", []),
                "window_s": snapped_data.get("window_s"),
                "source": snapped_data.get("source", "ecg_to_ppg_snap"),
            }

            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)

            return toast_payload(
                "success",
                "Peaks Saved",
                f"Saved {len(payload['times_s'])} peaks to peaks_snapped.json",
            )
        except Exception as e:
            logging.exception("Saving peaks failed")
            return toast_payload("danger", "Error", f"Failed to save peaks: {e}")
