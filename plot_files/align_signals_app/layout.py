from dash import dcc, html, callback_context
import dash_bootstrap_components as dbc


def generate_dash_app_layout(acceleration_results):
    layout = html.Div(
        [
            html.H1("Signal Alignment Dashboard"),
            html.Div(
                [
                    html.Label("Select signal folder:"),
                    dcc.Dropdown(
                        id="signal-folder",
                        options=[],  # Will be populated by callback
                        value="",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label(
                        "Select Acceleration Method (used for alignment offset):"
                    ),
                    dcc.Dropdown(
                        id="accel-method",
                        options=[
                            {"label": m, "value": m}
                            for m in acceleration_results.keys()
                        ],
                        value="Euclidean",
                    ),
                ]
            ),
            html.Div([html.Button("Load from JSON", id="button-load-json")]),
            html.Div(
                [
                    html.Label("Time Offset (seconds):"),
                    dcc.Input(
                        id="time-offset",
                        type="number",
                        value=acceleration_results["Euclidean"],
                        step=0.00001,
                    ),
                ]
            ),
            # Create a html div with dcc input for ppg hz, which is normally 100 Hz
            html.Div(
                [
                    html.Label("PPG Signal HZ:"),
                    dcc.Input(
                        id="ppg-signal-hz",
                        type="number",
                        value=100,  # Default PPG signal Hz
                        step=0.00000001,
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Vertical ECG Offset:"),
                    dcc.Input(
                        id="ecg-vertical-offset",
                        type="number",
                        value=0.0,  # Default vertical offset
                        step=0.1,
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Checkbox(
                        id="normalize-signals",
                        label="Normalize PPG & ECG to 0-1",
                        value=False,  # default: off
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Checkbox(
                        id="snap-peaks-enabled",
                        label="Snap ECG-derived peaks to nearest PPG local maxima",
                        value=False,
                    ),
                    html.Label("Snap window (s):"),
                    dcc.Input(
                        id="snap-window-seconds", type="number", value=0.12, step=0.01
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Checkbox(
                        id="show-ir-signal",
                        label="Show IR Signal",
                        value=False,
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Checkbox(
                        id="show-red-signal",
                        label="Show Red Signal",
                        value=False,
                    ),
                ]
            ),
            html.Div([html.Button("Save Peaks", id="button-save-peaks")]),
            html.Button("Save to JSON", id="button-save-json"),
            html.Button("Plot Graphs", id="button-plot-graphs"),
            html.Div(
                [
                    # ... your existing controls (button-save-json etc.) ...
                    dbc.Toast(
                        id="save-toast",
                        is_open=False,
                        header="Saved",
                        icon="success",  # green check; use "danger" for errors
                        duration=5000,  # auto-hide after 2 seconds
                        style={
                            "position": "fixed",
                            "top": 20,
                            "right": 20,
                            "zIndex": 2000,
                        },
                    )
                ]
            ),
            html.Div(
                [
                    dbc.Toast(
                        id="peaks-toast",
                        is_open=False,
                        header="Peaks",
                        icon="success",
                        duration=5000,
                        style={
                            "position": "fixed",
                            "top": 140,
                            "right": 20,
                            "zIndex": 2000,
                        },
                    )
                ]
            ),
            html.Div(
                [
                    dbc.Toast(
                        id="accel-toast",
                        is_open=False,
                        header="Acceleration Warning",
                        icon="warning",
                        duration=6000,
                        style={
                            "position": "fixed",
                            "top": 80,
                            "right": 20,
                            "zIndex": 2000,
                        },
                    )
                ]
            ),
            dcc.Graph(
                id="aligned-signals",
                config={
                    "toImageButtonOptions": {
                        "format": "svg",  # enable SVG export
                        "filename": "aligned_signals",
                        "scale": 2,  # higher = sharper image
                    }
                },
            ),
            dcc.Store(id="snapped-peak-times"),
        ]
    )
    return layout
