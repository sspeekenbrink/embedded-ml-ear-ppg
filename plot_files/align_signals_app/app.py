import dash
import sys
from layout import generate_dash_app_layout
from file_handling import get_available_directories
from callback import register_callbacks
import dash_bootstrap_components as dbc

top_level_path = "..."
# Add the path to sys.path to allow importing from subdirectories
sys.path.append(top_level_path)


folder_path = (
    "..."
)
# Initialize with empty list - will be populated by callback
available_folders = []

acceleration_results = {
    "Euclidean": 0.0,
    "SumAbs": 0.0,
    "Jerk": 0.0,
}

results = {
    "Euclidean": 0.0,
    "Dynamic Time Warping": 0.0,
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = generate_dash_app_layout(acceleration_results)
register_callbacks(app, folder_path)

if __name__ == "__main__":
    # Set debug=False to prevent automatic reloading that causes repeated execution
    # Change to debug=True only when actively developing
    app.run(debug=False)
