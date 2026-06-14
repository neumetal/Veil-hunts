import pandas as pd
import numpy as np
import app

app.st.session_state.grid_center_lat = 40.31
app.st.session_state.grid_center_lon = -75.13
app.st.session_state.grid_radius_mi = 40.0
app.st.session_state.observations = [
    {
        "date": "2025-11-15",
        "time_start": "03:00",
        "time_end": "06:00",
        "fog_observed": True
    }
]
app.st.session_state.plant_obs_dict = {
    "Japanese Stiltgrass": [
        {"lat": 40.31, "lon": -75.13}
    ]
}
app.st.session_state.selected_plants = [
    {"id": 1, "sci": "Microstegium vimineum", "common": "Japanese Stiltgrass"}
]
app.st.session_state.selected_csv = "C:\\veil_finder_project\\scans_export\\fog_master.csv"

scores = app.get_scores()
if scores is not None:
    print(scores.columns)
    print(scores.head()[["Lat", "Lon", "MatchRate", "PlantScore_Japanese Stiltgrass", "PlantScore", "CombinedScore"]])
else:
    print("Scores is None")
