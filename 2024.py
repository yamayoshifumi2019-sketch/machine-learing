import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")


# Load FastF1 2023 Australian GP race session
session_2023 = fastf1.get_session(2023, 3, "R")
session_2023.load()

#Extract lap times
laps_2023 = session_2023.laps[["Driver", "LapTime"]].copy()
laps_2023.dropna(subset=["LapTime"], inplace=True)
laps_2023["LapTime (s)"] = laps_2023["LapTime"].dt.total_seconds()

# 2024 Qualifying Data
qualifying_2024 = pd.DataFrame({
    "Driver": ["Max Verstappen", "Carlos Sainz", "Sergio Perez", "Lando Norris", "Charles Leclerc", "Oscar Piastri", "George Russell", "Yuki Tsunoda", 
        "Lance Stroll", "Fernando Alonso", "Lewis Hamilton", "Alexander Albon", "Valtteri Bottas", "Kevin Magnussen", "Esteban Ocon", "Nico Hulkenberg", 
        "Pierre Gasly", "Daniel Ricciardo", "Zhou Guanyu"],
    "QualifyingTime (s)": [75.915, 76.185, 76.274, 76.315, 76.435, 76.572, 76.724, 76.788, 
        77.072, 77.552, 76.960, 77.135, 77.340, 77.427, 77.697, 77.976, 77.982, 78.085, 78.188]
})

# Map full names to FastF1 3-letter codes
driver_mapping = {"Max Verstappen": "VER", "Carlos Sainz": "SAI", "Sergio Perez": "PER", "Lando Norris": "NOR",
    "Charles Leclerc": "LEC", "Oscar Piastri": "PIA", "George Russell": "RUS", "Yuki Tsunoda": "TSU", "Lance Stroll": "STR",
    "Fernando Alonso": "ALO", "Lewis Hamilton": "HAM", "Alexander Albon": "ALB", "Valtteri Bottas": "BOT",
    "Kevin Magnussen": "MAG", "Esteban Ocon": "OCO", "Nico Hulkenberg": "HUL", "Pierre Gasly": "GAS", "Daniel Ricciardo": "RIC","Zhou Guanyu": "ZHO"
}

qualifying_2024["DriverCode"] = qualifying_2024["Driver"].map(driver_mapping)

# Merge 2024 Qualifying Data with 2023 Race Data
merged_data = qualifying_2024.merge(laps_2023, left_on="DriverCode", right_on="Driver")

# Use only "QualifyingTime (s)" as a feature
X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime (s)"]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2025 qualifying times
predicted_lap_times = model.predict(qualifying_2024[["QualifyingTime (s)"]])
qualifying_2024["PredictedRaceTime (s)"] = predicted_lap_times 

# Rank drivers by predicted race time
qualifying_2024= qualifying_2024.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏁 Predicted 2025 Chinese GP Winner 🏁\n")
print(qualifying_2024[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")