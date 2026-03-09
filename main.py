from src.data.load_data import load_all_data
from src.data.windowing import create_windows

print("Script started")

datasets = load_all_data("data/raw")
df = datasets[0]

print("Original shape:", df.shape)
print("Patients:", df["id"].nunique())

X, y, feature_columns = create_windows(
    df,
    window_size=12,
    horizon_steps=4,   # 2 hours ahead if each step = 30 min
    stride=1
)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of features:", len(feature_columns))
print("Positive labels:", y.sum())