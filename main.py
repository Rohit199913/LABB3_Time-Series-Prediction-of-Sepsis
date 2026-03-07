from src.data.load_data import load_all_data

print("Script started")

datasets = load_all_data("data/raw")
df = datasets[0]

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print(df.isna().sum())