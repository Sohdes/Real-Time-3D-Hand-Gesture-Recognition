import pandas as pd

data = pd.read_csv("gesture_data.csv")

print(data["label"].value_counts())