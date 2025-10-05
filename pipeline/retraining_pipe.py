# Libraries
import joblib
import pandas as pd
from data_pipe import preprocess_data_retrain

# 1 Run preprocessing pipeline
df1 = preprocess_data_retrain()

# 2 Load additional CSVs from processed folder
df2 = pd.read_csv("../data/processed/train_fit.csv")
# Combine all data
df = pd.concat([df1, df2], ignore_index=True)

# 3 Load Model + Metadata
model, ref_column, saved_target = joblib.load("../models/model.pkl")

# 4 Separate features and target using saved metadata
X = df[ref_column]
y = df[saved_target]

# 5 Retrain the model on new data
model.fit(X, y)

# 6 Save updated model (overwrite or version up)
joblib.dump((model, ref_column, saved_target), "../models/model.pkl")
