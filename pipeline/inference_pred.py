# Libraries
import joblib
from data_pipe import preprocess_data_predict

# 1 Preprocess new incoming data
df = preprocess_data_predict()

# 2 Load trained model + metadata
model, ref_column, target = joblib.load("../models/model.pkl")

# 3 Ensure the same feature order as training
X = df[ref_column]

# 4 Make predictions
predictions = model.predict(X)

# 5 Append predictions to the dataframe
df[target] = predictions

# 6 Keep only 'key' and predicted fare amount
df = df[["key", target]]

# 7 Save predictions to CSV
df.to_csv("../data/processed/predictions.csv", index=False)
