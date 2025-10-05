# Libraires
import numpy as np
import pandas as pd


def preprocess_data_retrain():
    """
    Preprocess NYC Taxi Fare dataset (training only).
    Returns df cleaned & feature-engineered DataFrame for fitting
    """

    # 1. Load dataset
    df = pd.read_csv("../data/raw/train.csv")

    # 2. Drop missing values and unused columns
    df = df.dropna()
    df.drop(columns=["key"], inplace=True)

    # 3. Convert datetime to NYC local time
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], utc=True, errors="coerce"
    )
    df["pickup_datetime_nyc"] = df["pickup_datetime"].dt.tz_convert("America/New_York")

    # 4. Remove outliers
    df = df[
        (df["fare_amount"] >= 1.0)
        & (df["fare_amount"] <= 500.0)
        & (df["pickup_longitude"].between(-75, -72))
        & (df["dropoff_longitude"].between(-78, -70))
        & (df["pickup_latitude"].between(40, 42))
        & (df["dropoff_latitude"].between(38, 44))
        & (df["passenger_count"].between(1, 6))
    ]

    # 5. Time-based features
    df["year"] = df["pickup_datetime_nyc"].dt.year
    df["month"] = df["pickup_datetime_nyc"].dt.month
    df["day_of_month"] = df["pickup_datetime_nyc"].dt.day
    df["day_of_week"] = df["pickup_datetime_nyc"].dt.weekday
    df["hour"] = df["pickup_datetime_nyc"].dt.hour
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # 6. US holidays
    month = df["pickup_datetime_nyc"].dt.month
    day = df["pickup_datetime_nyc"].dt.day
    df["is_holiday"] = (
        ((month == 1) & (day == 1))
        | ((month == 7) & (day == 4))
        | ((month == 11) & (day == 11))
        | ((month == 12) & (day == 25))
    ).astype(int)

    # 7. Drop original datetime
    df.drop(columns=["pickup_datetime", "pickup_datetime_nyc"], inplace=True)

    # 8. Distance features
    def haversine_np(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        return 6367 * c

    df["distance_km"] = haversine_np(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    df = df[df["distance_km"].between(0.5, 100)]

    # 9. Airport & landmark distances
    airports = {
        "jfk": (-73.7781, 40.6413),
        "lga": (-73.8740, 40.7769),
        "ewr": (-74.1745, 40.6895),
    }
    landmarks = {
        "times_sq": (-73.9855, 40.7580),
        "central_park": (-73.9654, 40.7829),
        "wtc": (-74.0099, 40.7126),
        "met": (-73.9632, 40.7794),
        "midtown": (-73.9851, 40.7549),
    }
    for name, (lon, lat) in airports.items():
        df[f"{name}_pickup_dist"] = haversine_np(
            df["pickup_longitude"], df["pickup_latitude"], lon, lat
        )
    for name, (lon, lat) in landmarks.items():
        df[f"{name}_dropoff_dist"] = haversine_np(
            df["dropoff_longitude"], df["dropoff_latitude"], lon, lat
        )

    return df


def preprocess_data_predict():
    """
    Preprocess new incoming NYC Taxi data for inference.
    - Generates time-based features
    - Adds holiday, distance, airport, and landmark features
    - Returns preprocessed DataFrame ready for model prediction
    """
    # 1. Load dataset
    df = pd.read_csv("../data/raw/test.csv")

    # 2. Convert datetime to NYC local time
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], utc=True, errors="coerce"
    )
    df["pickup_datetime_nyc"] = df["pickup_datetime"].dt.tz_convert("America/New_York")

    # 3. Time-based features
    df["year"] = df["pickup_datetime_nyc"].dt.year
    df["month"] = df["pickup_datetime_nyc"].dt.month
    df["day_of_month"] = df["pickup_datetime_nyc"].dt.day
    df["day_of_week"] = df["pickup_datetime_nyc"].dt.weekday
    df["hour"] = df["pickup_datetime_nyc"].dt.hour
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # 4. US holidays
    month = df["pickup_datetime_nyc"].dt.month
    day = df["pickup_datetime_nyc"].dt.day
    df["is_holiday"] = (
        ((month == 1) & (day == 1))
        | ((month == 7) & (day == 4))
        | ((month == 11) & (day == 11))
        | ((month == 12) & (day == 25))
    ).astype(int)

    # 5. Drop original datetime
    df.drop(columns=["pickup_datetime", "pickup_datetime_nyc"], inplace=True)

    # 6. Distance features
    def haversine_np(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        return 6367 * c

    df["distance_km"] = haversine_np(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )

    # 7. Airport & landmark distances
    airports = {
        "jfk": (-73.7781, 40.6413),
        "lga": (-73.8740, 40.7769),
        "ewr": (-74.1745, 40.6895),
    }
    landmarks = {
        "times_sq": (-73.9855, 40.7580),
        "central_park": (-73.9654, 40.7829),
        "wtc": (-74.0099, 40.7126),
        "met": (-73.9632, 40.7794),
        "midtown": (-73.9851, 40.7549),
    }
    for name, (lon, lat) in airports.items():
        df[f"{name}_pickup_dist"] = haversine_np(
            df["pickup_longitude"], df["pickup_latitude"], lon, lat
        )
    for name, (lon, lat) in landmarks.items():
        df[f"{name}_dropoff_dist"] = haversine_np(
            df["dropoff_longitude"], df["dropoff_latitude"], lon, lat
        )

    return df
