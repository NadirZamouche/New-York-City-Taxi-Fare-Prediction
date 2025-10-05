# Libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1 Data Collection
# Load datasets
df = pd.read_csv("../raw/train.csv")
df.head()

# 2 Data Inspection
df.info()  # for some reson no null NA count has been displayed
df.describe()
print(df.isna().sum())

# 3 Data Cleaning
# 3.1 droping rows with missing values (376/55423856 ≈ 0.00068%)
print("Total rows with NA:", len(df))
df = df.dropna()  # Dropping NA
print("Total rows without NA:", len(df))

# 3.2 droping unused columns
df.drop(columns=["key"], inplace=True)
df.head()

# 3.3 Convert pickup_datetime to pandas datetime with UTC
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True, errors="coerce")
# Convert from UTC to New York local time (handles DST automatically)
df["pickup_datetime_nyc"] = df["pickup_datetime"].dt.tz_convert("America/New_York")
df.head()

""" Filtering out outliers
- Fare amount:
  • Can't be negative.
  • Very high fares (e.g. > $500) are unrealistic for a yellow cab trip in NYC (typical max is <$200, even JFK → far suburbs).

- Coordinates (lat/lon):
  • NYC sits around longitude ~ -74, latitude ~ 40.7.
  • So anything far outside [-75, -72] and [40, 42] is not NYC.
  
- Passenger count:
  • A yellow cab can officially hold up to 6 passengers.
  • 0 or negative passengers is invalid.
"""


def remove_outliers(df):
    return df[
        (df["fare_amount"] >= 1.0)
        & (df["fare_amount"] <= 500.0)
        & (df["pickup_longitude"] >= -75)  # Strict for pickups (must be NYC)
        & (df["pickup_longitude"] <= -72)  # Strict for pickups (must be NYC)
        & (df["dropoff_longitude"] >= -78)  # Looser for dropoffs (can be farther away)
        & (df["dropoff_longitude"] <= -70)  # Looser for dropoffs (can be farther away)
        & (df["pickup_latitude"] >= 40)  # Strict for pickups (must be NYC)
        & (df["pickup_latitude"] <= 42)  # Strict for pickups (must be NYC)
        & (df["dropoff_latitude"] >= 38)  # Looser for dropoffs (can be farther away)
        & (df["dropoff_latitude"] <= 44)  # Looser for dropoffs (can be farther away)
        & (df["passenger_count"] >= 1)
        & (df["passenger_count"] <= 6)
    ]


df = remove_outliers(df)
print("Total rows:", len(df))

# 4 Feature Engineering
# 4.1 Extracting time realted features
df["year"] = df["pickup_datetime_nyc"].dt.year
df["month"] = df["pickup_datetime_nyc"].dt.month
df["day_of_month"] = df["pickup_datetime_nyc"].dt.day
df["day_of_week"] = df["pickup_datetime_nyc"].dt.weekday
df["hour"] = df["pickup_datetime_nyc"].dt.hour

# Weekend flag (Saturday=5, Sunday=6)
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Rush hour flag (morning: 7–9, evening: 16–19)
df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)


# Holiday flag (using US holidays)
def add_us_holidays(df, datetime_col="pickup_datetime_nyc"):
    """Add US holiday flag based on fixed-date federal holidays."""
    dates = df[datetime_col]
    month = dates.dt.month
    day = dates.dt.day

    holiday = (
        ((month == 1) & (day == 1))  # New Year's Day
        | ((month == 7) & (day == 4))  # Independence Day
        | ((month == 11) & (day == 11))  # Veterans Day
        | ((month == 12) & (day == 25))  # Christmas Day
    )

    df["is_holiday"] = holiday.astype(int)
    return df


df = add_us_holidays(df)
df.head()
df.tail()

# Drop original datetime columns
df = df.drop(columns=["pickup_datetime", "pickup_datetime_nyc"])
df.head()


# 4.2 Adding ride distance
def haversine_np(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length."""

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )  # Haversine function

    c = 2 * np.arcsin(
        np.sqrt(a)
    )  # great-circle distance (the actual distance on Earth’s surface)
    km = 6367 * c  # 1 R ≈ 6367 km
    return km


df["distance_km"] = haversine_np(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
)
df.head()
df.tail()
df["distance_km"].describe()
print("Total rows:", len(df))

df = df[(df["distance_km"] > 0.5) & (df["distance_km"] < 100)]
print("Total rows:", len(df))
df["distance_km"].describe()
df["distance_km"].min()

# 4.3 Adding distance to Popular Landmarks features
# 4.3.1 Adding distance from pickup locations to NYC Airports features
# Define Airports coordinates (lon, lat)
airports = {
    "jfk": (-73.7781, 40.6413),  # JFK Airport
    "lga": (-73.8740, 40.7769),  # LaGuardia Airport
    "ewr": (-74.1745, 40.6895),  # Newark Airport
}


def add_pickup_airports_distances(df, airports_dict):
    """
    Add pickup and dropoff distances to popular NYC landmarks.

    df : pandas DataFrame with pickup/dropoff latitude/longitude
    landmarks_dict : dictionary of landmarks with (lon, lat)
    """
    for name, (lon, lat) in airports_dict.items():
        # Pickup distance
        df[f"{name}_pickup_dist"] = haversine_np(
            df["pickup_longitude"], df["pickup_latitude"], lon, lat
        )
    return df


# Apply to your df
df = add_pickup_airports_distances(df, airports)
df.head()

# 4.3.2 Adding distance from dropoff locations to NYC Tourist Landmarks features
landmarks = {
    "times_sq": (-73.9855, 40.7580),  # Times Square
    "central_park": (-73.9654, 40.7829),  # Central Park
    "wtc": (-74.0099, 40.7126),  # World Trade Center
    "met": (-73.9632, 40.7794),  # Metropolitan Museum of Art
    "midtown": (-73.9851, 40.7549),  # Midtown Manhattan
}


def add_dropoff_landmarks_distances(df, landmarks_dict):
    """
    Add pickup and dropoff distances to popular NYC landmarks.

    df : pandas DataFrame with pickup/dropoff latitude/longitude
    landmarks_dict : dictionary of landmarks with (lon, lat)
    """
    for name, (lon, lat) in landmarks_dict.items():
        # Pickup distance
        df[f"{name}_dropoff_dist"] = haversine_np(
            df["dropoff_longitude"], df["dropoff_latitude"], lon, lat
        )
    return df


# Apply to your df
df = add_dropoff_landmarks_distances(df, landmarks)
df.head()

# 5 Exploartory Data Analysis EDA & Visualization
# 5.1 Box Plot
# Columns to check for outliers (11st "fare_amount" is not included)
columns_selection = df.columns[1:]


# Define a function to identify outliers using IQR method
def identify_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))


# Create subplots for each column
fig, axes = plt.subplots(
    nrows=len(columns_selection), ncols=1, figsize=(8, 4 * len(columns_selection))
)
fig.subplots_adjust(hspace=0.5)
# Loop through each selected column
for i, column in enumerate(columns_selection):
    # Draw box plot
    axes[i].boxplot(df[column])
    axes[i].set_title(f"Box Plot for {column}")
    axes[i].set_ylabel(column)
# Show the box plots and outliers
plt.show()

# 5.2 Correlation Matrix
# calculate correlation matrix
corrmat = df.corr()
# select column names for plotting
top_corr_features = corrmat.index
# plot heat map
plt.figure(figsize=(13, 13))
g = sns.heatmap(
    corrmat[top_corr_features].loc[top_corr_features], annot=True, cmap="RdBu"
)
plt.show()

# 5.3 Histograms
# Calculate the number of rows and columns for subplots
num_columns = 3
num_rows = math.ceil(len(columns_selection) / num_columns)
# Create subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(15, 4 * num_rows))
fig.subplots_adjust(hspace=0.5)
# Loop through each selected column
for i, column in enumerate(columns_selection):
    row_num = i // num_columns
    col_num = i % num_columns
    # Plot histogram with bins=50
    df[column].hist(bins=50, ax=axes[row_num, col_num])
    axes[row_num, col_num].set_title(f"Histogram for {column}")
    axes[row_num, col_num].set_xlabel(column)
    axes[row_num, col_num].set_ylabel("Frequency")
# Remove any empty subplots
for i in range(len(columns_selection), num_rows * num_columns):
    fig.delaxes(axes.flatten()[i])
# Show the
plt.show()

# 5.4 Histogram of Fare Amount (0-500)
plt.figure(figsize=(10, 6))
plt.hist(df["fare_amount"], bins=100, color="skyblue", edgecolor="black")
plt.title("Distribution of Fare Amount", fontsize=14)
plt.xlabel("Fare Amount ($)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xlim(0, 500)  # cap at 100$ to remove extreme outliers and focus on normal trips
plt.grid(axis="y", alpha=0.75)
plt.show()

# 5.5 Histogram of Fare Amount (0-100)
plt.figure(figsize=(10, 6))
plt.hist(df["fare_amount"], bins=100, color="skyblue", edgecolor="black")
plt.title("Distribution of Fare Amount", fontsize=14)
plt.xlabel("Fare Amount ($)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xlim(0, 100)  # cap at 100$ to remove extreme outliers and focus on normal trips
plt.grid(axis="y", alpha=0.75)
plt.show()

# 5.6 Bar Plot of Passenger Counts
passenger_counts = df["passenger_count"].value_counts()
plt.bar(passenger_counts.index, passenger_counts.values)
plt.xlabel("Passenger Count")
plt.ylabel("Frequency")
plt.title("Distribution of Passenger Counts")
plt.show()

# 5.7 Line Plot of Fare Amount over Time
fare_over_time = df.groupby("year")["fare_amount"].mean()
plt.plot(fare_over_time.index, fare_over_time.values)
plt.xlabel("Date")
plt.ylabel("Average Fare Amount")
plt.title("Fare Amount over Time")
plt.xticks(rotation=45)
plt.show()

# 6 No need for feature scaling since i'll be using Tree-based models (Decision Trees, Random Forest and XGBoost) later on.

# Downsampling from approxiamtely 52 million to 5 million rows
# Define target for stratification (bin fare_amount into categories)
num_bins = 50  # higher = finer distribution control
df["fare_bin"] = pd.qcut(df["fare_amount"], q=num_bins, duplicates="drop")

# Stratified sampling to 5 million rows
df_sampled, _ = train_test_split(
    df, train_size=2_000_000, stratify=df["fare_bin"], random_state=42
)

# Drop helper column
df_sampled = df_sampled.drop(columns=["fare_bin"])

print("Original rows:", len(df))
print("Downsampled rows:", len(df_sampled))

# 6 Save DataFrames as CSV
df_sampled.to_csv("../processed/train_models_sel.csv", index=False)
df.to_csv("../processed/train_fit.csv", index=False)
