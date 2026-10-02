# Logistics Delivery Performance and Route Optimization
# Strategic Planning Project

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("logistics_data.csv")

print("First 5 records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# --------------------------------------------------
# 2. Data Cleaning
# --------------------------------------------------

# Remove duplicate records
df = df.drop_duplicates()

# Convert date columns
df["order_date"] = pd.to_datetime(df["order_date"])
df["delivery_date"] = pd.to_datetime(df["delivery_date"])

# Calculate delivery time
df["delivery_time"] = (
    df["delivery_date"] - df["order_date"]
).dt.days


# --------------------------------------------------
# 3. KPI Analysis
# --------------------------------------------------

# On-Time Delivery Rate
on_time_rate = (
    df["delivery_status"].eq("On-Time").mean() * 100
)

# Average Delivery Time
average_delivery_time = df["delivery_time"].mean()

# Average Distance
average_distance = df["distance"].mean()

# Average Transportation Cost
average_cost = df["transportation_cost"].mean()

print("\n----- Logistics KPIs -----")
print(f"On-Time Delivery Rate: {on_time_rate:.2f}%")
print(f"Average Delivery Time: {average_delivery_time:.2f} days")
print(f"Average Delivery Distance: {average_distance:.2f} km")
print(f"Average Transportation Cost: {average_cost:.2f}")


# --------------------------------------------------
# 4. Exploratory Data Analysis
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["delivery_time"], bins=10)

plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Deliveries")
plt.title("Distribution of Delivery Time")

plt.show()


# --------------------------------------------------
# 5. Distance vs Delivery Time
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["distance"],
    df["delivery_time"]
)

plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (Days)")
plt.title("Distance vs Delivery Time")

plt.show()


# --------------------------------------------------
# 6. Business Interpretation
# --------------------------------------------------

print("\n----- Business Insights -----")

print(
    "The analysis can help identify delayed deliveries, "
    "high-distance routes and inefficient logistics operations."
)

print(
    "Regression can be used to predict delivery time, "
    "while clustering can group similar delivery locations."
)

print(
    "Optimization techniques can later be used to improve "
    "vehicle and route allocation."
)
