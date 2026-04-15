# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# %%
np.random.seed(42)
script_start_time = time.time()

df = pd.read_csv("A_train.csv")
#df = pd.read_csv(r"C:\Users\upeks\OneDrive - Quantrax\Research\Final\FullRun\Single Restaurant\A_train.csv")

# %%
start_time = pd.to_datetime("00:01:01").time()
end_time = pd.to_datetime("23:59:59").time()
start_date = pd.to_datetime("2022-03-01")
end_date = pd.to_datetime("2022-04-06")
full_date_range = pd.date_range(start=start_date, end=end_date)


# %%
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Time_Orderd"] = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce")
df["Type_of_vehicle"] = df["Type_of_vehicle"].str.strip()
df = df[
    (df["Restaurant_latitude"] != 0.0) &
    (df["Restaurant_longitude"] != 0.0) &
    (df["Delivery_location_latitude"] != 0.0) &
    (df["Delivery_location_longitude"] != 0.0) &
    (df["Type_of_vehicle"].isin(["scooter","motorcycle"])) 
]
df.dropna(inplace=True)

# %%
restaurant_latitude = 26.911378
restaurant_longitude = 75.789034

print(f"Restaurant Latitude: {restaurant_latitude}")
print(f"Restaurant Longitude: {restaurant_longitude}")

# %%
main_df = df[
    (df["Restaurant_latitude"] == restaurant_latitude) &
    (df["Restaurant_longitude"] == restaurant_longitude) &
    (df["Time_Orderd"].dt.time.between(start_time, end_time)) &
    (df["Order_Date"].between(start_date, end_date))
]
total_deliveries = len(main_df)
print(f"Total Deliveries: {total_deliveries}")

# %%
main_df.to_csv("cl_org_dataset.csv", index=False)

# %%
def add_noise(coordinate):
    return coordinate + np.random.uniform(-0.05, 0.05)

main_df["Delivery_location_latitude"] = main_df["Delivery_location_latitude"].apply(add_noise)
main_df["Delivery_location_longitude"] = main_df["Delivery_location_longitude"].apply(add_noise)

# %%
main_df.to_csv("cl_org_dataset_noise.csv", index=False)
# %%
scaler1 = StandardScaler()
main_df[["Scaled_Customer_Latitude", "Scaled_Customer_Longitude"]] = scaler1.fit_transform(main_df[["Delivery_location_latitude", "Delivery_location_longitude"]])

# %%
start_k = 2
max_k = 10+1

def find_best_k(data, max_k):
    silhouette_scores = []
    kmeans_labels = []
    for k in range(start_k, max_k+1):
        kmeans_optimal = KMeans(n_clusters=k, n_init="auto", random_state=42)
        kmeans_optimal.fit(data)
        kmeans_labels.append((k, kmeans_optimal.labels_))
        silhouette_scores.append((k, silhouette_score(data, kmeans_optimal.labels_)))
    return silhouette_scores, kmeans_labels

# %%
silhouette_scores, kmeans_labels = find_best_k(main_df[["Scaled_Customer_Latitude", "Scaled_Customer_Longitude"]], max_k)

pd.DataFrame(
    {
        "K": [ss[0] for ss in silhouette_scores],
        "Silhouette scores": [float(ss[1]) for ss in silhouette_scores]
    }
).to_csv(f"cl_silhouette_scores.csv", index=False)

best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
print(f"Best K Value: {best_k}")
best_labels = next(lbl for k, lbl in kmeans_labels if k == best_k)
main_df["K-Means Label"] = best_labels

# %%
main_df.to_csv("cl_main_dataset.csv", index=False)

# %%
fig, ax = plt.subplots()
scatter = ax.scatter(
    x=main_df["Delivery_location_latitude"], 
    y=main_df["Delivery_location_longitude"], 
    c=main_df["K-Means Label"],
    )
legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="Clusters")
ax.add_artist(legend1)

plt.scatter(restaurant_latitude, restaurant_longitude, color="red",edgecolors="black", label="Restaurant", s=75)
plt.legend(loc="lower right")

plt.xlabel("Delivery Location Latitudes", fontsize=12)
plt.ylabel("Delivery Location Longitudes", fontsize=12)

plt.grid(True)
plt.savefig("cl_clusters.png", dpi=300, bbox_inches="tight")
plt.close()

# %%
script_end_time = time.time()
script_execution_time = script_end_time - script_start_time
print(f"Execution Time: {script_execution_time/60} minutes")