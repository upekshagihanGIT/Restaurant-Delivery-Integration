import numpy as np
import pandas as pd

import time

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

np.random.seed(42)

script_start_time = time.time()
# %%
test_prediction = 5
total_deliveries = 172

restaurant_latitude = 26.911378
restaurant_longitude = 75.789034

# %%
type_of_order_mapping = {
    "Buffet":4,
    "Drinks":2,
    "Meal":3,
    "Snack":1
}

# %%
cluster_list = []
historical_weight_list = []

for k in range(2):
    cluster_list.append(k)
    print(f"Processing Cluster {k}...")
    # %%
    vrp_df = pd.read_csv(f"nn_cluster{k}_dataset.csv")
    # %%
    total_deliveries_per_cluster = len(vrp_df)
    # %%
    def generate_synthetic_points(centroid, spread, num_points):
        cov_matrix = np.array([[spread, 0], [0, spread]])
        synthetic_points = multivariate_normal.rvs(mean=centroid, cov=cov_matrix, size=num_points)
        return synthetic_points
    # %%
    def hybrid_location_generation(vrp_df, cluster_centroid, historical_weight):
        predicted_deliveries = test_prediction
        num_historical = int(predicted_deliveries*historical_weight)
        print(f"Historical Proportion: {num_historical}")
        num_synthetic = predicted_deliveries - num_historical
        print(f"Synthetic Proportion: {num_synthetic}")

        historical_df = vrp_df.sample(n=num_historical, replace=True, random_state=42)

        spread = 0.01

        synthetic_points = generate_synthetic_points(
            centroid=[cluster_centroid["Restaurant_latitude"], cluster_centroid["Restaurant_longitude"]],
            spread=spread,
            num_points=num_synthetic
        )
        synthetic_df = pd.DataFrame(synthetic_points, columns=["Delivery_location_latitude", "Delivery_location_longitude"])
        historical_dates_times = vrp_df[['Time_Orderd', 'Type_of_order']].sample(n=num_synthetic, replace=True, random_state=42)
        synthetic_df['Time_Orderd'] = historical_dates_times['Time_Orderd'].values
        synthetic_df['Type_of_order'] = historical_dates_times['Type_of_order'].values

        final_df = pd.concat([historical_df.reset_index(drop=True), synthetic_df.reset_index(drop=True)], axis=0)

        return final_df

    # %%
    random_df = vrp_df[["Delivery_location_latitude", "Delivery_location_longitude", "Time_Orderd", "Type_of_order"]]
    past_locations = vrp_df[["Delivery_location_latitude", "Delivery_location_longitude"]]
    # cluster_centroid = vrp_df[["Restaurant_latitude", "Restaurant_longitude"]].mean()
    cluster_centroid = vrp_df[["Delivery_location_latitude", "Delivery_location_longitude"]].mean()
    historical_weight = total_deliveries_per_cluster/total_deliveries
    historical_weight = total_deliveries_per_cluster/total_deliveries
    historical_weight_list.append(historical_weight)
    random_df = hybrid_location_generation(random_df, cluster_centroid, historical_weight)
    new_locations = random_df[["Delivery_location_latitude", "Delivery_location_longitude"]]

    # %%
    cutoff_time = pd.to_datetime("23:19:59").time()
    
    random_df["Type_of_order"] = random_df["Type_of_order"].str.strip()
    random_df["Type_of_order"] = random_df["Type_of_order"].map(type_of_order_mapping)
    random_df["Time_Orderd"] = pd.to_datetime(random_df["Time_Orderd"])
    random_df["Time_Delivery"] = random_df["Time_Orderd"] + pd.Timedelta(minutes=40)
    random_df["Start_seconds"] = (
        (random_df["Time_Orderd"].dt.hour*3600)+(random_df["Time_Orderd"].dt.minute*60)+(random_df["Time_Orderd"].dt.second) 
    )
    random_df["End_seconds"] = np.where(
        random_df["Time_Orderd"].dt.time <= cutoff_time,
        ((random_df["Time_Delivery"].dt.hour*3600)+(random_df["Time_Delivery"].dt.minute*60)+(random_df["Time_Delivery"].dt.second)),
        ((24*3600)+(random_df["Time_Delivery"].dt.minute*60)+(random_df["Time_Delivery"].dt.second))
    )
    random_df.to_csv(f"sg_cluster{k}_selected_vrp.csv", index=False)

    # %%
    plt.figure(figsize=(10, 6))
    plt.scatter(past_locations["Delivery_location_latitude"], past_locations["Delivery_location_longitude"], label="Past Deliveries")
    plt.scatter(new_locations["Delivery_location_latitude"], new_locations["Delivery_location_longitude"], label="New Deliveries", s=50)
    plt.scatter(restaurant_latitude, restaurant_longitude, color="red",edgecolors="black", label="Restaurant", s=75)
    plt.xlabel("Latitudes", fontsize=12)
    plt.ylabel("Longitudes", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f"sg_cluster{k}_past&new_locations.png", dpi=300, bbox_inches="tight")
    plt.close()

pd.DataFrame(
    {
        "Clusters": cluster_list,
        "Historical Weights": historical_weight_list
    }
).to_csv("sg_historical_weights.csv", index=False)

# %%
script_end_time = time.time()
script_execution_time = script_end_time - script_start_time
print(f"Execution Time: {script_execution_time/60} minutes")