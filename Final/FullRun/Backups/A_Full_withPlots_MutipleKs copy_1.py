# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

import requests
import time
import copy
import random

from scipy.stats import multivariate_normal, pearsonr

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

import folium

# %%
script_start_time = time.time()

test_prediction = 5
max_vehicles = 8
API_key = "AIzaSyA3fBBiDxW36wCNjLUsxVOnwG5RJTUNQlI"

df = pd.read_csv("A _train.csv")
# df = pd.read_csv(r"C:\Users\upeks\OneDrive - Quantrax\Research\Final\FullRun\A _train.csv")

# %%
start_time = pd.to_datetime("00:01:01").time()
end_time = pd.to_datetime("23:59:59").time()
start_date = pd.to_datetime("2022-03-01")
end_date = pd.to_datetime("2022-04-06")
full_date_range = pd.date_range(start=start_date, end=end_date)

# %%
type_of_order_mapping = {
    "Buffet":4,
    "Drinks":2,
    "Meal":3,
    "Snack":1
}

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
temp_df = df[["Restaurant_latitude", "Restaurant_longitude"]]
temp_df = temp_df.groupby(by=["Restaurant_latitude", "Restaurant_longitude"]).size().reset_index(name="Count")
temp_df.sort_values(by="Count", ascending=False, inplace=True)
temp_df = temp_df[
    (temp_df["Restaurant_latitude"].apply(lambda x: 26.0 <= x <= 27.0)) &
    (temp_df["Restaurant_longitude"].apply(lambda x: 75.0 <= x <= 76.0))
]

# %%
restaurant_latitude = temp_df["Restaurant_latitude"].mean()
restaurant_longitude = temp_df["Restaurant_longitude"].mean()

# %%
main_df = df[
    (df["Restaurant_latitude"].isin(temp_df["Restaurant_latitude"])) &
    (df["Restaurant_longitude"].isin(temp_df["Restaurant_longitude"])) &
    (df["Time_Orderd"].dt.time.between(start_time, end_time)) &
    (df["Order_Date"].between(start_date, end_date))
]
total_deliveries = len(main_df)

# %%
scaler1 = StandardScaler()
main_df[["Scaled_Customer_Latitude", "Scaled_Customer_Longitude"]] = scaler1.fit_transform(main_df[["Delivery_location_latitude", "Delivery_location_longitude"]])

# %%
start_k = 2

def find_best_k(data, max_k):
    silhouette_scores = []
    kmeans_labels = []
    for k in range(start_k, max_k+1):
        kmeans_optimal = KMeans(n_clusters=k)
        kmeans_optimal.fit(data)
        kmeans_labels.append((k, kmeans_optimal.labels_))
        silhouette_scores.append((k, silhouette_score(data, kmeans_optimal.labels_)))
    return silhouette_scores, kmeans_labels

# %%
silhouette_scores, kmeans_labels = find_best_k(main_df[["Scaled_Customer_Latitude", "Scaled_Customer_Longitude"]], 10)

pd.DataFrame(
    {
        "K": [ss[0] for ss in silhouette_scores],
        "Silhouette scores": [float(ss[0]) for ss in silhouette_scores]
    }
).to_csv(f"silhouette_scores.csv", index=False)

best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
best_labels = next(lbl for k, lbl in kmeans_labels if k == best_k)
main_df["K-Means Label"] = best_labels

# %%
unique_labels = main_df["K-Means Label"].unique()
colors_labels = plt.cm.get_cmap("viridis", len(unique_labels))

cluster_scatter = plt.scatter(
    x=main_df["Delivery_location_latitude"], 
    y=main_df["Delivery_location_longitude"], 
    c=main_df["K-Means Label"]
    )

cbar = plt.colorbar(cluster_scatter)
cbar.set_label("K-Means Cluster Labels")
cbar.set_ticks(unique_labels)

plt.scatter(restaurant_latitude, restaurant_longitude, color="red",edgecolors="black", label="Depot", s=150, zorder=5)
plt.text(restaurant_latitude, restaurant_longitude, "  Depot",fontsize=12, horizontalalignment="left",fontweight="bold")

plt.xlabel("Customer Location Latitudes", fontsize=12)
plt.ylabel("Customer Location Longitudes", fontsize=12)
plt.title("Customer Clusters based on Delivery Locations", fontsize=14, fontweight="bold")

plt.savefig("cluster.png", dpi=300, bbox_inches="tight")
plt.close()

# %%
cluster_list = []
vehicle_type_count_list = []
objective_list = []
historical_weight_list = []

# %%
for k in range(best_k):

    cluster_list.append(f"Cluster {k}")
    print(f"Processing Cluster: {k}")

# %%
    cluster_df = main_df[main_df["K-Means Label"] == k]
    total_deliveries_per_cluster = len(cluster_df)

    vrp_df = main_df[main_df["K-Means Label"] == k]
    vrp_df.to_csv(f"cluster{k}_dataset.csv", index=False)

    # %%
    cluster_df = cluster_df.groupby(by="Order_Date").size().reset_index(name="Number of Deliveries")
    cluster_df.sort_values(by="Order_Date", inplace=True)
    cluster_df.set_index(keys="Order_Date", inplace=True)
    cluster_df = cluster_df.reindex(labels=full_date_range, fill_value=0)
    cluster_df.reset_index(inplace=True)
    cluster_df.columns = ["Order_Date", "Number of Deliveries"]
      
    # %%
    sns.set_style("whitegrid")
    cm_deliveries = plt.cm.viridis_r(cluster_df["Number of Deliveries"]/cluster_df["Number of Deliveries"].max())
    plt.figure(figsize=(12, 6))
    plt.bar(cluster_df["Order_Date"],cluster_df["Number of Deliveries"], color=cm_deliveries, edgecolor="black")
    plt.title(f"Number of Deliveries - Cluster {k}", fontsize=14, fontweight="bold")
    plt.xlabel("Order Dates", fontsize=12)
    plt.ylabel("Number of Deliveries", fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"num_of_deliveries_cluster{k}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # %%
    data = cluster_df["Number of Deliveries"].to_numpy()

    # %%
    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data)-sequence_length):
            seq = data[i:i+sequence_length]
            lbl = data[i+sequence_length]
            sequences.append((seq, lbl))
        return sequences

    # %%
    sequences = create_sequences(data, 5)

    # %%
    X = np.array([seq[0] for seq in sequences])
    y = np.array([[lbl[1]] for lbl in sequences])

    # %%
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # %%
    X_train, X_next, y_train, y_next = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_next, y_next, test_size=0.5, shuffle=False)

    # %%
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # %%
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            self.initialize_weights()

        def initialize_weights(self):
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LSTM):
                    for name, param in layer.named_parameters():
                        if "weight" in name:
                            nn.init.xavier_uniform_(param.data)
                        elif "bias" in name:
                            nn.init.zeros_(param.data)
        
        def init_hidden(self, batch_size):
            return (
                torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size)
            )
        
        def forward(self, x):
            batch_size = x.size(0)
            hidden = self.init_hidden(batch_size)
            out, _ = self.lstm(x, hidden)
            output = self.fc(out[:, -1, :])
            return output

    # %%
    hidden_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    learning_rates = np.logspace(-4, -1, 30, endpoint=True).tolist()
    epochs_list = np.linspace(10, 500, 50).astype(int).tolist()
    num_layers_list = [1]

    pd.DataFrame(
        {
        "Hidden Sizes": pd.Series(hidden_sizes),
        "Learning Rates": pd.Series(learning_rates),
        "Epochs": pd.Series(epochs_list),
        "Number of Layers": pd.Series(num_layers_list)
        }
    ).to_csv(f"cluster{k}_lstm_hyperparameters.csv", index=False)

    combinations1 = itertools.product(hidden_sizes, learning_rates, epochs_list, num_layers_list) 

    # %%
    best_accuracy = float('inf')
    best_params = (None, None, None, None)

    # %%
    for hidden_size, lr, epochs, num_layers in combinations1:
        if (hidden_size == 2) and (lr == 0.0001) and (epochs == 100) and (num_layers == 1):
            print(f"Training with hidden_size={hidden_size}, lr={lr}, epochs={epochs}, num_layers={num_layers}")
            model = LSTMModel(1, hidden_size, num_layers, 1)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            for epoch in range(epochs):         
                model.train()
                output = model(X_train.unsqueeze(-1))
                loss = criterion(output, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_output = model(X_val.unsqueeze(-1))
                    val_loss = criterion(val_output, y_val)
                    # print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()} - Val Loss: {val_loss.item()}")
                    if val_loss.item() < best_accuracy:
                        best_accuracy = val_loss.item()
                        best_params = (hidden_size, lr, epochs, num_layers)

    pd.DataFrame(
        {
            "Best Hidden Size": [best_params[0]],
            "Best Learning Rate": [best_params[1]],
            "Best Epochs": [best_params[2]],
            "Best Number of Layers": [best_params[3]],
            "Best Loss": [best_accuracy]
        }
    ).to_csv(f"cluster{k}_best_parameters.csv", index=False)
    print(f"Best parameters: hidden_size={best_params[0]}, lr={best_params[1]}, epochs={best_params[2]}, num_layers={best_params[3]} with loss={best_accuracy}")

    # %%
    o_model = LSTMModel(1, best_params[0], best_params[3], 1)
    o_optimizer = torch.optim.Adam(o_model.parameters(), lr=best_params[1])
    o_criterion = nn.MSELoss()
    o_epochs = best_params[2]

    # %%
    for epoch in range(o_epochs):
        o_model.train()
        o_output = o_model(X_train.unsqueeze(-1))
        o_loss = o_criterion(o_output, y_train)
        o_optimizer.zero_grad()
        o_loss.backward()
        o_optimizer.step()
        print(F"Epoch: {epoch+1} Loss: {o_loss.item()}")

    # %%
    o_model.eval()
    with torch.no_grad():
        o_predictions = o_model(X_test.unsqueeze(-1))
        o_loss = o_criterion(o_predictions, y_test)
        o_pearson_corr, _ = pearsonr(o_predictions.numpy(), y_test.numpy())
        print(f"MSE Loss for Cluster {k}: {o_loss.item()}")
        print(f"Pearson's Correlation Coefficient for Cluster {k}: {o_pearson_corr.item()}")

    pd.DataFrame(
        {
            "MSE Loss": [o_loss.item()],
            "Pearson's Correlation Coefficient": [o_pearson_corr.item()]
        }
    ).to_csv(f"cluster{k}_prediction_metrics.csv", index=False)

    # %%
    o_predictions_arr = np.clip(o_predictions.numpy(), 0, None)

    plt.figure(figsize=(10, 6))
    plt.plot(o_predictions_arr, label="Predicted Values", marker="x", linestyle="--", color="tab:blue", markersize=6)
    plt.plot(y_test.numpy(), label="Actual Values", marker="o", linestyle="--", color="tab:orange", markersize=6)

    plt.title(f"Scaled Prediction for Cluster {k}", fontsize=14, fontweight="bold")
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Scaled Value", fontsize=12)

    plt.xticks(ticks=np.arange(len(o_predictions_arr)), labels=np.arange(1, len(o_predictions_arr)+1))

    plt.legend(loc="best", fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(f"scaled_prediction_cluster{k}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # %%
    o_predictions_org = scaler_y.inverse_transform(o_predictions.numpy())
    y_test_org = scaler_y.inverse_transform(y_test.numpy())

    # %%
    o_predictions_org_arr = np.clip(o_predictions_org, 0, None)

    plt.figure(figsize=(10, 6))
    plt.plot(o_predictions_org_arr, label="Predicted Values", marker="x", linestyle="--", color="tab:blue", markersize=6)
    plt.plot(y_test_org, label="Actual Values", marker="o", linestyle="--", color="tab:orange", markersize=6)

    plt.title(f"Original Prediction for Cluster {k}", fontsize=14, fontweight="bold")
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Original Value", fontsize=12)

    plt.xticks(ticks=np.arange(len(o_predictions_org_arr)), labels=np.arange(1, len(o_predictions_arr)+1))

    plt.legend(loc="best", fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(f"original_prediction_cluster{k}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # %%
    print(f"Maximum Prediction for Cluster {k}: {int(o_predictions_org.max())}")

    # %%
    def generate_synthetic_points(centroid, spread, num_points):
        cov_matrix = np.array([[spread, 0], [0, spread]])
        synthetic_points = multivariate_normal.rvs(mean=centroid, cov=cov_matrix, size=num_points)
        return synthetic_points
    
    # %%
    def hybrid_location_generation(vrp_df, cluster_centroid, historical_weight):
        predicted_deliveries = test_prediction
        num_historical = int(predicted_deliveries*historical_weight)
        num_synthetic = predicted_deliveries - num_historical

        historical_df = vrp_df.sample(n=num_historical, replace=True)

        spread = 0.01

        synthetic_points = generate_synthetic_points(
            centroid=[cluster_centroid["Delivery_location_latitude"], cluster_centroid["Delivery_location_longitude"]],
            spread=spread,
            num_points=num_synthetic
        )
        synthetic_df = pd.DataFrame(synthetic_points, columns=["Delivery_location_latitude", "Delivery_location_longitude"])
        historical_dates_times = vrp_df[['Time_Orderd', 'Type_of_order']].sample(n=num_synthetic, replace=True)
        synthetic_df['Time_Orderd'] = historical_dates_times['Time_Orderd'].values
        synthetic_df['Type_of_order'] = historical_dates_times['Type_of_order'].values

        final_df = pd.concat([historical_df.reset_index(drop=True), synthetic_df.reset_index(drop=True)], axis=0)

        return final_df
    
    def get_full_matrix(locations, API_key):
        max_batch_size = 10  
        n = len(locations)
        duration_matrix = np.zeros((n, n), dtype=int)  
        distance_matrix = np.zeros((n, n), dtype=int)

        for i in range(0, n, max_batch_size): 
            for j in range(0, n, max_batch_size): 
                batch_origins = locations[i:i + max_batch_size]
                batch_destinations = locations[j:j + max_batch_size]
                origins = "|".join([f"{lat},{long}" for lat, long in batch_origins])
                destinations = "|".join([f"{lat},{long}" for lat, long in batch_destinations])

                url = f"https://maps.googleapis.com/maps/api/distancematrix/json"
                params = {
                    "origins": origins,
                    "destinations": destinations,
                    "key": API_key
                }
                response = requests.get(url, params=params)
                if response.status_code != 200:
                    raise ValueError("Error with Distance Matrix API: " + response.text)
                
                data = response.json()

                for origin_idx, row in enumerate(data["rows"]):
                    for dest_idx, elem in enumerate(row["elements"]):
                        # Fill in the corresponding cells in the matrices
                        duration_matrix[i + origin_idx, j + dest_idx] = elem["duration"]["value"]
                        distance_matrix[i + origin_idx, j + dest_idx] = elem["distance"]["value"]

        return duration_matrix.tolist(), distance_matrix.tolist()

    # %%
    def solve_vrptw_with_constraints(
            duration_matrix, distance_matrix, time_windows, demands, vehicle_capacity, max_distance, vehicle_count=1 
    ):    
        manager = pywrapcp.RoutingIndexManager(len(duration_matrix), vehicle_count, 0)
        routing = pywrapcp.RoutingModel(manager)

        def duration_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return duration_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(duration_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        routing.AddDimension(
            transit_callback_index,
            300,
            time_windows[0][1],
            False,
            "Time"
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        for location_idx, (start, end) in enumerate(time_windows):
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(start, end)

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands[from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            [vehicle_capacity] * vehicle_count,
            True,
            "Capacity"
        )

        capacity_dimension = routing.GetDimensionOrDie("Capacity")

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        
        distance_callback_index = routing.RegisterTransitCallback(distance_callback)

        routing.AddDimension(
            distance_callback_index,
            0,
            max_distance,
            True,
            "Distance"
        )

        distance_dimension = routing.GetDimensionOrDie("Distance")

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        )
        search_parameters.time_limit.seconds = 500

        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            objective_value = solution.ObjectiveValue()
            routes = []
            for vehicle_id in range(vehicle_count):
                route = []
                index = routing.Start(vehicle_id)
                total_distance = 0
                total_load = 0
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    time_var = time_dimension.CumulVar(index)
                    load_var = capacity_dimension.CumulVar(index)
                    distance_var = distance_dimension.CumulVar(index)
                    time_min = solution.Min(time_var)
                    time_max = solution.Max(time_var)
                    load = solution.Value(load_var)
                    distance = solution.Value(distance_var)
                    total_distance += distance
                    total_load += load
                    route.append((node_index, time_min, time_max, load, distance))
                    index = solution.Value(routing.NextVar(index))
                if len(route) > 1:
                    routes.append(route)  
            return routes, objective_value
        else:
            print("No Solution!")
            return None, None

    def export_matrices(duration_matrix, distance_matrix, demands):
        labels = [f"Location {i}" for i in range(len(duration_matrix))]

        duration_df = pd.DataFrame(duration_matrix, index=labels, columns=labels)
        distance_df = pd.DataFrame(distance_matrix, index=labels, columns=labels)
        demands_df = pd.DataFrame({"Location":labels, "Demand":demands})

        duration_df.to_csv(f"cluster{k}_durations.csv", index=True)
        distance_df.to_csv(f"cluster{k}_distances.csv", index=True)
        demands_df.to_csv(f"cluster{k}_demands.csv", index=False)

    # %%
    all_deliveries_feasible = False
    while not all_deliveries_feasible:
        # %%
        random_df = vrp_df[["Delivery_location_latitude", "Delivery_location_longitude", "Time_Orderd", "Type_of_order"]]
        cluster_centroid = random_df[["Delivery_location_latitude", "Delivery_location_longitude"]].mean()
        historical_weight = total_deliveries_per_cluster/total_deliveries
        random_df = hybrid_location_generation(random_df, cluster_centroid, historical_weight)

        #%%
        all_deliveries_feasible = True # assuming generated deliveries are feasible

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
        random_df.to_csv(f"cluster{k}_selected_vrp.csv", index=False)

        # %%
        locations = []
        timewindows = []
        demands = []

        restaurant_opening_time = pd.to_datetime("10:00:00").time()
        restaurant_closing_time = (pd.to_datetime("23:59:59") + pd.Timedelta(minutes=40)).time()

        locations.append((float(restaurant_latitude), float(restaurant_longitude)))

        timewindows.append((
            restaurant_opening_time.hour*3600+restaurant_opening_time.minute*60+restaurant_opening_time.second,
            (24*3600)+restaurant_closing_time.minute*60+restaurant_closing_time.second 
        ))
        demands.append(0)

        # %%
        for _, row in random_df.iterrows():
            locations.append((row["Delivery_location_latitude"], row["Delivery_location_longitude"]))
            timewindows.append((row["Start_seconds"], row["End_seconds"]))
            demands.append(row["Type_of_order"])

        # %%
        print(f"Demands for Cluster {k}: {demands}")

        # %%
        duration_matrix = []
        distance_matrix = []
        duration_matrix, distance_matrix = get_full_matrix(locations, API_key)

        # %%
        export_matrices(duration_matrix, distance_matrix, demands)

        # %%
        for q in range(1, len(timewindows)):
            if timewindows[q][1] < timewindows[q][0] + duration_matrix[0][q]:
                all_deliveries_feasible = False
                print("Not Feasible time window found!")           
    # %%
    historical_weight_list.append(historical_weight)
    # %%
    vehicle_capacity_list = [50, 130]
    max_distance_list = [60000, 90000]
    max_vehicles_list = np.linspace(1, max_vehicles, max_vehicles, dtype=int).tolist()
    combinations2 = itertools.product(vehicle_capacity_list, max_distance_list, max_vehicles_list)

    # %%
    objective_list2 = []
    best_params = (None, None, None)
    best_objective = float("inf")
    best_routes = None

    veh_type1_count = max_vehicles
    veh_type2_count = max_vehicles

    veh1_objective = float("inf")
    veh2_objective = float("inf")

    # %%
    for vehicle_capacity, max_distance, num_vehicles in combinations2:    
        if (vehicle_capacity == 50 and max_distance == 90000) or (vehicle_capacity == 130 and max_distance == 60000):
            continue
        print(f"Capacity: {vehicle_capacity} | Distance: {max_distance} | Number of Vehicles: {num_vehicles}")
        objective_value = None
        routes = []
        routes, objective_value = solve_vrptw_with_constraints(
            duration_matrix,
            distance_matrix,
            timewindows,
            demands,
            vehicle_capacity,
            max_distance,
            num_vehicles
        )
        objective_list2.append((k, vehicle_capacity, max_distance, num_vehicles, objective_value))
        if (objective_value != None) and routes:
            num_vehicles = len(routes)
            if (vehicle_capacity == 50 and max_distance == 60000):
                if num_vehicles < veh_type1_count:
                    veh1_objective = objective_value
                    veh_type1_count = num_vehicles
            if (vehicle_capacity == 130 and max_distance == 90000):
                if num_vehicles < veh_type2_count:
                    veh2_objective = objective_value
                    veh_type2_count = num_vehicles
            if (objective_value < best_objective):
                print(f"Objective Value: {objective_value}")
                print(routes)
                best_routes = copy.deepcopy(routes)
                best_objective = objective_value
                best_params = (vehicle_capacity, max_distance, num_vehicles)

    # %%
    num_vehicles_comparison = {
        "Cluster":[cp[0] for cp in objective_list2],
        "Capacity":[cp[1] for cp in objective_list2],
        "Distance":[cp[2] for cp in objective_list2],
        "Number of Vehicles":[cp[3] for cp in objective_list2],
        "Objective":[cp[4] for cp in objective_list2]
    }
    pd.DataFrame(num_vehicles_comparison).to_csv(f"cluster{k}_num_of_vehicles.csv", index=False)
    # %%
    if all(elm is None for elm in best_params):
        print(f"No Solution for Cluster {k}")
        continue

    # %%
    print(best_params)

    # %%
    vehicle_type_count_list.append((veh_type1_count, veh_type2_count))
    objective_list.append((veh1_objective, veh2_objective))

    # %%
    if best_routes:
            print(f"Best Route for Cluster {k}")
            print(f"Objective Value: {objective_value}")
            for route_id, route in enumerate(best_routes):
                print(f"Route for vehicle {route_id + 1}:")
                for stop in route:
                    print(f"Location: {stop[0]}, "
                        f"Arrival Window: [{stop[1]}, {stop[2]}], "
                        f"Load: {stop[3]}, "
                        f"Distance: {stop[4]} meters")

    # %%
    def generate_colors(n):
        colors_follium = []
        colors_matplotlib = sns.color_palette("hsv", n)
        for _ in range(n):
            r = random.randint(0, 100)  # Dark red
            g = random.randint(0, 100)  # Dark green
            b = random.randint(0, 100)  # Dark blue
            colors_follium.append(f"rgb({r}, {g}, {b})")
        return colors_follium, colors_matplotlib 

    # %%
    def plot_routes_follium(routes, locations, colors):

        depot_x, depot_y = locations[0]

        m = folium.Map(location=[depot_x, depot_y], zoom_start=12)
        # folium.Marker([depot_x, depot_y], popup="depot", icon=folium.Icon(color="red")).add_to(m)

        for vehicle_id, route in enumerate(routes):
            route_coords = [(locations[stop[0]][0], locations[stop[0]][1]) for stop in route]
            route_coords.append((locations[0]))
            folium.PolyLine(route_coords, color=colors[vehicle_id], weight=4, opacity=0.8).add_to(m)
            for loc_id, arrival, departure, load, distance in route:
                lat, lon = locations[loc_id]
                folium.Marker(
                    [lat, lon],
                    popup=f"Loc: {loc_id}, Arrival: {arrival}, Load: {load}",
                    icon=folium.Icon(color="blue" if loc_id != 0 else "red")
                ).add_to(m)
        return m
    
    # %%
    def plot_routes_pyplot(routes, locations, colors):
        plt.figure(figsize=(12, 7))

        for vehicle_id, route in enumerate(routes):
            x_coords = []
            y_coords = []

            for stop in route:
                loc_id = stop[0]
                x, y = locations[loc_id]
                x_coords.append(x)
                y_coords.append(y)
            
            x_coords.append(locations[0][0])
            y_coords.append(locations[0][1])

            plt.plot(x_coords, 
                     y_coords, 
                     marker='o',
                     markersize=6,
                     linewidth=2.5, 
                     color=colors[vehicle_id],
                     alpha=0.8, 
                     label=f"Vehicle {vehicle_id}"
                     )

            U = np.diff(x_coords)
            V = np.diff(y_coords)
            x_mid = x_coords[:-1]
            y_mid = y_coords[:-1]
            plt.quiver(
                x_mid, 
                y_mid, 
                U, 
                V, 
                angles="xy", 
                scale_units="xy", 
                scale=1.2, 
                color="black", 
                alpha=0.8, 
                width=0.007
            )

        plt.scatter(
            locations[0][0], 
            locations[0][1], 
            color="red",
            edgecolors="black", 
            label="Depot", 
            s=150, 
            zorder=5
        )
        plt.text(
            locations[0][0],
            locations[0][1],
            "  Depot",
            fontsize=12,
            verticalalignment="bottom",
            fontweight="bold"
        )

        plt.title("Optimized Vehicle Routes", fontsize=16, fontweight="bold", pad=15)
        plt.xlabel("Delivery Location Latitudes", fontsize=14)
        plt.ylabel("Delivery Location Longitudes", fontsize=14)
        plt.legend(loc="upper left", fontsize=12)
        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"vrp_routes_map_cluster{k}.png", dpi=300, bbox_inches="tight")
        plt.close()
            
    # %%
    colors_follium, colors_matplotlib = generate_colors(len(routes))
    vrp_map = plot_routes_follium(routes, locations, colors_follium)
    vrp_map.save(f"vrp_routes_map_cluster{k}.html")
    plot_routes_pyplot(routes, locations, colors_matplotlib)


# %%
vehicle_type_data = {
        "Clusters":cluster_list,
        "Vehicle Type 1":[vhcnt[0] for vhcnt in vehicle_type_count_list],
        "Objective - Vehicle Type 1":[objv[0] for objv in objective_list],
        "Vehicle Type 2":[vhcnt[1] for vhcnt in vehicle_type_count_list],
        "Objective - Vehicle Type 2":[objv[1] for objv in objective_list],
    }
pd.DataFrame(vehicle_type_data).to_csv("vehicle_type_analysis.csv", index=False)

pd.DataFrame(
    {
        "Clusters": cluster_list,
        "Historical Weights": historical_weight_list
    }
).to_csv("historical_weights.csv", index=False)

# %%
script_end_time = time.time()
script_execution_time = script_end_time - script_start_time
print(f"Execution Time: {script_execution_time/60} minutes")