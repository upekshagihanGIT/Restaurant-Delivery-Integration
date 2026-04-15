# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import requests
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# %%
script_start_time = time.time()

df = pd.read_csv("train.csv")

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
start_time = pd.to_datetime("00:01:01").time()
end_time = pd.to_datetime("23:59:59").time()
start_date = pd.to_datetime("2022-03-01")
end_date = pd.to_datetime("2022-04-06")
full_date_range = pd.date_range(start=start_date, end=end_date)
restaurant_latitude = temp_df["Restaurant_latitude"].mean()
restaurant_longitude = temp_df["Restaurant_longitude"].mean()

# %%
main_df = df[
    (df["Restaurant_latitude"].isin(temp_df["Restaurant_latitude"])) &
    (df["Restaurant_longitude"].isin(temp_df["Restaurant_longitude"])) &
    (df["Time_Orderd"].dt.time.between(start_time, end_time)) &
    (df["Order_Date"].between(start_date, end_date))
]

# %%
scaler1 = StandardScaler()
main_df[["Scaled_Customer_Latitude", "Scaled_Customer_Longitude"]] = scaler1.fit_transform(main_df[["Delivery_location_latitude", "Delivery_location_longitude"]])

# %%
def find_best_k(data, max_k):
    silhouette_scores = []
    kmeans_labels = []
    for k in range(2, max_k+1):
        kmeans_optimal = KMeans(n_clusters=k)
        kmeans_optimal.fit(data)
        kmeans_labels.append((k, kmeans_optimal.labels_))
        silhouette_scores.append((k, silhouette_score(data, kmeans_optimal.labels_)))
    return silhouette_scores, kmeans_labels

# %%
silhouette_scores, kmeans_labels = find_best_k(main_df[["Scaled_Customer_Latitude", "Scaled_Customer_Longitude"]], 10)
best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
best_labels = next(lbl for k, lbl in kmeans_labels if k == best_k)
main_df["K-Means Label"] = best_labels

# %%
plt.scatter(x=main_df["Delivery_location_latitude"], y=main_df["Delivery_location_longitude"], c=main_df["K-Means Label"])
plt.xlabel("Customer Location Latitudes")
plt.ylabel("Customer Location Longitudes")
plt.savefig("Cluster.png")
plt.close()

# %%
cluster_df = main_df[main_df["K-Means Label"] == 1]
cluster_df.to_csv("cluster_df.csv")

# %%
cluster_df = main_df[main_df["K-Means Label"] == 1]
cluster_df = cluster_df.groupby(by="Order_Date").size().reset_index(name="Number of Deliveries")
cluster_df.sort_values(by="Order_Date", inplace=True)
cluster_df.set_index(keys="Order_Date", inplace=True)
cluster_df = cluster_df.reindex(labels=full_date_range, fill_value=0)
cluster_df.reset_index(inplace=True)
cluster_df.columns = ["Order_Date", "Number of Deliveries"]

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
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
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
            torch.zeros(1, batch_size, self.hidden_size),
            torch.zeros(1, batch_size, self.hidden_size)
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

combinations = itertools.product(hidden_sizes, learning_rates, epochs_list) 

# %%
best_accuracy = float('inf')
best_params = (None, None, None)

# %%
for hidden_size, lr, epochs in combinations:
    if (hidden_size == 512) and (lr == 0.007278953843983146) and (epochs == 400):
        print(f"Training with hidden_size={hidden_size}, lr={lr}, epochs={epochs}")
        model = LSTMModel(1, hidden_size, 1)
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
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()} - Val Loss: {val_loss.item()}")

        if val_loss.item() < best_accuracy:
            best_accuracy = val_loss.item()
            best_params = (hidden_size, lr, epochs)

print(f"Best parameters: hidden_size={best_params[0]}, lr={best_params[1]}, epochs={best_params[2]} with loss={best_accuracy}")

# %%
o_model = LSTMModel(1, best_params[0], 1)
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
    print(f"Test Loss: {o_loss.item()}")

# %%
plt.plot(o_predictions.numpy(), label="Predicted Values", marker="x")
plt.plot(y_test.numpy(), label="Actual Values", marker="o")
plt.savefig("scaled_output.png")
plt.close()

# %%
o_predictions_org = scaler_y.inverse_transform(o_predictions.numpy())
y_test_org = scaler_y.inverse_transform(y_test.numpy())

# %%
plt.plot(o_predictions_org, label="Predicted Values", marker="x")
plt.plot(y_test_org, label="Actual Values", marker="o")
plt.savefig("original_output.png")
plt.close()

df2 = pd.read_csv("cluster_df.csv")

# %%
type_of_order_mapping = {
    "Buffet":4,
    "Drinks":2,
    "Meal":3,
    "Snack":1
}
max_vehicles = 100

# %%
cutoff_time = pd.to_datetime("23:19:59").time()
random_df = df2.sample(n=int(o_predictions_org.max()))
random_df = random_df[["Delivery_location_latitude", "Delivery_location_longitude", "Time_Orderd", "Type_of_order"]]
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

# %%
locations = []
timewindows = []
demands = []

restaurant_opening_time = pd.to_datetime("10:00:00").time()
restaurant_closing_time = (pd.to_datetime("23:59:59") + pd.Timedelta(minutes=40)).time()
API_key = "AIzaSyA3fBBiDxW36wCNjLUsxVOnwG5RJTUNQlI"

locations.append((restaurant_latitude, restaurant_longitude))

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
        24*3600,
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
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(60)

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
        return None, None

# %%
print(f"Demands: {demands}")

# %%
best_params = (None, None, None)
best_objective = float("inf")

# %%
duration_matrix = []
distance_matrix = []
duration_matrix, distance_matrix = get_full_matrix(locations, API_key)

print(duration_matrix)
print(distance_matrix)

# %%
vehicle_capacity_list = [50, 130]
max_distance_list = [60000, 90000]
max_vehicles_list = np.linspace(1, max_vehicles, max_vehicles, dtype=int).tolist()

combinations = itertools.product(vehicle_capacity_list, max_distance_list, max_vehicles_list)

# %%
for vehicle_capacity, max_distance, num_vehicles in combinations:
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
    if (objective_value != None) and (objective_value < best_objective):
        print(f"Objective Value: {objective_value}")
        print(routes)
        best_objective = objective_value
        best_params = (vehicle_capacity, max_distance, num_vehicles)  

# %%
if all(elm is None for elm in best_params):
    raise SystemExit

# %%
print(best_params)

# %%
routes, objective_value = solve_vrptw_with_constraints(
        duration_matrix,
        distance_matrix,
        timewindows,
        demands,
        best_params[0],
        best_params[1],
        best_params[2]
    )

if routes:
        print(f"Objective Value: {objective_value}")
        for route_id, route in enumerate(routes):
            print(f"Route for vehicle {route_id + 1}:")
            for stop in route:
                print(f"Location: {stop[0]}, "
                      f"Arrival Window: [{stop[1]}, {stop[2]}], "
                      f"Load: {stop[3]}, "
                      f"Distance: {stop[4]} meters")

script_end_time = time.time()
script_execution_time = script_end_time - script_start_time
print(f"Execution Time: {script_execution_time/60}")