# %%
import numpy as np
import pandas as pd
import requests
import itertools

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# %%
df = pd.read_csv("cluster_df.csv")

# %%
type_of_order_mapping = {
    "Buffet":4,
    "Drinks":2,
    "Meal":3,
    "Snack":1
}
max_vehicles = 50

# %%
cutoff_time = pd.to_datetime("23:19:59").time()
random_df = df.sample(n=60)
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
random_df

# %%
locations = []
timewindows = []
demands = []

restaurant_opening_time = pd.to_datetime("10:00:00").time()
restaurant_closing_time = (pd.to_datetime("23:59:59") + pd.Timedelta(minutes=40)).time()
API_key = "AIzaSyA8-2qB0oHHBSWFFCyfCXcmHxFBpf8lUmo"

locations.append((
    float(random_df["Restaurant_latitude"].mean()),
    float(random_df["Restaurant_longitude"].mean())
))

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


