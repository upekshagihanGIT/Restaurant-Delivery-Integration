import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

import requests
import time
import copy
import random

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

import folium

script_start_time = time.time()
# %%
restaurant_latitude = 26.911378
restaurant_longitude = 75.789034

max_vehicles = 8

API_key = "AIzaSyA3fBBiDxW36wCNjLUsxVOnwG5RJTUNQlI"

# %%
vehicle_type_count_list = []
objective_list = []

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
    search_parameters.time_limit.seconds = 30

    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        objective_value = solution.ObjectiveValue()
        routes = []
        total_distance = 0
        total_load = 0
        for vehicle_id in range(vehicle_count):
            route = []
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                load_var = capacity_dimension.CumulVar(index)
                distance_var = distance_dimension.CumulVar(index)
                time_min = solution.Min(time_var)
                time_max = solution.Max(time_var)
                load = solution.Value(load_var)
                distance = solution.Value(distance_var)
                route.append((node_index, time_min, time_max, load, distance))
                index = solution.Value(routing.NextVar(index))
            node_index = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            load_var = capacity_dimension.CumulVar(index)
            distance_var = distance_dimension.CumulVar(index)
            time_min = solution.Min(time_var)
            time_max = solution.Max(time_var)
            load = solution.Value(load_var)
            distance = solution.Value(distance_var)
            route.append((node_index, time_min, time_max, load, distance))   
            if len(route) > 2:
                routes.append(route)
                total_distance += distance
                total_load += load 
        print(f"Objective Value: {objective_value}")
        print(f"Total Distance: {total_distance}")
        print(f"Total Load: {total_load}")
        print("Routes: ")
        print(routes)
        return routes, objective_value, total_distance, total_load
    else:
        print("No Solution!")
        return None, None, None, None

def export_matrices(duration_matrix, distance_matrix, demands):
    labels = [f"Location {i}" for i in range(len(duration_matrix))]

    duration_df = pd.DataFrame(duration_matrix, index=labels, columns=labels)
    distance_df = pd.DataFrame(distance_matrix, index=labels, columns=labels)
    demands_df = pd.DataFrame({"Location":labels, "Demand":demands})

    duration_df.to_csv(f"vrp_EVEEZ_cluster{k}_durations.csv", index=True)
    distance_df.to_csv(f"vrp_EVEEZ_cluster{k}_distances.csv", index=True)
    demands_df.to_csv(f"vrp_EVEEZ_cluster{k}_demands.csv", index=False)


# %%
for k in range(2):
    print(f"Processing Cluster {k}...")
    # %%
    random_df = pd.read_csv(f"sg_cluster{k}_selected_vrp.csv")
    # %%
    locations = []
    timewindows = []
    demands = []

    restaurant_opening_time = pd.to_datetime("10:00:00").time()
    restaurant_closing_time = (pd.to_datetime("23:59:59") + pd.Timedelta(minutes=40)).time()

    locations.append((float(restaurant_latitude), float(restaurant_longitude)))

    timewindows.append([
        restaurant_opening_time.hour*3600+restaurant_opening_time.minute*60+restaurant_opening_time.second,
        (24*3600)+restaurant_closing_time.minute*60+restaurant_closing_time.second 
    ])
    demands.append(0)

    # %%
    for _, row in random_df.iterrows():
        locations.append((row["Delivery_location_latitude"], row["Delivery_location_longitude"]))
        timewindows.append([row["Start_seconds"], row["End_seconds"]])
        demands.append(row["Type_of_order"])

    # %%
    print(f"Demands for Cluster {k}: {demands}")

    # %%
    duration_matrix = []
    distance_matrix = []
    duration_matrix, distance_matrix = get_full_matrix(locations, API_key)

    # %%
    for q in range(1, len(timewindows)):
        if timewindows[q][1] < timewindows[q][0] + duration_matrix[0][q]:
            print("Not Feasible time window found!")
            print(f"Start Time: {timewindows[q][0]} End Time: {timewindows[q][1]} Duration: {duration_matrix[0][q]}")
            print(f"End Time should be {timewindows[q][0] + duration_matrix[0][q]} seconds")
            timewindows[q][1] = timewindows[q][0] + duration_matrix[0][q]
            print("Changed Time Window to...")
            print(f"Start Time: {timewindows[q][0]} End Time: {timewindows[q][1]} Duration: {duration_matrix[0][q]}")

    # %%
    export_matrices(duration_matrix, distance_matrix, demands)
    
    # %%
    vehicle_capacity_list = [120, 70]
    max_distance_list = [60000, 100000]
    max_vehicles_list = np.linspace(1, max_vehicles, max_vehicles, dtype=int).tolist()
    combinations2 = itertools.product(vehicle_capacity_list, max_distance_list, max_vehicles_list)

    # %%
    objective_list2 = []
    best_params = (None, None, None)
    best_objective = float("inf")
    best_routes = None

    have_vehicle_type1 = False

    veh_type1_count = max_vehicles
    veh_type2_count = max_vehicles

    veh1_objective = float("inf")
    veh2_objective = float("inf")

    # %%
    for vehicle_capacity, max_distance, num_vehicles in combinations2:
        if ((vehicle_capacity == 120 and max_distance == 100000) or
            (vehicle_capacity == 70 and max_distance == 60000)):
            continue
        print(f"Capacity: {vehicle_capacity} | Distance: {max_distance} | Number of Vehicles: {num_vehicles}")
        objective_value = None
        routes = []
        routes, objective_value, total_distance, total_load = solve_vrptw_with_constraints(
            duration_matrix,
            distance_matrix,
            timewindows,
            demands,
            vehicle_capacity,
            max_distance,
            num_vehicles
        )
        objective_list2.append((k, vehicle_capacity, max_distance, num_vehicles, objective_value, total_distance, total_load))
        if (objective_value != None) and routes:
            num_vehicles = len(routes)
            if (vehicle_capacity == 120 and max_distance == 60000):
                if num_vehicles < veh_type1_count:
                    veh1_objective = objective_value
                    veh_type1_count = num_vehicles
                    have_vehicle_type1 = True
                    if (objective_value < best_objective):
                        best_routes = copy.deepcopy(routes)
                        best_objective = objective_value
                        best_params = (vehicle_capacity, max_distance, num_vehicles)
            if (vehicle_capacity == 70 and max_distance == 100000):
                if num_vehicles < veh_type2_count:
                    veh2_objective = objective_value
                    veh_type2_count = num_vehicles
                    if (objective_value < best_objective):
                        best_routes = copy.deepcopy(routes)
                        best_objective = objective_value
                        best_params = (vehicle_capacity, max_distance, num_vehicles)
                    elif objective_value == best_objective and not have_vehicle_type1:
                        best_routes = copy.deepcopy(routes)
                        best_objective = objective_value
                        best_params = (vehicle_capacity, max_distance, num_vehicles)

    # %%
    num_vehicles_comparison = {
        "Cluster":[cp[0] for cp in objective_list2],
        "Capacity":[cp[1] for cp in objective_list2],
        "Distance":[cp[2] for cp in objective_list2],
        "Number of Vehicles":[cp[3] for cp in objective_list2],
        "Objective":[cp[4] for cp in objective_list2],
        "Total Distance of Route": [cp[5] for cp in objective_list2],
        "Total Load of Route": [cp[6] for cp in objective_list2]
    }
    pd.DataFrame(num_vehicles_comparison).to_csv(f"vrp_EVEEZ_cluster{k}_num_of_vehicles.csv", index=False)
    # %%
    if all(elm is None for elm in best_params):
        print(f"No Solution for Cluster {k}")
        continue

    # %%
    print(f"Best Parameters: Vehicle Capacity={best_params[0]} Maximum Distance={best_params[1]} Number of Vehicles={best_params[2]}")

    # %%
    vehicle_type_count_list.append((k, veh_type1_count, veh_type2_count))
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
            
            # x_coords.append(locations[0][0])
            # y_coords.append(locations[0][1])

            plt.plot(x_coords, 
                     y_coords, 
                     marker='o',
                     markersize=6,
                     linewidth=2.5, 
                     color=colors[vehicle_id],
                     alpha=0.8, 
                     label=f"Vehicle {vehicle_id+1}"
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
            label="Restaurant", 
            s=150, 
            zorder=5
        )
        plt.text(
            locations[0][0],
            locations[0][1],
            "",
            fontsize=12,
            verticalalignment="bottom",
            fontweight="bold"
        )

        plt.xlabel("Delivery Location Latitudes", fontsize=14)
        plt.ylabel("Delivery Location Longitudes", fontsize=14)
        plt.legend(loc="best", fontsize=12)
        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"vrp_EVEEZ_routes_map_cluster{k}.png", dpi=300, bbox_inches="tight")
        plt.close()
            
    # %%
    colors_follium, colors_matplotlib = generate_colors(len(routes))
    vrp_map = plot_routes_follium(routes, locations, colors_follium)
    vrp_map.save(f"vrp_EVEEZ_routes_map_cluster{k}.html")
    plot_routes_pyplot(routes, locations, colors_matplotlib)

# %%
vehicle_type_data = {
        "Clusters":[vhcnt[0] for vhcnt in vehicle_type_count_list],
        "Vehicle Type 1":[vhcnt[1] for vhcnt in vehicle_type_count_list],
        "Objective - Vehicle Type 1":[objv[0] for objv in objective_list],
        "Vehicle Type 2":[vhcnt[2] for vhcnt in vehicle_type_count_list],
        "Objective - Vehicle Type 2":[objv[1] for objv in objective_list],
    }
pd.DataFrame(vehicle_type_data).to_csv("vrp_EVEEZ_vehicle_type_analysis.csv", index=False)

# %%
script_end_time = time.time()
script_execution_time = script_end_time - script_start_time
print(f"Execution Time: {script_execution_time/60} minutes")