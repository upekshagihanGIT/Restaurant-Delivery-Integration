# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

import time

from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn

# %%
script_start_time = time.time()

EPOCHS = 500

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
main_df.to_csv("nn_org_dataset.csv", index=False)

# %%
def add_noise(coordinate):
    return coordinate + np.random.uniform(-0.05, 0.05)

main_df["Delivery_location_latitude"] = main_df["Delivery_location_latitude"].apply(add_noise)
main_df["Delivery_location_longitude"] = main_df["Delivery_location_longitude"].apply(add_noise)

# %%
main_df.to_csv("nn_org_dataset_noise.csv", index=False)
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
).to_csv(f"silhouette_scores.csv", index=False)

best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
print(f"Best K Value: {best_k}")
best_labels = next(lbl for k, lbl in kmeans_labels if k == best_k)
main_df["K-Means Label"] = best_labels

# %%
main_df.to_csv("nn_main_dataset.csv", index=False)

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
    vrp_df.to_csv(f"nn_cluster{k}_dataset.csv", index=False)

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
    pd.DataFrame(
        {
            "Total Records": [len(X_scaled)],
            "Training": [len(X_train)],
            "Validating": [len(X_val)],
            "Testing": [len(X_test)],
        }
    ).to_csv(f"nn_cluster{k}_dataset_info.csv", index=False)

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
    num_layers_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    pd.DataFrame(
        {
        "Hidden Sizes": pd.Series(hidden_sizes),
        "Learning Rates": pd.Series(learning_rates),
        "Number of Layers": pd.Series(num_layers_list),
        }
    ).to_csv(f"nn_cluster{k}_lstm_hyperparameters.csv", index=False)

    combinations1 = itertools.product(hidden_sizes, learning_rates, num_layers_list) 

    # %%
    best_accuracy = float('inf')
    best_params = (None, None, None, None)

    # %%
    for hidden_size, lr, num_layers in combinations1:
        print(f"Training with hidden_size={hidden_size}, lr={lr}, epochs={EPOCHS}, num_layers={num_layers}")
        model = LSTMModel(1, hidden_size, num_layers, 1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(EPOCHS):         
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
                if val_loss.item() < best_accuracy:
                    best_accuracy = val_loss.item()
                    best_params = (hidden_size, lr, (epoch+1), num_layers)

    pd.DataFrame(
        {
            "Best Hidden Size": [best_params[0]],
            "Best Learning Rate": [best_params[1]],
            "Best Epochs": [best_params[2]],
            "Best Number of Layers": [best_params[3]],
            "Best Loss": [best_accuracy],
        }
    ).to_csv(f"nn_cluster{k}_best_parameters.csv", index=False)
    print(f"Best parameters: hidden_size={best_params[0]}, lr={best_params[1]}, epochs={best_params[2]}, num_layers={best_params[3]}, with loss={best_accuracy}")
    print(best_params)
    # %%
    best_model = LSTMModel(1, best_params[0], best_params[3], 1)
    best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params[1])
    best_criterion = nn.MSELoss()
    best_epochs = best_params[2]

    # %%
    loss_list = []
    for epoch in range(best_epochs):
        best_model.train()
        best_output = best_model(X_train.unsqueeze(-1))
        best_loss = best_criterion(best_output, y_train)
        best_optimizer.zero_grad()
        best_loss.backward()
        best_optimizer.step()
        print(F"Epoch: {epoch+1} Loss: {best_loss.item()}")
        loss_list.append(best_loss.item())

    # %%
    best_model.eval()
    with torch.no_grad():
        best_predictions = best_model(X_test.unsqueeze(-1))
        best_loss = best_criterion(best_predictions, y_test)
        best_pearson_corr, _ = pearsonr(best_predictions.numpy(), y_test.numpy())
        print(f"MSE Loss for Cluster {k}: {best_loss.item()}")
        print(f"Pearson's Correlation Coefficient for Cluster {k}: {best_pearson_corr.item()}")

    pd.DataFrame(
        {
            "MSE Loss": [best_loss.item()],
            "Pearson's Correlation Coefficient": [best_pearson_corr.item()]
        }
    ).to_csv(f"nn_cluster{k}_prediction_metrics.csv", index=False)

    # %%
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(loss_list, label="Loss", color="tab:blue")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"nn_loss_curve_cluster{k}.png", bbox_inches="tight")
    plt.close()
    # %%
    best_predictions_arr = np.clip(best_predictions.numpy(), 0, None)

    plt.figure(figsize=(10, 6))
    plt.plot(best_predictions_arr, label="Predicted Values", marker="x", linestyle="--", color="tab:blue", markersize=6)
    plt.plot(y_test.numpy(), label="Actual Values", marker="o", linestyle="--", color="tab:orange", markersize=6)

    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Scaled Number of Deliveries", fontsize=12)

    plt.xticks(ticks=np.arange(len(best_predictions_arr)), labels=np.arange(1, len(best_predictions_arr)+1))

    plt.legend(loc="best", fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(f"scaled_prediction_cluster{k}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # %%
    best_predictions_org = scaler_y.inverse_transform(best_predictions.numpy())
    y_test_org = scaler_y.inverse_transform(y_test.numpy())

    # %%
    best_predictions_org_arr = np.clip(best_predictions_org, 0, None)

    plt.figure(figsize=(10, 6))
    plt.plot(best_predictions_org_arr, label="Predicted Values", marker="x", linestyle="--", color="tab:blue", markersize=6)
    plt.plot(y_test_org, label="Actual Values", marker="o", linestyle="--", color="tab:orange", markersize=6)

    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Number of Deliveries", fontsize=12)

    plt.xticks(ticks=np.arange(len(best_predictions_org_arr)), labels=np.arange(1, len(best_predictions_org_arr)+1))

    plt.legend(loc="best", fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(f"original_prediction_cluster{k}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # %%
    print(f"Maximum Prediction for Cluster {k}: {int(best_predictions_org.max())}")

# %%
script_end_time = time.time()
script_execution_time = script_end_time - script_start_time
print(f"Execution Time: {script_execution_time/60} minutes")