# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import time
import copy

from scipy.stats import pearsonr

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import torch
import torch.nn as nn

from bayes_opt import BayesianOptimization

# %%
script_start_time = time.time()

EPOCHS = 500
start_date = pd.to_datetime("2022-03-01")
end_date = pd.to_datetime("2022-04-06")
full_date_range = pd.date_range(start=start_date, end=end_date)

# %%
main_df = pd.read_csv("nn_main_dataset.csv")

# %%
main_df["Order_Date"] = pd.to_datetime(main_df["Order_Date"])
main_df["Time_Orderd"] = pd.to_datetime(main_df["Time_Orderd"], format="%H:%M:%S", errors="coerce")

# %%
cluster_list = []
vehicle_type_count_list = []
objective_list = []
historical_weight_list = []

# %%
for k in range(2):

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
    plt.figure(figsize=(12, 6))
    plt.bar(cluster_df["Order_Date"],cluster_df["Number of Deliveries"], color="tab:blue", edgecolor="black")
    plt.xlabel("Order Dates", fontsize=12)
    plt.ylabel("Number of Deliveries", fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"nn_cluster{k}_num_of_deliveries.png", dpi=300, bbox_inches="tight")
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
    def generate_XY(sequences):
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

        return X_train, X_val, X_test, y_train, y_val, y_test

    # %%
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, lstm_dropout, fc_dropout):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True, 
                dropout=lstm_dropout if num_layers > 1 else 0
            )
            self.dropout = nn.Dropout(fc_dropout)
            self.fc = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
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
            output = self.relu(self.fc(self.dropout(out[:, -1, :])))
            return output

    # %%
    pbounds = {
        "sequence_length": (2, 30),
        "hidden_size": (2, 16),
        "learning_rate": (0.01, 0.1),
        "num_layers": (1, 5),
        "lstm_dropout": (0.1, 0.5),
        "fc_dropout": (0.1, 0.5),
        "epochs": (600, 1000) 
    }

    # %%
    def train_evaluate(sequence_length, hidden_size, learning_rate, num_layers, lstm_dropout, fc_dropout, epochs):
        hidden_size = int(hidden_size)
        num_layers = int(num_layers)
        epochs = int(epochs)
        sequence_length = int(sequence_length)

        sequences = create_sequences(data, sequence_length)
        X_train, X_val, X_test, y_train, y_val, y_test = generate_XY(sequences)

        model = LSTMModel(1, hidden_size, num_layers, 1, lstm_dropout, fc_dropout)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
                
        return -val_loss.item()

    bayesian_optimizer = BayesianOptimization(
        f=train_evaluate,
        pbounds=pbounds,
        random_state=42,
    )

    bayesian_optimizer.maximize(init_points=30, n_iter=120)

    best_params = bayesian_optimizer.max["params"]

    sequences = create_sequences(data, int(best_params["sequence_length"]))
    X_train, X_val, X_test, y_train, y_val, y_test = generate_XY(sequences)

    best_model = LSTMModel(
        input_size=1,
        hidden_size=int(best_params["hidden_size"]),
        num_layers=int(best_params["num_layers"]),
        output_size=1,
        lstm_dropout=best_params["lstm_dropout"],
        fc_dropout=best_params["fc_dropout"]
    )
    best_optimizer = torch.optim.Adam(params=best_model.parameters(), lr=best_params["learning_rate"])
    best_criterion = nn.MSELoss()

    for epoch in range(int(best_params["epochs"])):
        best_model.train()
        best_output = best_model(X_train.unsqueeze(-1))
        best_loss = best_criterion(best_output, y_train)
        best_optimizer.zero_grad()
        best_loss.backward()
        best_optimizer.step()
        print(F"Epoch: {epoch+1} Loss: {best_loss.item()}")

    # %%
    best_model.eval()
    with torch.no_grad():
        best_predictions = best_model(X_test.unsqueeze(-1))
        best_loss = best_criterion(best_predictions, y_test)
        best_pearson_corr, _ = pearsonr(best_predictions.numpy(), y_test.numpy())
        print(f"MSE Loss for Cluster {k}: {best_loss.item()}")
        print(f"Pearson's Correlation Coefficient for Cluster {k}: {best_pearson_corr.item()}")

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

    plt.savefig(f"nn_cluster{k}_scaled_prediction.png", dpi=300, bbox_inches="tight")
    plt.close()

    continue

    # %%
    best_predictions_org = scaler_y.inverse_transform(best_predictions.numpy())
    y_test_org = scaler_y.inverse_transform(y_test.numpy())

    # %%
    best_predictions_org_arr = np.round(best_predictions_org).astype(int)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(data)), data, label="Actual Values", color="tab:orange", linewidth=2)
    plt.plot(np.arange(len(data)-(sequence_length-1), len(data)), best_predictions_org_arr, label="Predicted Values", color="tab:blue")

    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Number of Deliveries", fontsize=12)

    plt.legend(loc="best", fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(f"nn_cluster{k}_original_prediction.png", dpi=300, bbox_inches="tight")
    plt.close()

    # %%
    print(f"Maximum Prediction for Cluster {k}: {int(best_predictions_org.max())}")

# %%
script_end_time = time.time()
script_execution_time = script_end_time - script_start_time
print(f"Execution Time: {script_execution_time/60} minutes")