import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from scipy.stats import pearsonr

import torch
import torch.nn as nn

from bayes_opt import BayesianOptimization

# %%
np.random.seed(42)
torch.manual_seed(42)

# %%
script_start_time = time.time()
start_date = pd.to_datetime("2022-03-01")
end_date = pd.to_datetime("2022-04-06")
full_date_range = pd.date_range(start=start_date, end=end_date)

# %%
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lstm_dropout, fc_dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=lstm_dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        output = self.fc(self.dropout(out[:, -1, :]))
        return output

# %%
main_df = pd.read_csv("cl_main_dataset.csv")
main_df["Order_Date"] = pd.to_datetime(main_df["Order_Date"])
main_df["Time_Orderd"] = pd.to_datetime(main_df["Time_Orderd"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

# %%
cluster_list = []
vehicle_type_count_list = []
objective_list = []
historical_weight_list = []

for k in range(2):

    # %%
    def create_data(main_df):
        cluster_df = main_df[main_df["K-Means Label"] == k]
        cluster_df.to_csv(f"nn_cluster{k}_dataset.csv", index=False)

        cluster_df = cluster_df.groupby(by="Order_Date").size().reset_index(name="Number of Deliveries")
        cluster_df.sort_values(by="Order_Date", inplace=True)
        cluster_df.set_index(keys="Order_Date", inplace=True)
        cluster_df = cluster_df.reindex(labels=full_date_range, fill_value=0)
        cluster_df.reset_index(inplace=True)
        cluster_df.columns = ["Order_Date", "Number of Deliveries"]

        plt.figure(figsize=(12, 6))
        plt.bar(cluster_df["Order_Date"],cluster_df["Number of Deliveries"], color="tab:blue", edgecolor="black")
        plt.xlabel("Order Dates", fontsize=12)
        plt.ylabel("Number of Deliveries", fontsize=12)
        plt.xticks(cluster_df["Order_Date"], rotation=90)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(f"nn_cluster{k}_num_of_deliveries.png", dpi=300, bbox_inches="tight")
        plt.close()

        data = cluster_df["Number of Deliveries"].to_numpy()
        return data

    # %%
    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data)-sequence_length):
            seq = data[i:i+sequence_length]
            lbl = data[i+sequence_length]
            sequences.append((seq, lbl))
        return sequences 

    def create_scalers():
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        return scaler_X, scaler_y

    def generate_XY(sequences, scaler_X, scaler_y):
        X = np.array([seq[0] for seq in sequences])
        y = np.array([[lbl[1]] for lbl in sequences])

        X_train, X_next, y_train, y_next = train_test_split(X, y, test_size=0.3, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_next, y_next, test_size=0.5, shuffle=False)

        X_train = scaler_X.fit_transform(X_train)
        y_train = scaler_y.fit_transform(y_train)

        X_val = scaler_X.transform(X_val)
        X_test = scaler_X.transform(X_test)    
        y_val = scaler_y.transform(y_val)
        y_test = scaler_y.transform(y_test)

        pd.DataFrame(
            {
                "Total Records": [len(X)],
                "Training": [len(X_train)],
                "Validating": [len(X_val)],
                "Testing": [len(X_test)],
            }
        ).to_csv(f"nn_cluster{k}_dataset_info.csv", index=False)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
        return X_train, X_val, X_test, y_train, y_val, y_test   
    
    def train_evaluate(sequence_length, hidden_size, learning_rate, num_layers, lstm_dropout, fc_dropout, epochs, weight_decay):
        hidden_size = int(hidden_size)
        num_layers = int(num_layers)
        epochs = int(epochs)
        sequence_length = int(sequence_length)

        data = create_data(main_df)
        sequences = create_sequences(data, sequence_length)
        scaler_X, scaler_y = create_scalers()
        X_train, X_val, X_test, y_train, y_val, y_test = generate_XY(sequences, scaler_X, scaler_y)

        model = LSTMModel(1, hidden_size, num_layers, 1, lstm_dropout, fc_dropout)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for _ in range(epochs):         
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
    
    # %%
    pbounds = {
        "sequence_length": (1, 7),
        "hidden_size": (2, 128),
        "learning_rate": (0.01, 0.1),
        "num_layers": (1, 4),
        "lstm_dropout": (0.0, 0.5),
        "fc_dropout": (0.0, 0.5),
        "epochs": (100, 500),
        "weight_decay": (1e-6, 1e-2)
    }
    pd.DataFrame([pbounds]).to_csv(f"nn_cluster{k}_pbounds.csv")
    bayesian_optimizer = BayesianOptimization(
        f=train_evaluate,
        pbounds=pbounds,
        random_state=42,
    )
    bayesian_optimizer.maximize(init_points=210, n_iter=140)
    best_params = bayesian_optimizer.max["params"]
    print(
        f"Best parameters: "
        f"sequence_length={int(best_params['sequence_length'])}, "
        f"hidden_size={int(best_params['hidden_size'])}, "
        f"learning_rate={best_params['learning_rate']}, "
        f"num_layers={int(best_params['num_layers'])}, "
        f"lstm_dropout={best_params['lstm_dropout']}, "
        f"fc_dropout={best_params['fc_dropout']}, "
        f"epochs={int(best_params['epochs'])}",
        f"weight_decay={best_params['weight_decay']}"
    )
    pd.DataFrame([best_params]).to_csv(f"nn_cluster{k}_best_parameters.csv")

    data = create_data(main_df)
    sequences = create_sequences(data, int(best_params["sequence_length"]))
    scaler_X, scaler_y = create_scalers()
    X_train, X_val, X_test, y_train, y_val, y_test = generate_XY(sequences, scaler_X, scaler_y)

    model = LSTMModel(
        input_size=1,
        hidden_size=int(best_params["hidden_size"]),
        num_layers=int(best_params["num_layers"]),
        output_size=1,
        lstm_dropout=best_params["lstm_dropout"],
        fc_dropout=best_params["fc_dropout"]
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(int(best_params["epochs"])):
        model.train()
        output = model(X_train.unsqueeze(-1))
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(X_val.unsqueeze(-1))
            val_loss = criterion(val_output, y_val)
            val_losses.append(val_loss.item())

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), f"nn_cluster{k}_bestmodel_state.pth")

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{int(best_params["epochs"])}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    model.load_state_dict(torch.load(f"nn_cluster{k}_bestmodel_state.pth"))
    model.eval()
    with torch.no_grad():
        test_output = model(X_test.unsqueeze(-1))
        test_loss = criterion(test_output, y_test)
        print(f"Test Loss for Cluster {k}: {test_loss.item()}")

    # %%
    test_output = scaler_y.inverse_transform(test_output.numpy())
    y_test = scaler_y.inverse_transform(y_test.numpy())

    # %%
    pearson_corr, _ = pearsonr(test_output, y_test)
    mae = mean_absolute_error(y_test, test_output)
    r2 = r2_score(y_test, test_output)
    mse = mean_squared_error(y_test, test_output)
    rmse = np.sqrt(mse)

    # %%
    print(f"Cluster {k} metrics...")
    print(f"MSE Loss: {mse}")
    print(f"RMSE Loss: {rmse}")
    print(f"MAE Loss: {mae}")
    print(f"R2 Score: {r2}")
    print(f"Pearson's Correlation Coefficient for Cluster {k}: {pearson_corr.item()}")

    pd.DataFrame(
    {
        "MSE Loss": [mse],
        "RMSE Loss": [rmse],
        "MAE Loss": [mae],
        "R2 Score": [r2],
        "Pearson's Correlation Coefficient": [pearson_corr.item()]
    }
    ).to_csv(f"nn_cluster{k}_prediction_metrics_b4_rounding.csv", index=False)

    # %%
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Values", marker="o", linestyle="--", color="tab:orange", markersize=6)
    plt.plot(test_output, label="Predicted Values", marker="x", linestyle="--", color="tab:blue", markersize=6)
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Number of Deliveries", fontsize=12)
    plt.legend(loc="best", fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"nn_cluster{k}_prediction_b4_rounding.png", dpi=300, bbox_inches="tight")
    plt.close()


    # %%
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc="best", fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"nn_cluster{k}_loss_curve.png", dpi=300)
    plt.close()

    # %%
    print(f"Before Rounding up: {test_output.flatten()}")
    test_output = np.ceil(test_output).astype(int)
    print(f"After Rounding up: {test_output.flatten()}")
    print(f"Original Values: {y_test.flatten()}")

    # %%
    print(f"Calculating cluster {k} new metrics...")
    pearson_corr, _ = pearsonr(test_output, y_test)
    mae = mean_absolute_error(y_test, test_output)
    r2 = r2_score(y_test, test_output)
    mse = mean_squared_error(y_test, test_output)
    rmse = np.sqrt(mse)

    # %%
    print(f"Cluster {k} new metrics...")
    print(f"MSE Loss: {mse}")
    print(f"RMSE Loss: {rmse}")
    print(f"MAE Loss: {mae}")
    print(f"R2 Score: {r2}")
    print(f"Pearson's Correlation Coefficient for Cluster {k}: {pearson_corr.item()}")

    pd.DataFrame(
    {
        "MSE Loss": [mse],
        "RMSE Loss": [rmse],
        "MAE Loss": [mae],
        "R2 Score": [r2],
        "Pearson's Correlation Coefficient": [pearson_corr.item()]
    }
    ).to_csv(f"nn_cluster{k}_prediction_metrics_aft_rounding.csv", index=False)

    # %%
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Values", marker="o", linestyle="--", color="tab:orange", markersize=6)
    plt.plot(test_output, label="Predicted Values", marker="x", linestyle="--", color="tab:blue", markersize=6)
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Number of Deliveries", fontsize=12)
    plt.legend(loc="best", fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"nn_cluster{k}_prediction_aft_rounding.png", dpi=300, bbox_inches="tight")
    plt.close()

    # %%
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(len(data)-len(test_output.flatten())+1),
        data[:-(len(test_output.flatten())-1)], 
        label="Actual Values - Train",  
        color="tab:red",
        linewidth=4
    )
    plt.plot(
        np.arange((len(data)-len(y_test.flatten())), len(data)),
        y_test.flatten(), 
        label="Actual Values - Test", 
        color="tab:orange",
        linewidth=4
    )
    plt.plot(
        np.arange((len(data)-len(test_output.flatten())), len(data)),
        test_output.flatten(), 
        label="Predicted Values", 
        color="tab:blue",
        linewidth=2
    )
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Number of Deliveries", fontsize=12)
    plt.legend(loc="best", fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"nn_cluster{k}_finalplot.png", dpi=300, bbox_inches="tight")
    plt.close()

script_end_time = time.time()
script_execution_time = script_end_time - script_start_time
print(f"Execution Time: {script_execution_time/60} minutes") 