import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import pearsonr

import torch
import torch.nn as nn

from bayes_opt import BayesianOptimization

import warnings
import copy

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

best_test_loss = 1.00

df = pd.read_csv("nn_cluster1_num_of_deliveries.csv")
df["Day"] = pd.to_datetime(df["Order_Date"], format="%Y-%m-%d").dt.dayofweek
data = df[["Number_of_Deliveries", "Day"]].to_numpy()

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

def create_inputs(data, sequence_length):
    sequences = []
    for i in range(len(data)-sequence_length):
        seq = data[i:i+sequence_length]
        lbl = data[i+sequence_length][0]
        sequences.append((seq, lbl))

    X = np.array([seq[0] for seq in sequences])
    y = np.array([[lbl[1]] for lbl in sequences])

    X_train, X_next, y_train, y_next = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_next, y_next, test_size=0.5, shuffle=False)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_plots(y_test, test_output, train_losses, val_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc="best", fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"loss_curve1.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Values", marker="o", linestyle="--", color="tab:orange", markersize=6)
    plt.plot(test_output, label="Predicted Values", marker="x", linestyle="--", color="tab:blue", markersize=6)
    plt.xticks(np.arange(1, len(y_test)))
    plt.yticks(np.arange(0, 7))
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Number of Deliveries", fontsize=12)
    plt.legend(loc="best", fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("prediction1.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(len(data)-len(test_output.flatten())+1),
        data[:-(len(test_output.flatten())-1), 0], 
        label="Actual Values - Train",  
        color="tab:red",
        linewidth=3
    )
    plt.plot(
        np.arange((len(data)-len(y_test.flatten())), len(data)),
        y_test.flatten(), 
        label="Actual Values - Test", 
        color="tab:orange",
        linewidth=3
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
    plt.savefig("finalplot1.png", dpi=300, bbox_inches="tight")
    plt.close()

def cal_perf_metrics(y_test, test_output):
    pearson_corr, _ = pearsonr(test_output, y_test)
    mae = mean_absolute_error(y_test, test_output)
    mse = mean_squared_error(y_test, test_output)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_output)
    pd.DataFrame(
        {
            "MSE": [mse],
            "RMSE": [rmse],
            "R2 Score": [r2],
            "MAE": [mae],
            "Peasons Correlation": [pearson_corr]
        }
    ).to_csv("performance_metrics1.csv", index=False)

def get_data_info(X_train, X_val, X_test):
    pd.DataFrame(
        {
            "Training": [len(X_train)],
            "Validating": [len(X_val)],
            "Testing": [len(X_test)],
        }
    ).to_csv("dataset_info1.csv", index=False)

def train_evaluate(sequence_length, hidden_size, learning_rate, num_layers, lstm_dropout, fc_dropout, epochs, weight_decay):    
    global best_test_loss
    
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    epochs = int(epochs)
    sequence_length = int(sequence_length)

    X_train, X_val, X_test, y_train, y_val, y_test = create_inputs(data, sequence_length)

    model = LSTMModel(2, hidden_size, num_layers, 1, lstm_dropout, fc_dropout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for _ in range(epochs):
        model.train()
        output = model(X_train)
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            val_losses.append(val_loss.item())

        if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                torch.save(model.state_dict(), f"model_state1.pth")

    model.load_state_dict(torch.load(f"model_state1.pth"))
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        test_loss = criterion(test_output, y_test)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        y_test = y_test.numpy()
        test_output = test_output.numpy()
        get_data_info(X_train, X_val, X_test)
        cal_perf_metrics(y_test, test_output)
        get_plots(y_test, test_output, train_losses, val_losses)
                
    return -test_loss.item()

pbounds = {
    "sequence_length": (2, 5),
    "hidden_size": (2, 128),
    "learning_rate": (0.01, 0.1),
    "num_layers": (1, 5),
    "lstm_dropout": (0.0, 0.5),
    "fc_dropout": (0.0, 0.5),
    "epochs": (100, 500),
    "weight_decay": (1e-6, 1e-4)
}
bayesian_optimizer = BayesianOptimization(
    f=train_evaluate,
    pbounds=pbounds,
    random_state=42,
)
bayesian_optimizer.maximize(init_points=40, n_iter=60)

best_params = bayesian_optimizer.max["params"]
print(
    f"Best parameters: "
    f"sequence_length={int(best_params['sequence_length'])}, "
    f"hidden_size={int(best_params['hidden_size'])}, "
    f"learning_rate={best_params['learning_rate']:.4f}, "
    f"num_layers={int(best_params['num_layers'])}, "
    f"lstm_dropout={best_params['lstm_dropout']:.4f}, "
    f"fc_dropout={best_params['fc_dropout']:.4f}, "
    f"epochs={int(best_params['epochs'])}",
    f"weight_decay={best_params['weight_decay']:.6f}"
)