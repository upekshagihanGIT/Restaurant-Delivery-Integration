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
start_date = pd.to_datetime("2022-03-01")
end_date = pd.to_datetime("2022-04-06")
full_date_range = pd.date_range(start=start_date, end=end_date)

# %%




# %%
main_df = pd.read_csv("nn_main_dataset.csv")
main_df["Order_Date"] = pd.to_datetime(main_df["Order_Date"])
main_df["Time_Orderd"] = pd.to_datetime(main_df["Time_Orderd"], format="%H:%M:%S", errors="coerce")

# %%
main_df = pd.read_csv("nn_main_dataset.csv")
main_df["Order_Date"] = pd.to_datetime(main_df["Order_Date"])
main_df["Time_Orderd"] = pd.to_datetime(main_df["Time_Orderd"], format="%H:%M:%S", errors="coerce")

# %%
cluster_list = []
vehicle_type_count_list = []
objective_list = []
historical_weight_list = []











    


bayesian_optimizer = BayesianOptimization(
        f=train_evaluate,
        pbounds=pbounds,
        random_state=42,
    )

bayesian_optimizer.maximize(init_points=30, n_iter=120)

best_params = bayesian_optimizer.max["params"]

sequences = create_sequences(data, best_params["sequence_length"])
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