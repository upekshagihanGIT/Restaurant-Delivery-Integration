#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn


# In[2]:


df = pd.read_csv("train.csv")


# In[ ]:


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


# In[4]:


temp_df = df[["Restaurant_latitude", "Restaurant_longitude"]]
temp_df = temp_df.groupby(by=["Restaurant_latitude", "Restaurant_longitude"]).size().reset_index(name="Count")
temp_df.sort_values(by="Count", ascending=False, inplace=True)
temp_df = temp_df[
    (temp_df["Restaurant_latitude"].apply(lambda x: 26.0 <= x <= 27.0)) &
    (temp_df["Restaurant_longitude"].apply(lambda x: 75.0 <= x <= 76.0))
]


# In[5]:


start_time = pd.to_datetime("00:01:01").time()
end_time = pd.to_datetime("23:59:59").time()
start_date = pd.to_datetime("2022-03-01")
end_date = pd.to_datetime("2022-04-06")
full_date_range = pd.date_range(start=start_date, end=end_date)
restaurant_latitude = temp_df["Restaurant_latitude"].mean()
restaurant_longitude = temp_df["Restaurant_longitude"].mean()


main_df = df[
    (df["Restaurant_latitude"].isin(temp_df["Restaurant_latitude"])) &
    (df["Restaurant_longitude"].isin(temp_df["Restaurant_longitude"])) &
    (df["Time_Orderd"].dt.time.between(start_time, end_time)) &
    (df["Order_Date"].between(start_date, end_date))
]
main_df["Restaurant_latitude"] = restaurant_latitude
main_df["Restaurant_longitude"] = restaurant_longitude


scaler1 = StandardScaler()
main_df[["Scaled_Customer_Latitude", "Scaled_Customer_Longitude"]] = scaler1.fit_transform(main_df[["Delivery_location_latitude", "Delivery_location_longitude"]])


# In[8]:


def find_best_k(data, max_k):
    silhouette_scores = []
    kmeans_labels = []
    for k in range(2, max_k+1):
        kmeans_optimal = KMeans(n_clusters=k)
        kmeans_optimal.fit(data)
        kmeans_labels.append((k, kmeans_optimal.labels_))
        silhouette_scores.append((k, silhouette_score(data, kmeans_optimal.labels_)))
    return silhouette_scores, kmeans_labels


# In[ ]:


silhouette_scores, kmeans_labels = find_best_k(main_df[["Scaled_Customer_Latitude", "Scaled_Customer_Longitude"]], 10)
best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
best_labels = next(lbl for k, lbl in kmeans_labels if k == best_k)
main_df["K-Means Label"] = best_labels


# In[ ]:


plt.scatter(x=main_df["Delivery_location_latitude"], y=main_df["Delivery_location_longitude"], c=main_df["K-Means Label"])
plt.xlabel("Customer Location Latitudes")
plt.ylabel("Customer Location Longitudes")
plt.legend()
plt.show()


# In[ ]:


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


# In[2]:


for k in range(best_k):
    print(f"Processing Cluster k={k}")
    cluster_df = pd.DataFrame()
    cluster_df = main_df[main_df["K-Means Label"] == k]
    cluster_df = cluster_df.groupby(by="Order_Date").size().reset_index(name="Number of Deliveries")
    cluster_df.sort_values(by="Order_Date", inplace=True)
    cluster_df.set_index(keys="Order_Date", inplace=True)
    cluster_df = cluster_df.reindex(labels=full_date_range, fill_value=0)
    cluster_df.reset_index(inplace=True)
    cluster_df.columns = ["Order_Date", "Number of Deliveries"]

    data = cluster_df["Number of Deliveries"].to_numpy()
    
    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data)-sequence_length):
            seq = data[i:i+sequence_length]
            lbl = data[i+sequence_length]
            sequences.append((seq, lbl))
        return sequences

    sequences = create_sequences(data, 5)

    X = np.array([seq[0] for seq in sequences])
    y = np.array([[lbl[1]] for lbl in sequences])

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_next, y_train, y_next = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_next, y_next, test_size=0.5, shuffle=False)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    hidden_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    learning_rates = np.logspace(-4, -1, 30, endpoint=True).tolist()
    epochs_list = np.linspace(10, 500, 50).astype(int).tolist()

    combinations = itertools.product(hidden_sizes, learning_rates, epochs_list)

    best_accuracy = float('inf')
    best_params = (None, None, None)

    for hidden_size, lr, epochs in combinations:
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

    o_model = LSTMModel(1, best_params[0], 1)
    o_optimizer = torch.optim.Adam(o_model.parameters(), lr=best_params[1])
    o_criterion = nn.MSELoss()
    o_epochs = best_params[2]

    for epoch in range(o_epochs):   
        o_model.train()
        o_output = o_model(X_train.unsqueeze(-1))
        o_loss = o_criterion(o_output, y_train)
        o_optimizer.zero_grad()
        o_loss.backward()
        o_optimizer.step()
        print(F"Epoch: {epoch+1} Loss: {o_loss.item()}")

    o_model.eval()
    with torch.no_grad():
        o_predictions = o_model(X_test.unsqueeze(-1))
        o_loss = o_criterion(o_predictions, y_test)
        print(f"Test Loss: {o_loss.item()}")

    plt.plot(o_predictions, label="Predicted Values", marker="x")
    plt.plot(y_test, label="Actual Values", marker="o")
    plt.title(f"k={k} Scaled")
    plt.savefig(f"Scaled_k={k}.png", format="png", dpi=300)

    o_predictions_org = scaler_y.inverse_transform(o_predictions)
    y_test_org = scaler_y.inverse_transform(y_test)

    plt.plot(o_predictions_org, label="Predicted Values", marker="x")
    plt.plot(y_test_org, label="Actual Values", marker="o")
    plt.title(f"k={k} Transformed")
    plt.savefig(f"Transformed_k={k}.png", format="png", dpi=300)
