# %%
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import rand_score, mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

import matplotlib.pyplot as plt

import warnings

import time

warnings.filterwarnings('ignore')

np.random.seed(42)

# %%
df = pd.read_csv("delivery_locations.csv")
df.head()

# %%
zones_map = {
    "Financial District": 0,
    "SoHo": 1,
    "Chelsea": 2, 
    "Midtown": 3,
    "Upper East Side": 4,
    "Greenwich Village": 5,
    "Gramercy": 6,
    "Tribeca": 7,
    "Hell's Kitchen": 8,
    "Upper West Side": 9
}

df["zones_label"] = df["delivery_zone"].map(zones_map)

# %%
scaler = StandardScaler()
df[["scaled_latitude", "scaled_longitude"]] = scaler.fit_transform(df[["delivery_location_latitude", "delivery_location_longitude"]])
df.head()

# %%
start_time = time.time()
model = MiniBatchKMeans(n_clusters=10)
model.fit(df[["scaled_latitude", "scaled_longitude"]])
end_time = time.time()
print(f"Run time: {end_time-start_time}")

# %%
df["clusters"] = model.labels_

# %%
latitudes = df["delivery_location_latitude"].to_list()
longitudes = df["delivery_location_longitude"].to_list()

# %%
fig, ax = plt.subplots()

scatter = ax.scatter(latitudes, longitudes, c=df["clusters"].to_list())

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="best", title="Clusters")
ax.add_artist(legend1)
ax.set_ylabel("Longitudes")
ax.set_xlabel("Latitudes")

plt.savefig("plot.png", dpi=300, bbox_inches="tight")
plt.close()

# %%
sl = silhouette_score(df[["scaled_latitude", "scaled_longitude"]], df["clusters"])
db = davies_bouldin_score(df[["scaled_latitude", "scaled_longitude"]], df["clusters"])
ch = calinski_harabasz_score(df[["scaled_latitude", "scaled_longitude"]], df["clusters"])

r = rand_score(df["zones_label"].to_list(), df["clusters"].to_list())
mi = mutual_info_score(df["zones_label"].to_list(), df["clusters"].to_list())
fm = fowlkes_mallows_score(df["zones_label"].to_list(), df["clusters"].to_list())

ar = adjusted_rand_score(df["zones_label"].to_list(), df["clusters"].to_list())
ami = adjusted_mutual_info_score(df["zones_label"].to_list(), df["clusters"].to_list())

nmi = normalized_mutual_info_score(df["zones_label"].to_list(), df["clusters"].to_list())

h = homogeneity_score(df["zones_label"].to_list(), df["clusters"].to_list())
c = completeness_score(df["zones_label"].to_list(), df["clusters"].to_list())
vm = v_measure_score(df["zones_label"].to_list(), df["clusters"].to_list())

# %%
pd.DataFrame(
    {
        "Silhouette Score": [sl],
        "Davies-Bouldin Index": [db],
        "Calinski-Harabasz Index": [ch],
        "Rand index": [r],
        "Mutual Information": [mi],
        "Fowlkes-Mallows Index": [fm],
        "Adjusted Rand Index": [ar],
        "Adjusted Mutual Information": [ami],
        "Normalized Mutual Information": [nmi],
        "Homogeneity": [h],
        "Completeness ": [c],
        "V-measure": [vm],
        
    }
).to_csv("metrics.csv", index=False)


