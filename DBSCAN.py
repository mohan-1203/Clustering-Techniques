import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import metrics
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN 
from sklearn import preprocessing

mall_dataset = pd.read_csv('C:/Users/mohan/OneDrive/Desktop/Mall_Customers.csv')
mall_dataset.head()



label_encoder = preprocessing.LabelEncoder() 

mall_dataset['Genre'] = label_encoder.fit_transform(mall_dataset['Genre'])
mall_dataset.head()

from itertools import product

eps_values = np.arange(8,12.75,0.25) # eps values to be investigated
min_samples = np.arange(3,10) # min_samples values to be investigated

DBSCAN_params = list(product(eps_values, min_samples))

from sklearn.metrics import silhouette_score

sil_score = []
X = mall_dataset.drop(['CustomerID'], axis=1)

for p in DBSCAN_params:
    DBS_clustering = DBSCAN(eps=p[0], min_samples=p[1]).fit(X)
    sil_score.append(silhouette_score(X, DBS_clustering.labels_))


max_score = max(sil_score)


max_index = sil_score.index(max(sil_score))


best_params = DBSCAN_params[max_index]

print(max_score, max_index, best_params)


DBS_clustering = DBSCAN(eps=best_params[0], min_samples=best_params[1]).fit(X)


mall_dataset.loc[:,'cluster'] = DBS_clustering.labels_ 


mall_dataset['cluster'].value_counts().reset_index().sort_values(by=['cluster'])

fig = plt.figure(figsize=(12,6))

colors = ["#0f96c7","#90e0ef", "#023e8a","#ffbd00","#ff5400", "#000"]

for r in range(6):
    
    
    if r==5:
        r=-1
    
    clustered_customer = mall_dataset[mall_dataset["cluster"] == r]
    plt.scatter(clustered_customer["Annual Income (k$)"], clustered_customer["Spending Score (1-100)"], color=colors[r])
    
plt.title("Mall Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()