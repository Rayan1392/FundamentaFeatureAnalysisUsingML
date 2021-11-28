import pandas as pd
from mrmr import mrmr_classif
from sklearn.datasets import make_classification

# data = pd.read_csv("/Volumes/HDD/Finance/Thesis/Data/FS/R5Cat-B3Cat/NoAdj/Stat/Data_For_FS_Stat.csv")
data = pd.read_csv("/Volumes/HDD/Finance/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/Data_For_FS_Stat.csv")

for i in range(1,101):
    print(i)
    data1 = data.sample(frac=0.8)
    X = data1.drop(['Unnamed: 0', '1', 'r_mean', 'r_cat'], axis=1) # CompanyId
    y = data1['r_cat'] # return
    X = pd.DataFrame(X)
    # X = X.astype(int)
    y = pd.Series(y)
    # y = y.astype(int)

    # use mrmr classification
    selected_features = mrmr_classif(X, y, K = 320)
    sf_df = pd.DataFrame(selected_features)
    # sf_df.to_csv(f"/Volumes/HDD/Finance/Thesis/Data/FS/R5Cat-B3Cat/NoAdj/Stat/{i}.csv", sep=",", encoding="utf-8")
    sf_df.to_csv(f"/Volumes/HDD/Finance/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/{i}.csv", sep=",", encoding="utf-8")
    print(selected_features)
