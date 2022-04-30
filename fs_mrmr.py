import pandas as pd
from mrmr import mrmr_classif
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import operator
import csv

fs_repeat = 10
prediction_cat = 'b_cat'
prediction_name = 'Beta'
remove_cat = 'r_cat'
remove_name = 'Return'

data_path = "/Volumes/GoogleDrive/My Drive/Thesis/Data"
in_file_path = f"{data_path}/FS/R2Cat-B3Cat/NoAdj/NonStat/Data_For_FS_NonStat.csv"
out_file_path = f"{data_path}/FS/R2Cat-B3Cat/NoAdj/NonStat/{prediction_name}"

data = pd.read_csv(in_file_path)

for i in range(1, fs_repeat + 1):
    print(i)
    data1 = data.sample(frac=0.8)
    X = data1.drop(['Unnamed: 0', prediction_cat, prediction_name, remove_cat, remove_name], axis=1) 
    y = data1[prediction_cat] # return
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # use mrmr classification
    selected_features = mrmr_classif(X, y, K = 24)
    sf_df = pd.DataFrame(selected_features)
    sf_df.to_csv(f"{out_file_path}/{i}.csv", sep=",", encoding="utf-8")
    print(selected_features)

file_path = f"/Volumes/GoogleDrive/My Drive/Thesis/Data/FS/R2Cat-B3Cat/NoAdj/NonStat/{prediction_name}"

fs_df = pd.DataFrame()
for i in range(1, fs_repeat + 1):
    print(i)
    data = pd.read_csv(f"{file_path}/{i}.csv")
    fs_df[i] = data['0']
print(fs_df)
fs_df.to_csv(f'{file_path}/Final_FS.csv')
print('Done!')

file_path = f"/Volumes/GoogleDrive/My Drive/Thesis/Data/FS/R2Cat-B3Cat/NoAdj/NonStat/{prediction_name}"
df = pd.read_csv(f"{file_path}/Final_FS.csv")

features_list = {}
for i in range(1, fs_repeat + 1):
    ix = 1
    for f in df[str(i)]:
        if(f in features_list.keys()):
            features_list[f].append(ix)
        else:
            features_list[f] = [ix]
        ix += 1
        print(f)

f_scores = {}
for key in features_list:
    f_scores[key] = np.mean(features_list[key], axis=0)

sorted_x = sorted(f_scores.items(), key=operator.itemgetter(1))
with open(f'{file_path}/sorted_f.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(['Feature', 'score'])
    write.writerows(sorted_x)
