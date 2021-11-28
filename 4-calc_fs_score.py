import pandas as pd
import numpy as np
import operator
import csv 

# df = pd.read_csv("/Volumes/HDD/Finance/Thesis/Data/FS/R5Cat-B3Cat/NoAdj/Stat/Final_FS.csv")
# df = pd.read_csv("/Volumes/HDD/Finance/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/Final_FS.csv")
df = pd.read_csv("D:/Data/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/Final_FS.csv")

features_list = {}
for i in range(1, 101):
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
# with open('/Volumes/HDD/Finance/Thesis/Data/FS/R5Cat-B3Cat/NoAdj/Stat/sorted_f.csv', 'w') as f:
# with open('/Volumes/HDD/Finance/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/sorted_f.csv', 'w') as f:
with open('D:/Data/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/sorted_f.csv', 'w') as f:
    write = csv.writer(f) 
    write.writerow(['feature', 'score']) 
    write.writerows(sorted_x) 
