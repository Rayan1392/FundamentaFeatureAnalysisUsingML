import pandas as pd
import numpy as np
import operator
import csv
from mrmr import mrmr_classif
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
# from lightgbm import LGBMClassifier

fs_repeat = 6
prediction_cat = 'r_cat'
prediction_name = 'Return'
num_feats = 24
data_path = "/Volumes/GoogleDrive/My Drive/Thesis/Data"

in_file_path = f"{data_path}/FS/R2Cat-B3Cat/NoAdj/NonStat/Data_For_FS_NonStat.csv"
out_file_path = f"{data_path}/FS/R2Cat-B3Cat/NoAdj/NonStat"

def mrmr_fs(X, y, num_feats):
    # use mrmr classification
    selected_features = mrmr_classif(X, y, K = num_feats)
    return selected_features

def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

def chi2_fs(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return chi_support, chi_feature

def pre_fs(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()

    return rfe_support, rfe_feature

def lasso_fs(X, y, num_feats):
    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
    X_norm = MinMaxScaler().fit_transform(X)
    embeded_lr_selector.fit(X_norm, y)

    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
    return embeded_lr_support, embeded_lr_feature

def randomforest_fs(X, y, num_feats):
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embeded_rf_selector.fit(X, y)

    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    return embeded_rf_support, embeded_rf_feature


# def lgbm_fs(X, y, num_feats):
#     lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
#     reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

#     embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
#     embeded_lgb_selector.fit(X, y)

#     embeded_lgb_support = embeded_lgb_selector.get_support()
#     embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
#     return embeded_lgb_support, embeded_lgb_feature


data = pd.read_csv(in_file_path)

# for i in range(1, 51):
# print(i)
data1 = data
X = data1.drop(['Unnamed: 0', 'r_cat', 'b_cat', prediction_name], axis=1) # CompanyId
# y = data1['r_cat'] # return
y = data1[prediction_cat] # Beta
X = pd.DataFrame(X)
y = pd.Series(y)

feature_name = X.columns.tolist()

# use mrmr classification
selected_features = mrmr_fs(X, y, num_feats)
cor_support, cor_selected_features = cor_selector(X, y, num_feats)
chi_support, chi_selected_features = chi2_fs(X, y, num_feats)
rfe_support, rfe_selected_features = pre_fs(X, y, num_feats)
embeded_lr_support, embeded_lr_selected_features = lasso_fs(X, y, num_feats)
embeded_rf_support, embeded_rf_selected_features = randomforest_fs(X, y, num_feats)
# embeded_lgb_support, embeded_lgb_selected_features = lgbm_fs(X, y, num_feats)

# put all selection together
feature_selection_df = pd.DataFrame(
    {'Feature':feature_name
    , 'Pearson':cor_support
    , 'Chi-2':chi_support
    , 'RFE':rfe_support
    , 'Logistics':embeded_lr_support
    , 'Random Forest':embeded_rf_support 
#, 'LightGBM':embeded_lgb_support
                                    })
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
print(feature_selection_df.head(num_feats))

feature_selection_df.to_csv(f"{out_file_path}/feature_selection_{prediction_name}.csv", sep=",", encoding="utf-8")

sf_df = pd.DataFrame(feature_selection_df['Feature'])

sf_df.to_csv(f"{out_file_path}/selected_f.csv", sep=",", encoding="utf-8")

print(str(len(feature_name)), 'feature count')
i = 1
sf_df = pd.DataFrame(cor_selected_features)
sf_df.to_csv(f"{out_file_path}/{i}.csv", sep=",", encoding="utf-8")
print(cor_selected_features)
i += 1
sf_df = pd.DataFrame(chi_selected_features)
sf_df.to_csv(f"{out_file_path}/{i}.csv", sep=",", encoding="utf-8")
print(chi_selected_features)
i += 1
sf_df = pd.DataFrame(rfe_selected_features)
sf_df.to_csv(f"{out_file_path}/{i}.csv", sep=",", encoding="utf-8")
print(rfe_selected_features)
i += 1
sf_df = pd.DataFrame(embeded_lr_selected_features)
sf_df.to_csv(f"{out_file_path}/{i}.csv", sep=",", encoding="utf-8")
print(embeded_lr_selected_features)
i += 1
sf_df = pd.DataFrame(embeded_rf_selected_features)
sf_df.to_csv(f"{out_file_path}/{i}.csv", sep=",", encoding="utf-8")
print(embeded_rf_selected_features)
i += 1
sf_df = pd.DataFrame(selected_features)
sf_df.to_csv(f"{out_file_path}/{i}.csv", sep=",", encoding="utf-8")
print(selected_features)
# i += 1
# sf_df = pd.DataFrame(embeded_lgb_selected_features)
# sf_df.to_csv(f"{out_file_path}/{i}.csv", sep=",", encoding="utf-8")
# print(embeded_lgb_selected_features)

file_path = "/Volumes/GoogleDrive/My Drive/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/NonStat"

fs_df = pd.DataFrame()
for i in range(1, fs_repeat + 1):
    print(i)
    data = pd.read_csv(f"{file_path}/{i}.csv")
    fs_df[i] = data['0']
print(fs_df)
fs_df.to_csv(f'{file_path}/Final_FS.csv')
print('Done!')

file_path = "/Volumes/GoogleDrive/My Drive/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/NonStat"
df = pd.read_csv(f"{file_path}/Final_FS.csv")

features_list = {}
for i in range(1, fs_repeat + 1):
    ix = 1
    for f in df[str(i)]:
        if(pd.isna(f)):
            continue
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
