import pandas as pd

df_new = pd.DataFrame()
# df = pd.read_csv('/Volumes/HDD/Finance/Thesis/Data/Data_For_FS.csv') # MacOS
# df = pd.read_csv('/Volumes/HDD/Finance/Thesis/Data/All_Data_Without_FS_Data.csv') # MacOS
# df = pd.read_csv('G:\My Drive\Thesis\Data\Data_For_FS.csv')
df = pd.read_csv('D:/Data/Thesis/Data/All_Data_Without_FS_Data.csv') # Win


def calc_mean(g_data):
    for col in df.columns:
        # Symbol
        if(col == '1'):
            continue
        # Return
        elif(col == '7'):
            df_new['r_mean'] = list(g_data['7'].mean().round(2))
        else:
            col_values = list(g_data[col].mean().round(2))
            df_new[f'{col}_mean'] = col_values

def calc_std(g_data):
    for col in df.columns:
        # Symbol or return
        if(col == '1' or col == '7'):
            continue
        col_values = list(g_data[col].std().round(2))
        df_new[f'{col}_std'] = col_values

def calc_skew(g_data):
    for col in df.columns:
        # symbol or return
        if(col == '1' or col == '7'):
            continue
        col_values = list(g_data[col].skew().round(2))
        df_new[f'{col}_skew'] = col_values

def calc_interquartile(g_data):
    for col in df.columns:
        # symbol or return
        if(col == '1' or col == '7'):
            continue
       
        col_values3 = list(g_data[col].quantile(0.75).round(2))
        col_values1 = list(g_data[col].quantile(0.25).round(2))

        col_values = []
        for i in range(0, len(col_values3)):
            col_values.append(col_values3[i] - col_values1[i])
    
        df_new[f'{col}_interquartile'] = col_values

def calc_var(g_data):
    for col in df.columns:
        # symbol or return
        if(col == '1' or col == '7'):
            continue
        col_values = list(g_data[col].var().round(2))
        df_new[f'{col}_var'] = col_values

def calc_count(g_data):
    df_new['count'] = list(g_data['7'].count())

def calc_RCat5(g_data):
    row_r_cat = []
    for _, row in df_new.iterrows():
        ret = row['r_mean']
        if(ret >=30):
            row_r_cat.append(1)
        elif(ret >=20 and ret < 30):
            row_r_cat.append(2)
        elif(ret >=10 and ret < 20):
            row_r_cat.append(3)
        elif(ret >=0 and ret < 10):
            row_r_cat.append(4)
        elif(ret < 0):
            row_r_cat.append(5)
    df_new['r_cat'] = row_r_cat

def calc_RCat3(g_data):
    row_r_cat = []
    for _, row in df_new.iterrows():
        ret = row['r_mean']
        if(ret >=30):
            row_r_cat.append(1)
        elif(ret >=10 and ret < 30):
            row_r_cat.append(2)
        elif(ret < 10):
            row_r_cat.append(3)
    df_new['r_cat'] = row_r_cat


    
#SymbolFa, SymbolEn, PeriodType, PersianYear, PersianQuarter, Beta, Return_Cat, Beta_Cat
df = df.drop(['2', '3', '4', '5', '6', '8', '9', '10'], axis=1) 
# Group by CompanyId
g_data = df.groupby('1') 
df_new['1'] = list(g_data.groups.keys())
# calc_count(g_data)
calc_mean(g_data)
# calc_RCat5(g_data)
calc_RCat3(g_data)
calc_std(g_data)
calc_skew(g_data)
calc_var(g_data)
calc_interquartile(g_data)
#calc_quantile(g_data, .25, 1)
#calc_quantile(g_data, .50, 2)
#calc_quantile(g_data, .75, 3)

#df_new.to_csv('/Volumes/HDD/Finance/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/Data_For_FS_Stat.csv')
# df_new.to_csv('/Volumes/HDD/Finance/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/All_Data_Without_FS_Data_Stat.csv')
#df_new.to_csv('D:\Data\Thesis\Data\Data_For_FS_Stat.csv')
df_new.to_csv('D:/Data/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/All_Data_Without_FS_Data_Stat.csv')
print('Done!')


