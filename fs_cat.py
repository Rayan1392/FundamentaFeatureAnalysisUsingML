import pandas as pd

fromYear = 1390
toYear = 1399
data_path = "G:/My Drive/Thesis/Data"

# in_file_path = f"{data_path}/Data_For_FS.csv" 
# out_file_path = f"{data_path}/FS/R2Cat-B3Cat/NoAdj/NonStat/Data_For_FS_NonStat.csv"
in_file_path = f"{data_path}/All_Data_Without_FS_Data.csv"
out_file_path = f"{data_path}/FS/R2Cat-B3Cat/NoAdj/NonStat/All_Data_Without_FS_Data_NonStat.csv"

df = pd.read_csv(f'{in_file_path}') 

def calc_RCat5(df):
    row_r_cat = []
    for _, row in df.iterrows():
        ret = row['Return']
        if(ret > 100):
            row_r_cat.append(1)
        elif(ret > 50 and ret <= 100):
            row_r_cat.append(2)
        elif(ret > 25 and ret <= 50):
            row_r_cat.append(3)
        elif(ret > 0 and ret <= 25):
            row_r_cat.append(4)
        elif(ret <= 0):
            row_r_cat.append(5)
    df['r_cat'] = row_r_cat

def calc_RCat3(df):
    row_r_cat = []
    for _, row in df.iterrows():
        ret = row['Return']
        if(ret >=85):
            row_r_cat.append(1)
        elif(ret >= 8.5 and ret < 85):
            row_r_cat.append(2)
        elif(ret < 8.5):
            row_r_cat.append(3)
    df['r_cat'] = row_r_cat

def calc_BCat3(df):
    row_b_cat = []
    for _, row in df.iterrows():
        beta = row['Beta']
        if(beta > 1):
            row_b_cat.append(1)
        elif(beta >= 0 and beta <= 1):
            row_b_cat.append(2)
        elif(beta < 0):
            row_b_cat.append(3)
    df['b_cat'] = row_b_cat

def calc_BCat2(df):
    row_b_cat = []
    for _, row in df.iterrows():
        beta = row['Beta']
        if(beta > 1 or beta < 0):
            row_b_cat.append(1)
        elif(beta >= 0 and beta <= 1):
            row_b_cat.append(2)
        # elif(beta < 0):
        #     row_b_cat.append(3)
    df['b_cat'] = row_b_cat

def calc_RCat2(df):
    row_r_cat = []
    for _, row in df.iterrows():
        ret = row['Return']
        if(ret >33):
            row_r_cat.append(1)
        else:
            row_r_cat.append(2)
    df['r_cat'] = row_r_cat
 
#SymbolFa, SymbolEn, PeriodType, PersianYear, PersianQuarter, Beta, Return_Cat, Beta_Cat
df = df.drop(df[df.PersianYear < fromYear].index)
df = df.drop(df[df.PersianYear > toYear].index)
df = df.drop(['CompanyId', 'PersianYear',], axis=1)

df = df.dropna()
# calc_RCat5(df)
calc_RCat2(df)
calc_BCat3(df)
# df = df.drop(['Return'], axis=1) 

df.to_csv(f'{out_file_path}')   
print('Done!')
