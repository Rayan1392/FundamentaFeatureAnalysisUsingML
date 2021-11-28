import pandas as pd

fs_df = pd.DataFrame()
for i in range(1, 101):
    print(i)
    # data = pd.read_csv(f"/Volumes/HDD/Finance/Thesis/Data/FS/R5Cat-B3Cat/NoAdj/Stat/{i}.csv")
    # data = pd.read_csv(f"/Volumes/HDD/Finance/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/{i}.csv") 
    data = pd.read_csv(f"D:/Data/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/{i}.csv")
    
    fs_df[i] = data['0']
print(fs_df)
# fs_df.to_csv('/Volumes/HDD/Finance/Thesis/Data/FS/R5Cat-B3Cat/NoAdj/Stat/Final_FS.csv')
# fs_df.to_csv('/Volumes/HDD/Finance/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/Final_FS.csv')
fs_df.to_csv('D:/Data/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/Final_FS.csv')
print('Done!')