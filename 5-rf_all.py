import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# All Data stat without fs data 
# df = pd.read_csv('/Volumes/HDD/Finance/Thesis/Data/FS/R5Cat-B3Cat/NoAdj/Stat/All_Data_Without_FS_Data_Stat.csv')  
#df = pd.read_csv('/Volumes/HDD/Finance/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/All_Data_Without_FS_Data_Stat.csv')  
df = pd.read_csv('D:/Data/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/All_Data_Without_FS_Data_Stat.csv')  
df = df.dropna()
# features = pd.read_csv('/Volumes/HDD/Finance/Thesis/Data/FS/R5Cat-B3Cat/NoAdj/Stat/sorted_f.csv')
# features = pd.read_csv('/Volumes/HDD/Finance/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/sorted_f.csv')
features = pd.read_csv('D:/Data/Thesis/Data/FS/R3Cat-B3Cat/NoAdj/Stat/sorted_f.csv')
features = features['feature']
list_of_selected_features = []
for i in range(0, 19):
    list_of_selected_features.append(features[i])

data_with_selected_features = df[list_of_selected_features]

X = data_with_selected_features
y = df['r_cat']  # Labels


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) # 70% training and 30% test


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
# prediction on test set
y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Creating the confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
# Assigning columns names
# cm_df = pd.DataFrame(cm, 
#             columns = ['Predicted Negative', 'Predicted Positive'],
#             index = ['Actual Negative', 'Actual Positive'])
# Showing the confusion matrix
print(cm)

print('Done!')

    

