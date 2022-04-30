from pickle import TRUE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from sklearn.metrics import roc_auc_score, plot_precision_recall_curve, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier



prediction_cat = 'b_cat'
prediction_name = 'beta'

# calculated after reading file
num_feats = 0

data_path = "G:\My Drive\Thesis\Data"
file_path = f'{data_path}/FS/R2Cat-B3Cat/NoAdj/NonStat/All_Data_Without_FS_Data_NonStat.csv'
f_file_path = f'{data_path}/FS/R2Cat-B3Cat/NoAdj/NonStat/{prediction_name}/sorted_f.csv'
# f_file_path = f'{data_path}/FS/R2Cat-B3Cat/NoAdj/NonStat/selected_f.csv'


def get_selected_features():
    list_of_selected_features = []
    num_feats = len(features) - 1
    for i in range(0, num_feats):
        # if(str(features[i]) not in {"CompanyId", "PersianYear"}):
        list_of_selected_features.append(str(features[i]))

    return list_of_selected_features

def get_train_test(df, year):
    df_train = df[df.PersianYear <= year]
    df_test = df[df.PersianYear >= year + 1]

    lis_of_selected_features = get_selected_features()
    X_train = df_train[lis_of_selected_features].drop(['CompanyId', 'PersianYear'], axis=1)
    y_train = df_test[lis_of_selected_features].drop(['CompanyId', 'PersianYear'], axis=1)
    X_test = df_train[prediction_cat]
    y_test = df_test[prediction_cat]
    # lis_of_selected_features = get_selected_features()
    # #df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    # X_train = X_train[lis_of_selected_features]
    # y_train = y_train[lis_of_selected_features]
    return X_train, X_test, y_train, y_test

def decisionTree(num_feats, max_accuracy_dt, max_accuracy_dt_num_feats, plot_roc_auc):
    '''
    Decision Tree Classifier 
    '''
    # Create Decision Tree classifer object
    model = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    model = model.fit(X_train,y_train)
    #Predict the response for test dataset
    y_pred = model.predict(X_test)
    # scores = clf.score(X_test,y_test)
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # metrics.accuracy_score(y_test, y_pred)
    print(f"DecisionTreeClassifier Accuracy with {num_feats} features: ", np.mean(scores).round(4))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    calc_Specificity(conf_matrix)
    calc_Sensitivity(conf_matrix)
    # print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    if(np.mean(scores) > max_accuracy_dt):
        max_accuracy_dt_num_feats = num_feats
        max_accuracy_dt = np.mean(scores)
        if(len(y_test.value_counts()) == 2):
            # Get the predicited probability of testing data
            y_score = model.predict_proba(X_test)[:, 1]
            # Average precision score
            average_precision = average_precision_score(y_test, y_score)
            roc_auc = roc_auc_score(y_test, y_score)
            print(f'ROC/AUC Score: {roc_auc}')
            print(f'PR/AUC Score: {average_precision}')
            if(plot_roc_auc == True):
                metrics.plot_roc_curve(model, X_test, y_test)
                plt.show()
                disp = plot_precision_recall_curve(model, X_test, y_test)
                disp.ax_.set_title('Binary class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
                plt.show()
                disp = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
                plt.show()
        else:
            print(f'ROC/AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="weighted")}')
    return max_accuracy_dt, max_accuracy_dt_num_feats

def rf(num_feats, max_accuracy_rf, max_accuracy_rf_num_feats, plot_roc_auc):
    '''
    Random Forest
    '''
    model = RandomForestClassifier(n_estimators=100, bootstrap=True, criterion="entropy", max_features='log2', random_state=100)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    scores=model.score(X_test, y_test)
    # cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    print(f"RandomForestClassifier Accuracy with {num_feats} features: ", np.mean(scores).round(4))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    calc_Specificity(conf_matrix)
    calc_Sensitivity(conf_matrix)
    if(len(y_test.value_counts()) == 2):
        # Get the predicted probability of testing data
        y_score = model.predict_proba(X_test)[:, 1]    
        # Average precision score
        average_precision = average_precision_score(y_true= y_test, y_score= y_score)
        roc_auc = roc_auc_score(y_test, y_score)
        print(f'ROC/AUC Score: {roc_auc}')
        print(f'PR/AUC Score: {average_precision}')
        
        if(np.mean(scores) > max_accuracy_rf):
            max_accuracy_rf_num_feats = num_feats
            max_accuracy_rf = np.mean(scores)
            if(plot_roc_auc == True):
                metrics.plot_roc_curve(model, X_test, y_test)
                disp = plot_precision_recall_curve(model, X_test, y_test)
                disp.ax_.set_title('Binary class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

                disp = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
                plt.show()
    else:
        if(np.mean(scores) > max_accuracy_rf):
            max_accuracy_rf_num_feats = num_feats
            max_accuracy_rf = np.mean(scores)
        print(f'ROC/AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="weighted")}')
    return max_accuracy_rf, max_accuracy_rf_num_feats

def lr(num_feats, max_accuracy_lr, max_accuracy_lr_num_feats, plot_roc_auc):
    '''
    LogisticRegression
    '''
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = model.score(X_test,y_test)
    # cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('LogisticRegression %d features Accuracy: %.4f (%.4f)' % (num_feats,  np.mean(scores), np.std(scores)))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    calc_Specificity(conf_matrix)
    calc_Sensitivity(conf_matrix)

    if(len(y_test.value_counts()) == 2):
        # Get the predicited probability of testing data
        y_score = model.predict_proba(X_test)[:, 1]
        # Average precision score
        average_precision = average_precision_score(y_test, y_score)
        roc_auc = roc_auc_score(y_test, y_score)

        print(f'ROC/AUC Score: {roc_auc}')
        print(f'PR/AUC Score: {average_precision}')
        if(np.mean(scores)>max_accuracy_lr):
            max_accuracy_lr_num_feats = num_feats
            max_accuracy_lr = np.mean(scores)
            if(plot_roc_auc == True):
                metrics.plot_roc_curve(model, X_test, y_test)
                plt.show()
                plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
                plt.show()
    else:
        print(f'ROC/AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="weighted")}')
        if(np.mean(scores)>max_accuracy_lr):
            max_accuracy_lr_num_feats = num_feats
            max_accuracy_lr = np.mean(scores)
    return max_accuracy_lr, max_accuracy_lr_num_feats

def gb(num_feats, max_accuracy_nb, max_accuracy_nb_num_feats, plot_roc_auc):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # scores = model.score(X_test, y_test)
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(f"GaussianNB Accuracy with {num_feats} features: {np.mean(scores).round(4)}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    calc_Specificity(conf_matrix)
    calc_Sensitivity(conf_matrix)
    if(np.mean(scores)>max_accuracy_nb):
        if(len(y_test.value_counts()) == 2):
            # Average precision score
            # average_precision = average_precision_score(y_test, y_score)
            # print(f'PR/AUC Score: {average_precision}')
            # Get the predicited probability of testing data
            y_score = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_score)
            print(f'ROC/AUC Score: {roc_auc}')
            max_accuracy_nb_num_feats = num_feats
            max_accuracy_nb = np.mean(scores)
            if(plot_roc_auc == True):
                metrics.plot_roc_curve(model, X_test, y_test)
                plt.show()
                plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
                plt.show()
        else:
            print(f'ROC/AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="weighted")}')
            max_accuracy_nb_num_feats = num_feats
            max_accuracy_nb = np.mean(scores)
    return max_accuracy_nb, max_accuracy_nb_num_feats

def knn(num_feats, max_accuracy_knn, max_accuracy_knn_num_feats):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # scores = model.score(X_test, y_test)
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    print(f"KNeighborsClassifier Accuracy with {num_feats} features: {np.mean(scores).round(4)}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    calc_Specificity(conf_matrix)
    calc_Sensitivity(conf_matrix)

    if(np.mean(scores)>max_accuracy_knn):
        max_accuracy_knn_num_feats = num_feats
        max_accuracy_knn = np.mean(scores)
        if(len(y_test.value_counts()) == 2):
            y_score = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_score)
            print(f'ROC/AUC Score: {roc_auc}')
            metrics.plot_roc_curve(model, X_test, y_test)
            plt.show()
            plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
            plt.show()
    # print(f'ROC/AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1])}')
    # if(roc_auc_score == True):
    #     metrics.plot_roc_curve(model, X_test, y_test)
    #     plt.show()
    return max_accuracy_knn, max_accuracy_knn_num_feats

def svc(num_feats, max_accuracy_svc, max_accuracy_svc_num_feats):
    model = svm.SVC(probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # scores = model.score(X_test, y_test)
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(f"SVM Accuracy with {num_feats} features: {np.mean(scores).round(4)}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    calc_Specificity(conf_matrix)
    calc_Sensitivity(conf_matrix)
    if(np.mean(scores)>max_accuracy_svc):
        max_accuracy_svc_num_feats = num_feats
        max_accuracy_svc = np.mean(scores)
        if(len(y_test.value_counts()) == 2):
            y_score = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_score)
            print(f'ROC/AUC Score: {roc_auc}')
            metrics.plot_roc_curve(model, X_test, y_test)
            plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
            plt.show()
        
    print(f'ROC/AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1])}')
    if(roc_auc_score == True):
        metrics.plot_roc_curve(model, X_test, y_test)
        plt.show()
    return max_accuracy_svc, max_accuracy_svc_num_feats

def calc_Specificity(conf_matrix):
    if(conf_matrix.shape[0] == 3):
        n_11 = conf_matrix[0][0]
        n_12 = conf_matrix[0][1]
        n_13 = conf_matrix[0][2]
        n_21 = conf_matrix[1][0]
        n_22 = conf_matrix[1][1]
        n_23 = conf_matrix[1][2]
        n_31 = conf_matrix[2][0]
        n_32 = conf_matrix[2][1]
        n_33 = conf_matrix[2][2]
        print(f'Specificity class1: {np.round((n_22+n_33)/(n_21+n_31+n_22+n_33), 4)}')
        print(f'Specificity class2: {np.round((n_11+n_33)/(n_12+n_32+n_11+n_33), 4)}')
        print(f'Specificity class3: {np.round((n_11+n_22)/(n_13+n_23+n_11+n_22), 4)}')
    elif(conf_matrix.shape[0] == 2):
        fp = conf_matrix[0][1]
        tn = conf_matrix[1][1]
        print(f'Specificity: {np.round(tn/(fp+tn), 4)}')

def calc_Sensitivity(conf_matrix):
    if(conf_matrix.shape[0] == 3):
        n_11 = conf_matrix[0][0]
        n_12 = conf_matrix[0][1]
        n_13 = conf_matrix[0][2]
        n_21 = conf_matrix[1][0]
        n_22 = conf_matrix[1][1]
        n_23 = conf_matrix[1][2]
        n_31 = conf_matrix[2][0]
        n_32 = conf_matrix[2][1]
        n_33 = conf_matrix[2][2]
        print(f'Sensitivity class1: {np.round((n_11)/(n_11+n_12+n_13), 4)}')
        print(f'Sensitivity class2: {np.round((n_22)/(n_21+n_22+n_23), 4)}')
        print(f'Sensitivity class3: {np.round((n_33)/(n_31+n_32+n_33), 4)}')
    elif(conf_matrix.shape[0] == 2):
        tp = conf_matrix[0][0]
        fn = conf_matrix[1][0]
        print(f'Sensitivity: {np.round(tp/(tp+fn), 4)}')

# All Data stat without fs data 
df = pd.read_csv(file_path) 
df = df.dropna()
features = pd.read_csv(f_file_path)
features = features['Feature']
num_feats = len(features) - 1

max_accuracy_dt = 0
max_accuracy_rf = 0
max_accuracy_lr = 0
max_accuracy_nb = 0
max_accuracy_knn = 0
max_accuracy_svc = 0
max_accuracy_dt_num_feats = 0
max_accuracy_rf_num_feats = 0
max_accuracy_lr_num_feats = 0
max_accuracy_nb_num_feats = 0
max_accuracy_knn_num_feats = 0
max_accuracy_svc_num_feats = 0
while(num_feats>=1):
    lis_of_selected_features = get_selected_features()
    data_with_selected_features = df[lis_of_selected_features]

    X = data_with_selected_features
    # y = df['b_cat']  # Labels
    y = df[prediction_cat]  # Labels
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) # 80% training and 30% test
    # X_train, X_test= np.split(X, [int(.77 *len(X))])
    # y_train, y_test = np.split(y, [int(.77 *len(y))])
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # DecisionTreeClassifier
    max_accuracy_dt, max_accuracy_dt_num_feats = decisionTree(num_feats, max_accuracy_dt, max_accuracy_dt_num_feats, True)
    # GaussianNB
    max_accuracy_nb, max_accuracy_nb_num_feats = gb(num_feats, max_accuracy_nb, max_accuracy_nb_num_feats, True)
    # KNeighborsClassifier
    max_accuracy_knn, max_accuracy_knn_num_feats = knn(num_feats, max_accuracy_knn, max_accuracy_knn_num_feats)
    # SVM
    max_accuracy_svc, max_accuracy_svc_num_feats = svc(num_feats, max_accuracy_svc, max_accuracy_svc_num_feats)
    # LogisticRegression
    max_accuracy_lr, max_accuracy_lr_num_feats = lr(num_feats, max_accuracy_lr, max_accuracy_lr_num_feats, True)
    # RandomForestClassifier
    max_accuracy_rf, max_accuracy_rf_num_feats = rf(num_feats, max_accuracy_rf, max_accuracy_rf_num_feats, True)
    
    # print(y_pred)
    # K-Fold Validation
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    # print(scores)
    # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    num_feats = num_feats - 1


print(f'Random forest highest Accuracy with {max_accuracy_rf_num_feats} features:{(max_accuracy_rf*100).round(2)}')
print(f'Logistic Regression highest Accuracy with {max_accuracy_lr_num_feats} features:{(max_accuracy_lr*100).round(2)}')
print(f'SVM highest Accuracy with {max_accuracy_svc_num_feats} features:{(max_accuracy_svc*100).round(2)}')
print(f'KNeighborsClassifier highest Accuracy with {max_accuracy_knn_num_feats} features:{(max_accuracy_knn*100).round(2)}')
print(f'Naive Bayes highest Accuracy with {max_accuracy_nb_num_feats} features:{(max_accuracy_nb*100).round(2)}')
print(f'Decision Tree  highest Accuracy with {max_accuracy_dt_num_feats} features:{(max_accuracy_dt*100).round(2)}')


