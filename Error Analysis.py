import numpy as np
import csv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as met
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import recall_score
#from sklearn.metrics import roc_curve, auc
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
import os

def settings():
    global file_name_patent_classified, path, test_size

    path ='D:/publication/thesis modification/3rd try/'
    os.chdir(path)
    file_name_patent_classified = 'TFIDF_OSC_edited_by_hand.csv'
    test_size = 0.30
    return

def data_reading():
    dataset=pd.read_csv(path+file_name_patent_classified)
    return dataset

def dataset_splitting(dataset):
    from sklearn.model_selection import train_test_split
    pos_data=dataset[dataset.Class==1]
    neg_data=dataset[dataset.Class==0]
    #ratio_pos_data=len(pos_data)/len(dataset)
    train_set1, test_set1= train_test_split(pos_data, test_size= test_size)
    train_set2, test_set2 = train_test_split(neg_data, test_size=test_size)
    train_set=train_set1.append(train_set2)
    test_set=test_set1.append(test_set2)
    X_train = train_set.drop(["Patent_Number","Class"], axis=1)
    X_test = test_set.drop(["Patent_Number","Class"], axis=1)
    y_train = train_set["Class"]
    y_test = test_set["Class"]

    '''X_train.reset_index(drop=True,inplace=True)
    X_test.reset_index(drop=True,inplace=True)
    y_train.reset_index(drop=True,inplace=True)
    y_test.reset_index(drop=True,inplace=True)'''

    return train_set,test_set,X_train, y_train, X_test, y_test

def classifier_modelling(X_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    y_predict=neigh.predict(X_test)

    return y_predict

def report_patents(test_set,y_predict):
    test_set.insert(1,'Predicted',y_predict)
    #test_set.insert(0,'Patent_Numbers',dataset.Patent_Number.iloc[y_test.index])
    test_set.to_csv('report.csv',index=False)
    return

if __name__ == '__main__':
    settings()
    dataset=data_reading()
    train_set, test_set, X_train, y_train, X_test, y_test = dataset_splitting(dataset)
    y_predict=classifier_modelling(X_test, y_test)
    tn, fp, fn, tp = confusion_matrix(y_test,y_predict).ravel()
    print("tn:%i, fp:%i, fn:%i, tp:%i" %(tn,fp,fn,tp))
    report_patents(test_set,y_predict)




