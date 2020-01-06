import numpy as np
import csv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import sklearn.metrics as met
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier

#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import recall_score
#from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_recall_curve
import os

def settings():
    global file_name_labeled_patent, file_name_unlabeled_patent,path, test_size, n_neighbors,seudo_labeled_data_size
    global iteration, measures,measures_list,Ci,Gamma, num_folds, n_estimators
    path ='D:/publication/thesis modification/3rd try/Semi-supervised classification/'
    os.chdir(path)

    file_name_labeled_patent = 'new_TFIDF.csv'
    file_name_unlabeled_patent='TFIDF_onto_sys_keywords_TA.csv'
    test_size = 0.30
    n_neighbors=range(3,11)
    Ci=[1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 10, 20, 50, 100]
    Gamma = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    n_estimators=10 #random forest parameter
    seudo_labeled_data_size=500
    num_folds=3
    iteration=100
    measures_list=['type','model','sensitivity','specificity','roc_auc','accuracy','precision','f_score']
    measures=pd.DataFrame(columns=measures_list,index=None)
    return

def data_reading():
    labeled_dataset=pd.read_csv(path+file_name_labeled_patent)
    unlabeled_dataset = pd.read_csv(path + file_name_unlabeled_patent)
    return labeled_dataset, unlabeled_dataset

def dataset_splitting(dataset,fields_to_be_dropped):
    from sklearn.model_selection import train_test_split
    pos_data=dataset[dataset.Class==1]
    neg_data=dataset[dataset.Class==0]
    #ratio_pos_data=len(pos_data)/len(dataset)
    train_set1, test_set1= train_test_split(pos_data, test_size= test_size)
    train_set2, test_set2 = train_test_split(neg_data, test_size=test_size)
    train_set=train_set1.append(train_set2)
    test_set=test_set1.append(test_set2)
    X_train = train_set.drop(fields_to_be_dropped, axis=1)
    X_test = test_set.drop(fields_to_be_dropped, axis=1)
    y_train = train_set["Class"]
    y_test = test_set["Class"]

    return train_set,test_set,X_train, y_train, X_test, y_test

def classifier_modelling(clf,X_train, y_train,X_test):

    clf.fit(X_train, y_train)
    y_predict=clf.predict(X_test)

    return y_predict, clf

def report_patents(test_set,y_test,y_predict,file_name_report):
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    print("tn:%i, fp:%i, fn:%i, tp:%i" % (tn, fp, fn, tp))
    test_set.insert(1,'Predicted',y_predict)
    #test_set.insert(0,'Patent_Numbers',dataset.Patent_Number.iloc[y_test.index])
    test_set.to_csv(file_name_report,index=False)
    test_set=test_set.drop(['Predicted'],axis=1)
    return test_set


def data_pseudo_labeling(unlabeled_dataset,neigh,keywords):
    from sklearn.utils.random import sample_without_replacement
    index_pseudo_data=sample_without_replacement(len(unlabeled_dataset), seudo_labeled_data_size)
    pseudo_data=unlabeled_dataset.iloc[sorted(index_pseudo_data)]
    pseudo_predict = neigh.predict(pseudo_data[keywords])
    pseudo_data.insert(0,'Class',pseudo_predict)
    selected_columns=keywords.insert(0,'Patent_Number')
    selected_columns=selected_columns.insert(0,'Class')
    pseudo_labeled_data=pseudo_data.filter(selected_columns)

    return pseudo_labeled_data


def measures_colection(type,model,y_test, y_predict):
    global measures
    sensitivity = met.recall_score(y_test, y_predict, pos_label=1, average='binary')
    specificity = met.recall_score(y_test, y_predict, pos_label=0, average='binary')
    roc_auc = met.roc_auc_score(y_test, y_predict)
    acc = met.accuracy_score(y_test, y_predict)
    prec = met.precision_score(y_test, y_predict, pos_label=1, average='binary')
    fscor = met.f1_score(y_test, y_predict, labels=None, pos_label=1, average='binary', sample_weight=None)
    new_row={'type':type,'model':model,'sensitivity':sensitivity,'specificity':specificity,'roc_auc':roc_auc,
             'accuracy':acc,'precision':prec,'f_score':fscor}
    measures=measures.append(new_row,ignore_index=True)

    return

def semi_supervised_classification(clf,labeled_dataset,unlabeled_dataset):

    for _ in range(iteration):

        #train_set, test_set, X_train, y_train, X_test, y_test = dataset_splitting(labeled_dataset,["Patent_Number","Class"])
        cv = StratifiedKFold(n_splits=num_folds, shuffle=True)
        X= labeled_dataset.drop(['Patent_Number','Class'],axis=1)
        y=labeled_dataset['Class']
        for train, test in cv.split(X,y):
            y_predict,model=classifier_modelling(clf,X.iloc[train], y.iloc[train],X.iloc[test])
            measures_colection('original',model,y[test], y_predict)
            #test_set=report_patents(test_set,y_test,y_predict,'original_labeled_data_report.csv')

        indices_original_test=test
        pseudo_labeled_data=data_pseudo_labeling(unlabeled_dataset,clf,X.columns)
        new_labeled_dataset = pd.concat([labeled_dataset, pseudo_labeled_data])
        new_X=new_labeled_dataset.drop(['Patent_Number','Class'],axis=1)
        new_y=new_labeled_dataset['Class']
        for train, test in cv.split(new_X, new_y):
            #new_train_set, new_test_set, new_X_train, new_y_train, new_X_test, new_y_test = dataset_splitting(new_labeled_dataset,["Patent_Number","Class"])
            new_y_predict, new_model = classifier_modelling(clf,new_X.iloc[train], new_y.iloc[train],X.iloc[indices_original_test]) #pay attention here that we used
            #X.iloc[test] because we want to predict the performance on the same test dataset for both regular and pseudo datasets
            #report_patents(test_set, y_test, new_y_predict,'new_labeled_data_report.csv')
            measures_colection('pseudo',new_model,y.iloc[indices_original_test], new_y_predict)

if __name__ == '__main__':
    settings()
    labeled_dataset,unlabeled_dataset=data_reading()

    for n in n_neighbors:
        clf = KNeighborsClassifier(n_neighbors=n)
        semi_supervised_classification(clf, labeled_dataset, unlabeled_dataset)

    for c in Ci:
        for g in Gamma:
            for k in ['rbf', 'linear']:
                clf = SVC(kernel=k, class_weight="balanced", C=c, gamma=g)
                semi_supervised_classification(clf, labeled_dataset, unlabeled_dataset)

    for i in range(2,n_estimators+1):
        clf=RandomForestClassifier(n_estimators=i)
        semi_supervised_classification(clf, labeled_dataset, unlabeled_dataset)


    clf = DummyClassifier()
    semi_supervised_classification(clf, labeled_dataset, unlabeled_dataset)

    measures.to_csv('report_dummy.csv',index=False)
