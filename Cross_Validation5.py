# -*- coding: utf-8 -*-
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn import svm
#import pandas as pd
#from sklearn.cross_validation import KFold, cross_val_score
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
import csv
import sklearn.metrics as met
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import recall_score
#from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score
#import pdb 
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings('always')  # filter warnings because of fscore calculations; since the data set is imbalanced,
# the positive label may not be classified in many iterations, so f-score or precision cannot be calculated so it is considered 0.

def Cross_Valid(clf):
 
    specif=[]
    sens=[]
    Roc_auc =[]
    Acc=[]
    Prec=[]
    Fscor=[]
    
    for train, test in cv.split(X,y):
                clf.fit(X[train], y[train])
                pred_test = clf.predict(X[test])#X
                sensitivity=recall_score(y[test], pred_test, pos_label=1, average='binary')
                specificity=recall_score(y[test], pred_test, pos_label=0, average='binary')
                sens.append(sensitivity)
                specif.append(specificity)
                roc_auc = met.roc_auc_score(y[test], pred_test)
                Roc_auc.append(roc_auc)
                acc=met.accuracy_score(y[test],pred_test)
                Acc.append(acc)
                prec=precision_score(y[test],pred_test, pos_label=1, average='binary')
                Prec.append(prec)

                fscor=f1_score(y[test], pred_test, labels=None, pos_label=1, average='binary',sample_weight=None)
                Fscor.append(fscor)

    return (np.mean(specif), np.std(specif),np.mean(sens), np.std(sens), np.mean(Roc_auc),
        np.std(Roc_auc),np.mean(Acc), np.std(Acc),
        np.mean(prec), np.std(prec),np.mean(Fscor), np.std(Fscor))

 
    

#########################################################################################################
address=''
#address = 'D:\\publication\\thesis modification\\3rd try\\'
#address='/home/fmadani/Python/Finex/downloads/C22CB22D/'

file_name='TFIDF-extracted.csv'                  #Just enter the name of sample
num_folds=5
iteration=100

data = np.loadtxt(address+file_name,delimiter=',',skiprows=1)
sbj=data[:,1]
y=data[:,0]
X=data[:,2:]
cv = StratifiedKFold(n_splits=num_folds, shuffle=True)
number_of_measures_collected=12


with open(address+'\\'+'RC_3rd_try_evaluation'
                       '_SVM.csv', 'w') as csvfile:
     writer = csv.writer(csvfile,lineterminator='\n')
     row=['filename:',file_name,'number folds=',num_folds]
     writer.writerow(row)
     row=['classifier','Parameters','Specificity(mean)','Specificity(std)','Sensitivity(mean)', 'Sensitivity(std)',
          ''
          'ROC_AUC(mean)','ROC_AUC(std)','ACC(mean)','ACC(std)',
          'Precision(mean)','precision(std)','Fscor(mean)','Fscor(std)']
     writer.writerow(row)



############################## 100 Iteration      
       
     for u in range(iteration):


        #######################           SVM
        print('Grid search for SVM')
        Gamma = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
        Ci = [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 10, 20, 50, 100]  # Ci represents parameter C

        for c in Ci:
            for g in Gamma:
                for k in ['rbf', 'linear']:
                    clf = svm.SVC(kernel=k, class_weight="balanced", C=c, gamma=g)
                    a = Cross_Valid(clf)
                    # row=[str(a[0]),str(a[1]),str(a[2]),str(a[3]),str(a[4]),str(a[5]),str(a[6]),"SVM",'  gamma='+str(g)]
                    row = []
                    row.append("SVM")
                    titr = ('C=', c, 'gamma=' + str(g), 'Kernel=' + k)
                    row.append(titr)
                    for j in range(number_of_measures_collected): row.append(str(a[j]))
                    writer.writerow(row)

