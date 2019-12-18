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
from sklearn.cross_validation import StratifiedKFold
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score
#import pdb 
from sklearn.dummy import DummyClassifier
import sys


##########bar progress functions

def bar_prog(i,total):
   point=(total/100)
   increment=(total/100)
   if ((i % 5*point)==0):
      	   sys.stdout.write("\r["+"="*(i/(increment)) + " "*((total-i)/(increment)) + "]" + str((i/(point))) + "%")
       	   sys.stdout.flush()




def Cross_Valid(num_folds,clf):
 
    specif=[]
    sens=[]
    Roc_auc =[]
    Acc=[]
    Recall=[]
    Prec=[]
    Fscor=[]
    cv = StratifiedKFold(y, n_folds=num_folds, shuffle=True)
 
    for train, test in cv:
                clf.fit(X[train], y[train])
                pred_test = clf.predict(X[test])#X        
                T = pred_test[np.where(y[test]==1)[0]] #y
                tp=sum(T)
                sensitivity = np.float(tp)/( len(T) + 1e-16)
                sens.append(sensitivity)
                N = pred_test[np.where(y[test]==0)[0]]
                tn = len(N) - sum(N)
                specifity = np.float(tn)/(len(N) + 1e-16)
                specif.append(specifity)
                roc_auc = met.roc_auc_score(y[test], pred_test)
                Roc_auc.append(roc_auc)
                acc=met.accuracy_score(y[test],pred_test)
                Acc.append(acc)
                rec=recall_score(map(float,y[test]),map(float,pred_test) , average=None)
                Recall.append(rec)
                prec=precision_score(map(float,y[test]),map(float,pred_test) , average=None)
                Prec.append(prec)
                fscor=f1_score(y[test], pred_test, labels=None, pos_label=1, average='binary',sample_weight=None)
                Fscor.append(fscor)

    return (np.mean(specif), np.std(specif),np.mean(sens), np.std(sens), np.mean(Roc_auc),
        np.std(Roc_auc),np.mean(Acc), np.std(Acc),np.mean(Recall), np.std(Recall),
        np.mean(prec), np.std(prec),np.mean(Fscor), np.std(Fscor))

 
    

#########################################################################################################
#address = 'D:\\ETM\\Thesis\\Research\\Finex\\Classification\\TA\\2nd_Try\\meysam\\'
address='/home/fmadani/Python/Finex/downloads/C22CB22D/Prediction/Classification/'

file_name='TFIDF_onto_syn_TA_Classified_patents.csv'
num_folds=5
iteration=100

data = np.loadtxt(address+file_name,delimiter=',',skiprows=1)
sbj=data[:,1]
y=data[:,0]
X=data[:,2:]

         
with open(address+'\\'+'TA_onto_syn_new_normalized_TFIDF.csv', 'w') as csvfile:                                      
     writer = csv.writer(csvfile,lineterminator='\n')
     row=['filename:',file_name,'number folds=',num_folds]
     writer.writerow(row)
     row=['classifier','Parameters','Specifity(mean)','Specifity(std)','Sensitivity(mean)', 'Sensitivity(std)',
          'ROC_AUC(mean)','ROC_AUC(std)','ACC(mean)','ACC(std)','Recall(mean)','Recall(std)',
          'Precision(mean)','precision(std)','Fscor(mean)','Fscor(std)']
     writer.writerow(row)



############################## 100 Iteration      
       
     for u in range(iteration):

	bar_prog(100*u,100*iteration)
        
            
    
###################### Random Forest  
        #print 'Grid search for Random Forst'
        for i in range(4,20):
                 #cv = StratifiedKFold(y, n_folds=num_folds, shuffle=True)
		 clf = RandomForestClassifier(n_estimators=i)    
                 a=Cross_Valid(num_folds,clf)        
                 #row=[str(a[0]),str(a[1]),str(a[2]),str(a[3]),str(a[4]),str(a[5]),str(a[6])
                 #                    ,str(a[1]),'RF','n_estimator=',i]
                 row=[]
                 row.append('RF')
                 row.append('n_estimator='+str(i))
                 for j in range(14):row.append(str(a[j]))
                 writer.writerow(row)
        
'''
#####################         Dummy        
        #cv = StratifiedKFold(y, n_folds=num_folds, shuffle=True)
	clf = DummyClassifier(strategy='stratified',random_state=0)
        a=Cross_Valid(num_folds,clf)
        row=[]
        row.append('Dummy')
        row.append('')
        for i in range(14):row.append(str(a[i]))
        writer.writerow(row)

###################### KNN       
        #print 'Grid search for kNN'
        
        for i in range(3,10):
            #cv = StratifiedKFold(y, n_folds=num_folds, shuffle=True)
	    clf = neighbors.KNeighborsClassifier(n_neighbors=i)
            a=Cross_Valid(num_folds,clf)        
            #row=[str(a[0]),str(a[1]),str(a[2]),str(a[3]),str(a[4]),str(a[5]),str(a[6]),"KNN, n="+str(i)]
            row=[]
            row.append('KNN')
            row.append('n='+str(i))
            for j in range(14):row.append(str(a[j]))
            writer.writerow(row)




#######################           SVM
        #print 'Grid search for SVM'
        Gamma=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
        Ci=[1e-9,1e-7,1e-5,1e-3,1e-2,1e-1,10,20,50,100]  #Ci represents parameter C
            
        for c in Ci:      
                for g in Gamma:
                    for k in ['rbf','linear']:
                        #cv = StratifiedKFold(y, n_folds=num_folds, shuffle=True)
			clf = svm.SVC(kernel=k,class_weight="balanced", C=c,gamma=g)
                        a=Cross_Valid(num_folds,clf)
                        #row=[str(a[0]),str(a[1]),str(a[2]),str(a[3]),str(a[4]),str(a[5]),str(a[6]),"SVM",'  gamma='+str(g)]
                        row=[]
                        row.append("SVM")
                        titr=('C=',c,'gamma='+str(g), 'kernel=',k)
                        row.append(titr)
                        for j in range(14):row.append(str(a[j]))
                        writer.writerow(row)

'''

