# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm, neighbors
import csv

   

###########################Preparing data & Classifiers
#ddress = 'D:\\ETM\\Thesis\\Research\\Finex\\Prediction\\'
address='/home/fmadani/Python/Finex/downloads/C22CB22D/Prediction/'

file_name_TA='TFIDF_onto_syn_TA_Classified_patents.csv'
file_name_MN='TFIDF_onto_syn_MN_Classified_patents.csv'
file_name_pre_TA='TFIDF_onto_sys_keywords_TA.csv'
file_name_pre_MN='TFIDF_onto_sys_keywords_MN'

######################### opening files


with open(address+file_name_TA, 'r') as csvfile:
    csv1 = csv.reader(csvfile, delimiter=',', quotechar='|')
    data_TA=[]   
    
    for row in csv1:
        data_TA.append(row)

with open(address+file_name_MN, 'r') as csvfile:
    csv2 = csv.reader(csvfile, delimiter=',', quotechar='|')
    data_MN=[]   
    
    for row in csv2:
        data_MN.append(row)

with open(address+file_name_pre_TA, 'r') as csvfile:
    csv3 = csv.reader(csvfile, delimiter=',', quotechar='|')
    data_pre_TA=[]   
    
    for row in csv3:
        data_pre_TA.append(row)


with open(address+file_name_pre_MN, 'r') as csvfile:
    csv4 = csv.reader(csvfile, delimiter=',', quotechar='|')
    data_pre_MN=[]   
    
    for row in csv3:
        data_pre_MN.append(row)

###################################### fitting classifiers 
y_TA=[]        
keywords_TA=data_TA[0][2:]
for i in range(1,len(data_TA)): y_TA.append(data_TA[i][0])
X_TA=data_TA[1:][0:]
X_TA=np.delete(X_TA,[0,1],axis=1)        #deleting class and patent number columns

clf_TA = neighbors.KNeighborsClassifier(n_neighbors=3)
clf_TA.fit(X_TA,y_TA)
         

y_MN=[]
#keywords_MN=data_MN[0,2:]
for i in range(1,len(data_MN)): y_MN.append(data_MN[i][0])
X_MN=data_MN[1:][0:]
X_MN=np.delete(X_MN,[0,1],axis=1)        #deleting class and patent number columns


clf_MN =  svm.SVC(kernel='rbf',class_weight="balanced", C=100,gamma=.1)
clf_MN.fit(X_MN,y_MN)



'''
########################## Extracting Classified TA patents
TFIDF_TA=np.zeros([len(data_pre_TA)-1,len(data_MN[0])-2])


for j in range(len(data_MN[0])):
    if data_MN[0][j] in data_pre_TA[0]:
        iindex=data_pre_TA[0].index(data_MN[0][j])
        for i in range(1,len(data_pre_TA)):
            TFIDF_TA[i-1][j-2]=data_pre_TA[i][iindex]



########################## Extracting Classified MN patents
TFIDF_MN=np.zeros([len(data_pre_MN)-1,len(data_TA[0])-2])

for j in range(2,len(data_TA[0])):
    if data_TA[0][j] in data_pre_MN[0]:
        iindex=data_pre_MN[0].index(data_TA[0][j])
        for i in range(1,len(data_pre_MN)):
            TFIDF_MN[i-1][j-2]=data_pre_MN[i][iindex]

'''


################################      Prediction
y_TA_pre=clf_TA.predict(data_pre_TA)
y_MN_pre=clf_MN.predict(data_pre_MN)



################################ wrinting patents to csv file
with open(address+'\\'+'Opportunities.csv', 'w') as csvfile:                                      
     writer = csv.writer(csvfile,lineterminator='\n')
     row=['TA Class:','MN Class','Patent Number']
     writer.writerow(row)
     for i in range(1,len(y_TA_pre)):
        row=[]
        row.append(y_TA_pre[i])
        row.append(y_MN_pre[i])
        row.append(data_pre_TA[i][0]) 
        writer.writerow(row)
     

    
