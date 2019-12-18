
#############################################################################

#             This module only calculate TF-IDF, and normalize it
#############################################################################



import csv
import numpy as np
import sys
from math import log, sqrt
##########bar progress functions

def bar_prog(i,total):
   #point=(total/100)
   #increment=(total/100)
   #if ((i % 5*point)==0):
    #  	   sys.stdout.write("\r["+"="*(i/(increment)) + " "*((total-i)/(increment)) + "]" + str((i/(point))) + "%")
     #  	   sys.stdout.flush()
     aa=1
###############################

#path='D:\\ETM\\Thesis\\Research\\Finex\\downloads\\C22CB22D\\Prediction\\'
path='/home/fmadani/Python/Finex/downloads/C22CB22D/Prediction/'

input_filename1='TF_onto_keywords_TA.csv'
output_filename1='TFIDF_onto_keywords_TA.csv'


###########
print 'Reading files'


with open(path+input_filename1, 'r') as csvfile:
    csv1 = csv.reader(csvfile, delimiter=',', quotechar='|')
    f1=[]   
    
    for row in csv1:
        f1.append(row)



idf=np.zeros([len(f1[0])])
num_docs_have_kw=[]
for j in range(0,len(f1[0])):
    bar_prog(j,len(f1[0]))
    num_docs_have_kw.append(0)    
    for i in range(len(f1)):
        if f1[i][j] >0: num_docs_have_kw[j]=num_docs_have_kw[j]+1
  
    idf[j]=log(float(len(f1))/float(num_docs_have_kw[j]),10)


print  "tf normalization"
sum_sq=['sum square']
for i in range(1,len(f1)):         #numerate patent numbers
       bar_prog(i,len(f1))       
       sum_sq.append(0)
       for j in range(1,len(f1[0])):            # numerate keywords
             sum_sq[i]=sum_sq[i]+int(float(f1[i][j]))*int(float(f1[i][j]))

       sum_sq[i]=sqrt((sum_sq[i])) 
       if sum_sq[i]<>0:
	     for j in range(1,len(f1[0])):
	     	f1[i][j]=int(float(f1[i][j]))/sum_sq[i]


#############################################################preparing CSV files
print " "
print "preparing csv files"


with open(path+'\\'+output_filename1, 'w') as csvfile:                                      
    writer = csv.writer(csvfile,lineterminator='\n')
    writer.writerows(f1)



