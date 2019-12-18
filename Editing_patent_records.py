import pandas as pd
import sys
import pdb

'''def find_PN():
    if len(data_destiny[data_destiny.Patent_Number==patent_number])==0:
        PN_not_found=PN_not_found.append([patent_number])
    else:
        row_to_be_copied=data_destiny[data_destiny.Patent_Number == int(patent_number)]
        data_origin=data_origin.append(row_to_be_copied)

def change_class(patent_number):
    data.Class[data.Patent_Number == patent_number] = 0

def del_patent(patent_number):
    data.drop(patent_number, axis=0, inplace=True)
'''

#path=os.getcwd()+'\\'
#path='D:\\publication\\thesis modification\\3rd try\\'
path='C:\\Users\\fereshteh\\Documents\\Farshad - Thesis\\'
#file_name='TFIDF_Classified_patents_TA.csv' #TFIDF_Classified_patents_TA

file_destiny='TFIDF_for_TF_matrix_after_removing_cutoff.csv'#destination.csv'#TFIDF_Classified_patents_TA_modified.csv'#TFIDF_for_TF_matrix_after_removing_cutoff.csv'##'
#file_origin= 'TFIDF_Classified_patents_TA_modified.csv'#'origin.csv'#
file_labeled_data='labeled_data.csv'
labeled_data=pd.read_csv(path+file_labeled_data)
#data_destiny=pd.read_csv(path+file_destiny)
#data_origin=pd.read_csv(path+file_origin)

labeled_data=labeled_data.rename(columns={"Patent Number": "Patent_Number"})
#data_origin=data_origin.rename(columns={"Patent Number": "Patent_Number"})
print('BEFORE')
print('labeled data',labeled_data.shape)
#print('destiniation',data_destiny.shape)
#print('origin',data_origin.shape)

data_origin=pd.DataFrame()
PN_not_found=pd.DataFrame()
chunksize = 500
for i in range(len(labeled_data.Patent_Number)):
    patent_number =int(labeled_data.Patent_Number[i])

    for data_destiny in pd.read_csv(path+file_destiny, chunksize=chunksize):
        data_destiny = data_destiny.rename(columns={"patent number": "Patent_Number"})
        sys.stdout.write('\r %d out of %d are processed' % (i, len(labeled_data.Patent_Number)))
        #find_PN()
        if (data_destiny.Patent_Number==patent_number).any():
            row_to_be_copied = data_destiny[data_destiny.Patent_Number == int(patent_number)]
            row_to_be_copied.insert(0, "Class", labeled_data.Class[i], True)
            data_origin = data_origin.append(row_to_be_copied)
            break



data_origin.to_csv(path+'new_origin.csv',index=False)

for i in range(len(labeled_data.Patent_Number)):
    patent_number =int(labeled_data.Patent_Number[i])
    if (patent_number==data_origin.Patent_Number).any()==False:
        PN_not_found = PN_not_found.append([patent_number])

if len(PN_not_found)!=0: PN_not_found.to_csv(path+'not_found_PNS.csv')

print('AFTER')
print('labeled data',labeled_data.shape)
print('destiniation',data_destiny.shape)
print('origin',data_origin.shape)


data_origin.to_csv(path+'data_origin_updated.csv',index=False)
if len(PN_not_found)!=0:
    PN_not_found.to_csv(path+'PN_not_found.csv',index=False)
else:
    print("All patents are found and added!")

#print(data.shape)

#data.to_csv(path+'TFIDF_Classified_patents_TA_modified.csv',index=False)

#data=data.rename(columns={"Patent Number": "Patent_Number"})


### Change class from 1 to 0 for three patents
'''change_class(4129443)
change_class(4035159)
change_class(6705848)'''

## Delete specific patent Number
'''data = data.set_index("Patent Number").astype(str)
del_patent(...)'''

