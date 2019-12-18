'''
In this module, I extract onto keywords and onto syn keywords from TF matrix,
because the TFIDF of onto and onto_syn should be normaliazed separately.
'''

import pandas as pd
import sys
import pdb

def show_progress(process_name,i,total):
    sys.stdout.write('\r%s: %d out of %d are processed' % (process_name,i,total))

def check_existence_of_keywords(keywords):
    existing_keywords = []
    data = pd.read_csv(path + input_filename1, chunksize=2)
    df=data.get_chunk(1)
    for keyword in keywords:
         if keyword in df.columns: existing_keywords.append(keyword)

    #existing_keywords=pd.Series(existing_keywords)
    return existing_keywords

def extract_new_rows(existing_keywords):
    df = pd.DataFrame(columns=['Class', 'Patent_Number']+existing_keywords)
    iterator=0
    for chunk in pd.read_csv(path + input_filename1, chunksize=chunksize):
        iterator+=1
        show_progress('chunk processing',iterator,int(65000/chunksize))
        for patent_number in patent_numbers:
            if patent_number in chunk['patent number'].tolist():
                df1 = labled_data[labled_data.Patent_Number == patent_number]
                df2 = chunk[chunk['patent number'] == patent_number].filter(items=existing_keywords)
                df2.insert(loc=0, column='Patent_Number', value=patent_number)
                new_row = pd.merge(df1,df2,on='Patent_Number')
                df = df.append(new_row)
                #pdb.set_trace()
                break
        if len(patent_numbers)==df.shape[0]: break

    return (df)

def export_csvs(df_new_rows):
    existing_onto_keywords=check_existence_of_keywords(onto_keywords)
    df_new_rows_onto=df_new_rows[existing_onto_keywords]
    df_new_rows_onto.to_csv(path+output_filename1, index=None, header=True) #ontological keywords
    df_new_rows.to_csv(path+output_filename2, index=None, header=True)      #ontological keywords +their synonyms



################## settings
#path='/home/fmadani/Python/Finex/downloads/C22CB22D/Prediction/'
path='D:\\publication\\thesis modification\\3rd try\\'
chunksize = 10**2

################## filenames
input_filename1='TFIDF.csv'
input_filename2='onto_keywords_TA_3rd_try.csv'
input_filename3='onto_syn_keywords_TA_3rd_try.csv'
input_filename4='labeled_data.csv'

output_filename1='TF_onto_keywords_TA_3rd_try.csv'
output_filename2='TF_onto_sys_keywords_TA_3rd_try.csv'



################## Data reading
print('Reading files')
onto_keywords=pd.read_csv(path+input_filename2)
onto_syn_keywords=pd.read_csv(path+input_filename3)
labled_data=pd.read_csv(path+input_filename4)
patent_numbers=labled_data.Patent_Number.tolist()



################## main module
if __name__ == '__main__':
    existing_keywords=check_existence_of_keywords(onto_syn_keywords)
    df_new_rows=extract_new_rows(existing_keywords)
    export_csvs(df_new_rows)










