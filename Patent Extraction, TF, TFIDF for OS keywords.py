import requests
import pandas as pd
import nltk
import numpy as np
import re
import string
from math import log, sqrt
import copy

def data_reading():

    onto_keywords=pd.read_csv(path + file_name_onto_keywords ).columns
    onto_syn_keywords= pd.read_csv(path+file_name_onto_syn_keywords).columns
    patent_numbers=pd.read_csv(path+file_name_patent_numbers)['Patent_Number']

    return patent_numbers, onto_keywords, onto_syn_keywords


def check_section_existence(text):
    sections_txt=[]
    for m in re.finditer('section itemprop=',text):
        end_sec_idx=(m.end())
        sections_txt.append(text[end_sec_idx:end_sec_idx+25])
    sections=[]
    for i in range(len(sections_txt)):
        if 'abstract' in sections_txt[i]:
            sections.append('abstract')
        elif 'description' in sections_txt[i]:
            sections.append('description')
        elif 'claims' in sections_txt[i]:
            sections.append('claims')
        elif 'application' in sections_txt[i]:
            sections.append('application')
        elif 'metadata' in sections_txt[i]:
            sections.append('metadata')
        elif 'family' in sections_txt[i]:
            sections.append('family')

    return sections


def extract_section(start_sign,end_sign,text):
    start_idx = []
    end_idx = []
    for m in re.finditer(start_sign, text):
        start_idx.append(m.end())
    for m in re.finditer(end_sign, text):
        end_idx.append(m.start())



    text_html=text[start_idx[0]:end_idx[0]]+'>' # I added '>' because I needed it for the next line when cleaner wants to remove html codes between <>
    cleaner = re.compile('<.*?>')
    section = re.sub(cleaner, '', text_html)
    return section


def patent_extraction(patent_numbers,onto_syn_keywords):
    all_keywords=[]
    TF_matrix = pd.DataFrame(patent_numbers, columns=['Patent_Number'])
    ps = nltk.stem.PorterStemmer()


    for patent_number in patent_numbers:
        url = 'https://patents.google.com/patent/US' + str(patent_number) + '/en'
        raw_html = requests.get(url).text
        stemmed_words = []
        sections = check_section_existence(raw_html) #section refers to abstract, description, and claims


        for section in sections:
            start_sign = section_signs[section_signs.section == section].start_sign.tolist()[0]
            start_sign_index = section_signs[(section_signs.section == section)].index

            if section =='abstract':
                if ('description' or 'claims') in sections:
                    end_sign = section_signs.start_sign.iloc[start_sign_index[0] + 1]
                else:
                    end_sign = section_signs[section_signs.section == 'application'].start_sign[3]
            elif section=='description':
                if 'claims' in sections:
                    end_sign = section_signs.start_sign.iloc[start_sign_index[0] + 1]
                else:
                    end_sign = section_signs[section_signs.section == 'application'].start_sign[3]

            elif section=='claims':
                end_sign = section_signs[section_signs.section == 'application'].start_sign[3]
            else:
                break


            try:
                cleaned_section=extract_section(start_sign, end_sign, raw_html)
            except:
                print('error in patent number', patent_number)
            tokenized_text = nltk.tokenize.word_tokenize(cleaned_section)


            filtered_words = []
            for w in tokenized_text:
                    if (w.lower() not in stop_words) and (w not in string.punctuation) and (w.isdigit()==False):
                        filtered_words.append(w.lower())


            for w in filtered_words:
                    stemmed_word= ps.stem(w)
                    if (stemmed_word in onto_syn_keywords):  #since onto_syn_keywords were stemmed, I had to filter in this line.
                        stemmed_words.append(stemmed_word)

            for word in stemmed_words:
                    if word not in all_keywords:
                        all_keywords.append(word)

        for keyword in all_keywords:
            if keyword not in TF_matrix.columns:
                TF_matrix[keyword] = pd.Series(np.zeros(len(patent_numbers)))
            TF_matrix.loc[TF_matrix.Patent_Number == patent_number, keyword]= stemmed_words.count(keyword)


    return TF_matrix


def tfidf_matrix(TF_matrix):
    number_of_patents= TF_matrix.shape[0]
    number_of_keywords= TF_matrix.shape[1]-1
    idf = np.zeros(number_of_keywords)
    num_docs_have_kw = []
    for j in range(1,number_of_keywords+1):
        num_docs_have_kw.append(0)
        for i in range(number_of_patents):
            if TF_matrix.iloc[i][j]!=0: num_docs_have_kw[j-1] = num_docs_have_kw[j-1] + 1

        idf[j-1] = log(float(number_of_patents) / float(num_docs_have_kw[j-1]), 10)

    TFIDF_matrix=copy.deepcopy(TF_matrix)
    sum_sq = []
    for i in range(0, number_of_patents):  # numerate patent numbers
        sum_sq.append(0)
        for j in range(1,number_of_keywords+1):  # numerate keywords
            sum_sq[i] = sum_sq[i] + int(float(TF_matrix.iloc[i][j])) * int(float(TF_matrix.iloc[i][j]))

        sum_sq[i] = sqrt((sum_sq[i]))
        if sum_sq[i]!=0:
            for j in range(1, number_of_keywords+1):
                TFIDF_matrix.set_value(i,TFIDF_matrix.columns[j],idf[j-1]*int(float(TF_matrix.iloc[i][j])) / sum_sq[i])


    return TFIDF_matrix


##########settings
path='D://publication//thesis modification//3rd try//'
file_name_onto_keywords='onto_keywords_TA_3rd_try.csv'
file_name_onto_syn_keywords='onto_syn_keywords_TA_3rd_try.csv'
file_name_patent_numbers = 'labeled_data.csv'

sections=['abstract','description','claims','application','metadata','family']
start_sign=['abstract">','class="description">','class="claim-text">','section itemprop="application','section itemprop="metadata','section itemprop="family']
section_signs=pd.DataFrame(np.array([sections,start_sign]).transpose(),columns=['section','start_sign'])

stop_words = set(nltk.corpus.stopwords.words("english"))

if __name__ == '__main__':
    patent_numbers, onto_keywords, onto_syn_keywords = data_reading()
    TF_matrix =patent_extraction(patent_numbers,onto_syn_keywords)
    TFIDF_matrix=tfidf_matrix(TF_matrix)
    TF_matrix.to_csv(path + 'TF_all_keywords.csv')
    TFIDF_matrix.to_csv(path+'TFIDF_OSC.csv')
    TFIDF_matrix.loc[:, onto_keywords.insert(0,'Patent_Number')].to_csv(path+'TFIDF_OS.csv')
