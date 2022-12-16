import os
import numpy as np
import pandas as pd
import math
from nltk.tokenize import word_tokenize
from natsort import natsorted
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def lemitization(tokens):
    lem = WordNetLemmatizer()
    result = []
    for token in tokens:
        result.append(lem.lemmatize(token))

    return result


#  * First part *  point 1
def read_file(path):
    f = open(path, "r")
    data = f.read()
    f.close()
    return data

X = ['in', 'to', 'where']

stop_word = list(stopwords.words('english'))
new_stop_words = [i for i in stop_word if i not in X]


def tokenized_documents(data):
    token = []
    tokens = word_tokenize(data)
    for word in tokens:
        if word not in new_stop_words:
            token.append(word.lower())

    return token


def query_optimization(query):
    result = {}
    final = []
    q = len(query)
    if (q==0):
        return  final
    elif q==1:
        return pos_index[query[0]][1].keys()
    else:
        for docid in pos_index[query[0]][1].keys(): # 1 2 6
            if docid in pos_index[query[1]][1].keys(): # 1 2 4
                result[docid] = 0
                for index in pos_index[query[0]][1][docid]:  # [0]
                    if index + 1 in pos_index[query[1]][1][docid]:  # [1]
                        result[docid] = index + 1  # 1
                    else:
                        del result[docid]  # {1: 1 ,  2:1}
        for i in range(2, q, 1):

            keys = list(result.keys())  # 1  2
            for docid in keys: # doc 1
                if docid in pos_index[query[i]][1].keys(): # 1 2 4 5 6
                    index = result[docid]   # value of doc 1 ---> 1
                    if index + 1 in pos_index[query[i]][1][docid]:   # next pos --> 2
                        result[docid] = index + 1
                    else:
                        del result[docid]
                else:
                    del result[docid]
        for r in result.keys():
            final.append(r)   #  1  2

        return final
    return final

files_data = []
projectPath = "D:\\Projects\\IR project"
folder_names = ["DocumentCollection"]
file_num = 1
names = []
pos_index = {}
file_sizes = {}
for folder_name in folder_names:
    file_names = natsorted(os.listdir(projectPath + "/" + folder_name))

    for file_name in file_names:

        stuff = read_file(projectPath + "/" + folder_name + "/" + file_name)
        files_data.append(stuff)

        names.append(file_name)

        final_token_list = tokenized_documents(stuff)
        final_token_list = lemitization(final_token_list)

        file_sizes[file_name] = len(final_token_list)


        for pos, term in enumerate(final_token_list):

            # If term already exists in the positional index dictionary.
            if term in pos_index:

                
                

                # Check if the term has existed in that DocID before.
                if file_name in pos_index[term][1]:
                    # Increment total freq by 1.
                    pos_index[term][0] = pos_index[term][0] + 1
                    
                    pos_index[term][1][file_name].append(pos)

                # add new doc in pos index
                else:
                    pos_index[term][1][file_name] = [pos]

            # If term does not exist in the positional index dictionary   (first encounter).
            else:

                pos_index[term] = []
                # The total frequency is 1.
                pos_index[term].append(1)
                # The postings list is initially empty.
                pos_index[term].append({})
                # Add doc ID to postings list.
                pos_index[term][1][file_name] = [pos]

        # Increment the file no. counter
        file_num += 1

num_file = file_num - 1

print("\n Positional Index :\n")
print(pos_index)

print("\nTF.IDF matrix : \n")

# setting the new stop words list to exclude in, to, where
vectorizer = TfidfVectorizer(stop_words=new_stop_words)

# the function to get the tf-idf table
vectorizer.fit_transform(files_data)

# feature name gets only the header of the table
feature_names = lemitization(vectorizer.get_feature_names_out())

# creating a matrix shape to store the tf idf in instead of a table, the matrix is empty at first
df = pd.DataFrame(0, columns=feature_names, index=names)

for term in pos_index:
    dic_of_term = (pos_index[term][1])
    docs = list(dic_of_term.keys())  # 1 2 6

    idf = math.log10(num_file / len(docs))

    for doc in docs:   # 1 ..... 2 ..... 6
        # term frequency
        tf = len(pos_index[term][1][doc])  # len --> [0]
        # Document frequency
        DF = len(dic_of_term[doc])
        # term frequency weight
        tfw = 1 + math.log10(tf)
        # Generate TF.IDF Matrix
        tf_idf = float(tfw) * float(idf)
        # insert in dataframe new value
        df.at[doc,term] = tf_idf

        #print("{} --->  TF : {}  in Document : {}  IDF :{} ".format(term,tf,doc,idf))

dff = df.transpose()
print(dff)

doc_length = {}
# calculate document length
for txt in dff:  # loop on each column   ....> 1.txt  ... 2.txt ... 3.txt  .... 4.txt ... etc
    y = 0.0
    x = list(dff[txt]) # list of each column
    for value in x:
        y = y + (value * value)
    doc_length[txt] = math.sqrt(y)

#print("Document length ".format(doc_length))

while True:
    # if enter true query
    try:
        query = (input(' Enter a phrase query  : '))
        tok_query = tokenized_documents(query.lower())
        lem_query = lemitization(tok_query)
        result_list = query_optimization(lem_query)   # return matched document after optimization

        # check query empty
        if len(result_list) == 0:
            print(" Can not Find matched !! Document Enter other Query :  ")

        else:

            q = [query]
            q_vector = vectorizer.transform(q).toarray() .reshape(dff.shape[0])
            qf = pd.DataFrame(q_vector, index=feature_names)

            cos_similarity_doc = {}
            for i in result_list:
                cos_similarity_doc[i] = 0 # 1 : 0
                for token in lem_query:
                    cos_similarity_doc[i] = cos_similarity_doc[i] + (
                        # normalized of term in doc
                        (dff[i][token] / doc_length[i])
                        # normalized of term in query
                        * qf[0][token])
            # sort dic in asc
            sorted_dic = dict(sorted(cos_similarity_doc.items(),key=lambda x:x[1],reverse=True))

            print("\nThe Cosine Similarity Of Matched Document is :\n {}".format(sorted_dic))

            print("\n Rank of matched Document :\n")

            for document, score in sorted_dic.items():
                print(" Document {}           Score : {}".format(document, score))


            print("\n Result: ")
            for document in sorted_dic.keys():
                print(" Document {} ".format(document), end=" ")

        restart = input('\nWould you like to restart? Enter yes or no.\n')
        if restart.lower() != 'yes':
            break

    except KeyError:
        print(" Can not Find matched !! Document Enter other Query :  ")
