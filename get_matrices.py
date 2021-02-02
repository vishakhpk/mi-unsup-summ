import os
import sys
sys.path.insert(0, "./models/")
sys.path.insert(0, "./read-data/")
from read import get_text
from gpt2 import GPT2
from tqdm import tqdm
import preprocess_subsequence
import nltk
import math
from nltk import tokenize
import numpy as np
#from sklearn.metrics import precision_score, recall_score
import pickle
import spacy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--index")
parser.add_argument("--input_data")
parser.add_argument("--index_file")
parser.add_argument("--output_format")
args = parser.parse_args()
print(args.index)
print(args.index_file)
print(args.input_data)
print(args.output_format)

data = pickle.load(open(args.input_data, "rb"))#
print("Length ", len(data))
indices = pickle.load(open(args.index_file, "rb"))
print(indices)
lower, upper = indices[int(args.index)]
print(lower, upper)

nlp = spacy.load("en_core_web_sm")

model = GPT2(device="cuda", location="./path/to/saved/model/")

if os.path.exists(args.output_format+args.index+".pkl"):
    output = pickle.load(open(args.output_format+args.index+".pkl","rb"))
else:
    output = {}

def get_probabilities(articles):
    """
    Given a batch of articles (can be any strings) run a forward pass on GPT2 and obtain word probabilities for the same
    """
    article_splits = [article.split(" ") for article in articles]
    payload = model.get_probabilities(articles, topk = 20)
    res = [[] for i in range(len(articles))]
    for t, article in enumerate(articles):
        context = ""
        idx = 0
        chain = False
        next_word = ""
        article_words = article_splits[t]
        #print(article, article_words)
        word_probability = 1.0
        gt_count = 0
        idx+=1
        found_words = []
        for i, word in enumerate(payload["context_strings"][t][:-1]):
            context = context+" "+word
            probability = payload['real_probs'][t][i]#[1]
            next_word_fragment = payload["context_strings"][t][i+1]

            next_word += next_word_fragment
            #print(next_word, article_words[gt_count])
            if next_word == article_words[gt_count]:
                chain = False
                gt_count+=1
            else:
                chain = True

            word_probability *= probability
            assert word_probability <= 1.0, print(word_probability, context)
            if chain == False:      
                #print("Word Probability: ", word_probability, next_word)
                res[t].append(word_probability)
                word_probability = 1.0 
                next_word = ""
            #print(gt_count, len(article_words))
            if gt_count == len(article_words):
                break
    return res


def get_npmi_matrix(sentences, method = 1, batch_size = 1):
    """
    Accepts a list of sentences of length n and returns 3 objects:
    - Normalised PMI nxn matrix - temp
    - PMI nxn matrix - temp2
    - List of length n indicating sentence-wise surprisal i.e. p(sentence) - p 

    To optimize performance, we do the forward pass batchwise by assembling the batch and maintaining batch indices
    For each batch we call get_probabilities
    """
    temp = np.zeros((len(sentences), len(sentences)))
    temp2 = np.zeros((len(sentences), len(sentences)))
    batch_indices = {}
    batch = []
    batchCount = 0
    batchSize = batch_size
    #print(len(sentences))
    c = 0
    p = []
    for i in range(len(sentences)):
        result = get_probabilities([sentences[i]])
        try:
            p.append(sum([math.log(i) for i in result[0]]))
        except:
            print("Math domain error surprise", i)
            return temp, temp2, p
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i==j: 
                temp[i][j] = -1
                temp2[i][j] = -1
                continue
            article = sentences[i] + " "+ sentences[j]
            #print(article)
            batch_indices[str(i)+"-"+str(j)+"-"+str(len(sentences[i].split()))] = batchCount 
            batch.append(article)
            batchCount+=1
            
            if batchCount == batchSize or (i == len(sentences)-1 and j == len(sentences)-1):
                #print(batch)
                c+=1
                result = get_probabilities(batch)
                for key in batch_indices.keys():
                    #print(key)
                    #print(key.split("-"))
                    idx_i, idx_j, idx_l = [int(idx) for idx in key.split("-")]
                    try:
                        pxy = sum([math.log(q) for q in result[batch_indices[key]][idx_l:]])
                        py = p[idx_j]
                        px = p[idx_i]
                    
                        temp[idx_i][idx_j] = (pxy - py)/(-1*(pxy+px))
                        temp2[idx_i][idx_j] = (pxy - py)
                    except ZeroDivisionError:
                        print("Zero division error ", idx_i, idx_j)
                        temp[idx_i][idx_j] = -1
                        temp2[idx_i][idx_j] = -1
                    except:
                        print("Math Domain Error", i, j)
                    if temp[idx_i][idx_j] > 1 or temp[idx_i][idx_j] < -1:
                        print("Normalise assert ", temp[idx_i][idx_j], idx_i, idx_j)
                batchCount = 0
                batch = []
                batch_indices = {}
    return temp, temp2, p

def remove_unicode(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def get_article(idx):
    """
    For each document in the dataset, split it into sentences and call get_npmi_matrix to create the matrices
    """
    print(idx)
    article, abstract = data[idx]
    #sentences = tokenize.sent_tokenize(article)
    doc = nlp(article)
    sentences = [remove_unicode(sentence.text) for sentence in doc.sents]
    normalised, vanilla, surprise = get_npmi_matrix(sentences, batch_size = 10)
    #avg = get_pmi_matrix(sentences, method = 1)
    output[idx] = {}
    output[idx]["vanilla"] = vanilla
    output[idx]["normalised"] = normalised
    output[idx]["surprise"] = surprise
    #output[idx]["averaging"] = avg
    #pickle.dump(output, open("full_set_1.pkl", "wb"))
    return

"""
Main iteration loop, creates matrices for each document in the dataset
"""
c = 0
for idx in range(len(data)):
    if idx>=lower and idx<upper and idx not in output.keys():
        get_article(idx) 
    if c%20 == 0:
        pickle.dump(output, open(args.output_format+args.index+".pkl", "wb"))
    c+=1        

pickle.dump(output, open(args.output_format+args.index+".pkl", "wb"))
