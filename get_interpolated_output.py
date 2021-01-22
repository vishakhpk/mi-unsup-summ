import random
import pickle
import spacy
import numpy as np
nlp = spacy.load("en_core_web_sm")

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--classifier")
parser.add_argument("--output_file")
parser.add_argument("--dataset_file")
parser.add_argument("--matrix_file_pattern")

args = parser.parse_args()

clf = pickle.load(open(args.classifier, "rb"))

data = pickle.load(open(args.dataset_file, "rb"))

for file_idx in range(5):
    count = 0
    output = pickle.load(open(args.matrix_file_pattern+str(file_idx)+".pkl", "rb"))
    for key in output.keys():
        #if key not in ref.keys():
        #    continue
        count+=1
        article, abstract = data[key]
        print("--------------------")
        print(key)
        selected = []
        doc = nlp(article)
        sentences = [sentence.text.strip() for sentence in doc.sents]

        matrix = output[key]['vanilla']
        matrix[matrix<0] = 0 
        relevance = []
        surprise = output[key]['surprise']
        for idx in range(len(sentences)):
            relevance.append(sum(matrix[idx]))

        penalty = [0 for i in range(len(sentences))]
        #print(surprise)
        #print(matrix)
        try:
            for j in range(1, 9):
                selected = []
                summary = ""
                for k in range(j):
                    maxIdx = -1
                    maxVal = -float('inf')
                    #print(maxIdx, selected)
                    for i in range(len(sentences)):
                        temp = np.dot(clf.coef_[0], [penalty[i], relevance[i]])
                        if temp > maxVal and i not in selected:
                            maxIdx = i
                            maxVal = temp 

                    #print(maxVal, maxIdx)
                    for i in range(len(sentences)):
                        penalty[i]+=matrix[i][maxIdx]

                    selected.append(maxIdx)
                #print(selected) 
                summary = ""
                for i in sorted(selected):
                    summary+= sentences[i]+" " 

                summary = ' '.join(summary.split())

                with open(args.output_file+str(j), "a") as f:
                    f.write(summary+'\n')

            with open("./path/to/output/gold", "a") as f:
                f.write(' '.join(abstract.split())+'\n')
        except:
            print("Missed ", key)

        #print("SUMMARY ", summary)
        
        #if count == 10:
        #    exit(0) 

