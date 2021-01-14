# Unsupervised Extractive Summarization using Mutual Information

## Steps to perform summarization 

1. Fine tune the required language model
2. Create sentence-wise mutual information matrices for documents in the dataset
3. Use the created matrices to perform extractive summarization
4. Evaluation

### Fine tune the required language model

We use GPT2-large as provided by [HF transformers](https://github.com/huggingface/transformers) with a standard fine tuning script as provided in the model directory.

### Creating mutual information matrices 

For each document in the dataset, if there are n sentences, we need to create the nxn pairwise matrix as described in the paper. <br/>

```
python get_matrices.py --index 0 --input_data ../../Summarization/dataset.pkl --index_file ../../Summarization/indices.pkl --output_format ../../Summarization/output/sent_train_set_
```

The parameters indicate the following:
1. input\_data - Pickle file containing the document-summary files. Check that the format matches the following input scheme: 
```
data = pickle.load(open(args.input_data, "rb"))
for i in range(len(data)):
    article, abstract = data[i]
```
2. index\_file - Since the process takes a while, we split the same into different parts by means of a pickled index file. The index file consists of a list of tuples of (start index, end index) within the indexing of the length of the dataset file. So if the dataset is of length 100, our index file might be [(0, 30), (30, 60), (60, 90), (90, 100)] and each execution of get\_matrices with a particular index creates the matrices associated with those datapoints in the dataset. The simplest way to handle this would be to make a list with just one tuple as [(0, length of dataset)].
3. index - Index within the index file to execute. Index file is a list, so if index is 0, then the matrices are created for the documents associated with the indices enclosed by the first tuple in the index file. So you can parallelize the process by running multiple at the same time. 
4. output\_format - File format/location where the output is stored. Output is stored in the form of a pickle file at output format location with the index appended at the end. 

#### Output Format
For the given execution line, the output would look like:
```
../../Summarization/output/sent_train_set_0.pkl
```
Where the location/directory was given by the output format and the 0 is given by the index executed by the current script (in practise you would have 0, 1, 2... all running in parallel)<br/>

The pickle file is indexed as 
```
data = pickle.load(open("../../Summarization/output/sent_train_set_0.pkl", "rb"))
doc_number = 0
print(data[doc_number]['vanilla'])
```
Within the pickle file, the primary index is the index of the document within the dataset pickle file. We create the 'vanilla' matrix which is an nxn sentence-wise PMI matrix as described in the paper. Additionally we also create the unused 'surprise' list which is the sentence-wise surprisal for each sentence and the 'normalised' PMI matrix.

