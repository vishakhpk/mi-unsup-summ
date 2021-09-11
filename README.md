# Unsupervised Extractive Summarization using Mutual Information

## Steps to perform summarization 

1. Fine tune the required language model
2. Create sentence-wise mutual information matrices for documents in the dataset
3. Use the created matrices to generate extractive summaries
4. Evaluation

### 1. Fine tune the required language model

We use GPT2-large as provided by [HF transformers](https://github.com/huggingface/transformers) with a standard fine tuning script as provided in the model directory.

### 2. Creating mutual information matrices 

For each document in the dataset, if there are n sentences, we need to create the nxn pairwise matrix as described in the paper. <br/>

```
python get_matrices.py --index 0 --input_data ./path/to/dataset.pkl --index_file ./path/to/indices.pkl --output_format ./path/to/output/sent_train_set_
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
./path/to/output/sent_train_set_0.pkl
```
Where the location/directory was given by the output format parameter in step 1 and the 0 is given by the index executed by the current script (in practise you would have 0, 1, 2... all running in parallel)<br/>

The pickle file is indexed as 
```
data = pickle.load(open("./path/to/output/sent_train_set_0.pkl", "rb"))
doc_number = 0
print(data[doc_number]['vanilla'])
```
Within the pickle file, the primary index is the index of the document within the dataset pickle file. We create the 'vanilla' matrix which is an nxn sentence-wise PMI matrix as described in the paper. Additionally we also create the unused 'surprise' list which is the sentence-wise surprisal for each sentence and the 'normalised' PMI matrix.

### 3. Generate Summaries
From the generated pmi matrices, we generate summaries using the algorithm detailed in the paper. 
```
python get_interpolated_output.py --output_file ./path/to/output/interpolated_ --dataset_file ./path/to/dataset.pkl --matrix_file_pattern ./path/to/output/sent_train_set_ 
```
Consumes the original dataset pickled file and the generated output files from the previous step. The script iterates over all the different output files created (hence why we aceept it as a pattern), and generates extractive summaries from length 1 to length 9 in 9 separate files using the format specified here. So here interpolated\_1 will consist of a text file where each line corresponds to a summary of length 1 sentence for each sentence in the dataset. And there will be 9 files like this. Additionally we also save all the gold summaries in a similar file. The reason for this file format is that it is compatible with the Rouge package in the next step. Additionally change the interpolation coefficients here if needed. To interpolate between relevance and redundancy, in the paper we used a classifier to learn weights assigned to each. A simpler alternative to just run inference is to set the weights to +1 for relevance and -1 for redundancy [here](https://github.com/vishakhpk/mi-unsup-summ/blob/196d3b646460f03cfcf9e41e1db621868a7156d0/get_interpolated_output.py#L54) and comment out the [line](https://github.com/vishakhpk/mi-unsup-summ/blob/196d3b646460f03cfcf9e41e1db621868a7156d0/get_interpolated_output.py#L17) loading the classifier. 

### 4. Evaluation 
Standard rouge evaluation using [rouge scorer](https://github.com/google-research/google-research/tree/master/rouge)
```
python3 -m rouge_score.rouge --target_filepattern=./path/to/output/gold --prediction_filepattern=./path/to/output/interpolated_3 --output_filename=./path/to/output/results.csv --use_stemmer=true
```

## Data for the paper
The following contains the preprocessed datasets, created PMI matrices, generated summaries and Rouge score reports used in our paper: <br>

[Google Drive Link](https://drive.google.com/drive/folders/1dBPd7trOOdKTNFDtUSGH9Z3zZ2PucDmL?usp=sharing) <br>

Please reach out if you would like to use our saved language models (vishakh@nyu.edu). We use GPT2 large, fine tuned on the document sentences from the various domains, using the script in the models directory. The input files for the fine tuning script are available in folder [LM-Files](https://drive.google.com/drive/folders/1XrlzvJqmvcK0IpYK-VwIN5tk2y6iIILi?usp=sharing) at the above drive location.

## TODOs
1. Convert scripts to a reusable utility
2. Maybe remove indices files to make things easier
