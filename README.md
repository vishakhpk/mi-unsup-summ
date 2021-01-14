# Unsupervised Extractive Summarization using Mutual Information

## Steps to perform summarization 

1. Fine tune the required language model
2. Create sentence-wise mutual information matrices for documents in the dataset
3. Use the created matrices to perform extractive summarization
4. Evaluation

### Fine tune the required language model

We use GPT2-large as provided by [HF transformers](https://github.com/huggingface/transformers) with a standard fine tuning script as provided in the model directory.


