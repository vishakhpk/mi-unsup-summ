# Model Directory

We use GPT2-large as provided by [HF transformers](https://github.com/huggingface/transformers)..

## Fine tuning GPT2

run\_language\_modeling.py is a standard fine tuning python script. <br/>
You only need to modify the train/test files within finetune\_large.sh and execute the same.

```
sh finetune_large.sh
```

## Using the model

gpt2.py is a utility used by other elements of our project. As a standalone, it loads the checkpoint saved by the fine tuning script and has a function to obtain the probability of each word in a paragraph/sentence of text. <br/>
