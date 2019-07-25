# Pytorch-Hierarchical-Attention-Network
This is pytorch implementation of [Hierarchial Attention Network (HAN)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)


### Dataset
To test model, I use a dataset of 50,000 movie reviews taken from IMDb. 
It is divied into 'train', 'test' dataset and each data has 25,000 movie reviews and labels(positive, negetive).
You can access to dataset with this [link](http://ai.stanford.edu/~amaas/data/sentiment/)


### How to use it?
Follow the example

#### 1 Generate Word2Vec Embeddings
I implement "gensim" library to generate Word2vec embeddings. To generate word2vec embeddings, follow the sample

```python
python word_embeder.py --train_path source/train.csv --dict_path word2vec --tokenizer_name word_tokenizer --size 200 --window 5 --min_count 3
```

#### 2 Train Model
There is a lot of options to check.
1. train_path : A File to train model
2. valid_path : A File to valid model
3. dict_path : A Path of Word2vec model for embeddings of HAN model
4. save_path : A Path to save result of HAN model
5. max_sent_len : Maximum length of sentence to analysis ( Sentences of each document which is exceed to the limit is eliminated to train model )
6. max_word_len : Maximum length of word to analysis ( Words of each sentence which is exceed to the limit is eliminated to train model )
7. tokenizer_name : There is two tokenizing options ( "word_tokenizer", "gensim" )
8. atten_size : A Attention size of model
9. hidden_size : A hidden size of GRU model
10. n_layers : A number of layers of GRU model
11. n_epochs : A number of epoches to train
12. lr : learning rate
13. early_stop : A early_stop condition. If you don't want to use this options, put -1
14. batch_size : Batch size to train

```python
python train.py --train_path source/train.csv --valid_path source/test.csv --dict_path word2vec/1 --tokenizer_name word_tokenizer --hidden_size 256 --atten_size 128 --min_count 3 --batch_size 16
```

#### 3 Visualize Model
Check the code["visualize_tutorial.ipynb"](https://github.com/JoungheeKim/Pytorch-Hierarchical-Attention-Network/blob/master/visualize_tutorial.ipynb)


![](img/visualized_sentiment_document.PNG)


### Result
Result with hyper parameter settings

| word2vec dimention | hidden size | Best Epoch |  lr  | train loss | valid loss | valid correction |
|--------------------|:-----------:|:----------:|:----:|:----------:|:----------:|:----------------:|
| 200                |    128      |      3     |0.0001|   0.0159   |   0.0174   |      0.88168     |
| 200                |    256      |      3     |0.0001|   0.0150   |   0.0180   |      0.88004     |
| 200                |    512      |      3     |0.0001|   0.0147   |   0.0168   |      0.88956     |
| 400                |    128      |      2     |0.0001|   0.0178   |   0.0174   |      0.88240     |
| 400                |    256      |      2     |0.0001|   0.0173   |   0.0168   |      0.88668     |
| 400                |    512      |      2     |0.0001|   0.0170   |   0.0170   |      0.88724     |
| 600                |    128      |      2     |0.0001|   0.0162   |   0.0168   |      0.88784     |
| 600                |    256      |      2     |0.0001|   0.0163   |   0.0168   |      0.88548     |
| 600                |    512      |      2     |0.0001|   0.0164   |   0.0171   |      0.88348     |

### Reference

My pytorch implementation is highly impressed by other works. Please check below to see other works.
1. https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification
2. https://github.com/sharkmir1/Hierarchical-Attention-Network


