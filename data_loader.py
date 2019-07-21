from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import gensim
from gensim.models import Word2Vec
import numpy as np
from sklearn import preprocessing
from torch.utils.data import DataLoader
import logging
from word_embeder import MyTokenizer
import tools
import os


def get_data(data_path):
    docs, labels = [], []
    with open(data_path, encoding="UTF8") as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            label = int(line[1])
            docs.append(line[0])
            labels.append(label)
    num_class = np.unique(labels)
    return docs, labels, num_class

class MyGensimModel():
    def __init__(self,
                 model_path):
        model = Word2Vec.load(model_path)
        embedding = model.wv.vectors
        dict_size = len(embedding)
        index2word = model.wv.index2word
        word_vec_dim = model.vector_size
        # Insert Unknown Token
        unknown_word = preprocessing.normalize(np.random.rand(1, word_vec_dim))
        embedding = torch.from_numpy(np.concatenate([unknown_word, embedding], axis=0).astype(np.float))
        index2word = ['[UNK]'] + index2word
        dict_size += 1
        word2index = {text: index for index, text in enumerate(index2word)}

        self.model = model
        self.embedding = embedding
        self.dict_size = dict_size
        self.index2word = index2word
        self.word2index = word2index
        self.word_vec_dim = word_vec_dim

class MyDataLoader():
    def __init__(self,
                 train_path,
                 valid_path,
                 dict_path = None,
                 batch_size = 32,
                 tokenizer_name="word_tokenizer",
                 max_sent_len=100,
                 max_word_len=100):

        ##Load vector from Word2vec

        model_path = os.path.join(dict_path, tools.WORD2VEC_NAME)

        ## Update gensim infomation
        self.model = MyGensimModel(model_path)

        self.tokenizer = MyTokenizer(tokenizer_name)

        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len

        self.train_path = train_path
        self.valid_path = valid_path
        self.dict_path = dict_path
        self.batch_size = batch_size

        self.data_sent_len = 0
        self.data_word_len = 0

    def get_dict_size(self):
        return self.model.dict_size

    def get_dict_vec_dim(self):
        return self.model.word_vec_dim

    def get_embedding(self):
        return self.model.embedding

    def get_dataset(self, docs, labels):
        docs_ids = []
        docs_sent_len = []
        docs_word_len = []

        for doc in docs:
            temp_index = [[self.model.word2index.get(word) if self.model.word2index.get(word) else 0 for word in
                           self.tokenizer.tokenize(sentences)] for sentences in sent_tokenize(doc)]

            for sentence in temp_index:
                ##Even though there is no word after preprocess procedure, must put something like "[UNK]" to run machine
                if len(sentence) == 0:
                    sentence.extend([0])

            temp_index = [sentences[:self.max_word_len] for sentences in temp_index][:self.max_sent_len]

            sent_len = len(temp_index)
            word_len = [len(sent) for sent in temp_index]

            #Update maximum word, sent Length of Documents
            if sent_len > self.data_sent_len:
                self.data_sent_len = sent_len
            for temp_len in word_len:
                if temp_len > self.data_word_len:
                    self.data_word_len = temp_len

            docs_ids.append(temp_index)
            docs_sent_len.append(sent_len)
            docs_word_len.append(word_len)

        return DocumentDataset(docs_ids, docs_sent_len, docs_word_len, labels, max_word_len=self.max_word_len, max_sent_len=self.max_sent_len)

    def get_train_valid(self):
        docs, labels, train_class = get_data(self.train_path)
        train = self.get_dataset(docs, labels)

        docs, labels, valid_class = get_data(self.valid_path)
        valid = self.get_dataset(docs, labels)

        """
        if self.compac_max_length():
            train.set_max_len(max_sent_len=self.data_sent_len, max_word_len=self.data_word_len)
            valid.set_max_len(max_sent_len=self.data_sent_len, max_word_len=self.data_word_len)
        """

        ##Put DataLoader
        train = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        valid = DataLoader(valid, batch_size=self.batch_size, shuffle=False)
        num_class = len(set(valid_class + train_class))

        return train, valid, num_class

    def compac_max_length(self):
        changed = False
        if self.max_word_len >= self.data_word_len:
            self.max_word_len = self.data_word_len
            changed = True
        if self.max_sent_len >= self.data_sent_len:
            self.max_sent_len = self.data_sent_len
            changed = True
        return changed


class DocumentDataset(Dataset):

    def __init__(self, ids, sent_len, word_len, labels, max_word_len=10, max_sent_len=10):
        super(DocumentDataset, self).__init__()

        self.ids = ids
        self.sent_len = sent_len
        self.word_len = word_len
        self.labels = labels
        self.len = len(labels)
        self.max_word_len=max_word_len
        self.max_sent_len=max_sent_len

    def __len__(self):
        return self.len

    def set_max_len(self, max_word_len, max_sent_len):
        self.max_word_len = max_word_len
        self.max_sent_len = max_sent_len
        return True

    def __getitem__(self, index):
        temp_index = self.ids[index]
        temp_sent_len = self.sent_len[index]
        temp_word_len = self.word_len[index]
        temp_labels = self.labels[index]

        for sent in temp_index:
            if len(sent) < self.max_word_len:
                extended_words = [0 for _ in range(self.max_word_len - len(sent))]
                sent.extend(extended_words)

        if len(temp_index) < self.max_sent_len:
            extended_sentences = [[0 for _ in range(self.max_word_len)] for _ in
                                  range(self.max_sent_len - len(temp_index))]
            temp_index.extend(extended_sentences)

        temp_index = [sentences[:self.max_word_len] for sentences in temp_index][:self.max_sent_len]

        if len(temp_word_len) < self.max_sent_len:
            extended_word_len = [0 for _ in range(self.max_sent_len - len(temp_word_len))]
            temp_word_len.extend(extended_word_len)
        temp_word_len = temp_word_len[:self.max_sent_len]

        temp_index = torch.tensor(temp_index)
        temp_sent_len = torch.tensor(temp_sent_len)
        temp_word_len = torch.tensor(temp_word_len)
        temp_labels = torch.tensor(temp_labels)

        return temp_index, temp_sent_len, temp_word_len, temp_labels



if __name__ == '__main__':
    train_path = 'data/train.csv'
    valid_path = 'data/test.csv'
    dict_path = 'word2vec/3/word2vec.model'
    loader = MyDataLoader(train_path, valid_path, dict_path, max_word_len=1, max_sent_len=1)
    train, valid, size = loader.get_train_valid()


