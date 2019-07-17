import numpy as np
import pandas as pd
import os
from model import HierAttModel
import gensim
from data_loader import MyTokenizer, MyGensimModel
import json
import torch
from nltk.tokenize import sent_tokenize

class TestModel():
    def __init__(self,
                 word2vec_config_path,
                 word2vec_model_path,
                 HAN_mdoel_path,
                 HAN_config_path,
                 tokenizer_name="word_tokenizer",
                 device=torch.device("cpu")):

        class Struct:
            def __init__(self, **entries):
                self.__dict__.update(entries)

        self.device = device

        ##Load word2vec config
        with open(word2vec_config_path, 'r') as f:
            word2vec_config = json.load(f)
        word2vec_config = Struct(**word2vec_config)

        self.word2vec_model = MyGensimModel(word2vec_model_path)

        ##Load tokenizer
        self.tokenizer = MyTokenizer(tokenizer_name)

        ##Load HAN config
        with open(HAN_config_path, 'r') as f:
            HAN_config = json.load(f)
        HAN_config = Struct(**HAN_config)

        ##Load HAN model
        self.model = HierAttModel(input_size=self.word2vec_model.dict_size,
                             word_vec_dim=self.word2vec_model.word_vec_dim,
                             hidden_size=HAN_config.hidden_size,
                             num_class=4,
                             running_size=HAN_config.running_size,
                             n_layers=HAN_config.n_layers,
                             device=device
                             ).to(device)
        self.model.set_embedding(self.word2vec_model.embedding)
        check_point = torch.load(HAN_mdoel_path)
        self.model.load_state_dict(check_point["model"])

    def analysis(self, doc):
        tokens = [[word for word in
                   self.tokenizer.tokenize(sentences, lemma=False)] for sentences in sent_tokenize(doc)]
        temp_index = [[self.word2vec_model.word2index.get(word) if self.word2vec_model.word2index.get(word) else 0 for word in
                       self.tokenizer.tokenize(sentences)] for sentences in sent_tokenize(doc)]
        for sentence in temp_index:
            ##Even though there is no word after preprocess procedure, must put something like "[UNK]" to run machine
            if len(sentence) == 0:
                sentence.extend([0])

        temp_sent_len = len(temp_index)
        temp_word_len = [len(sent) for sent in temp_index]

        max_sent_len = temp_sent_len
        max_word_len = max(temp_word_len)

        for sent in temp_index:
            if len(sent) < max_word_len:
                extended_words = [0 for _ in range(max_word_len - len(sent))]
                sent.extend(extended_words)

        if len(temp_index) < max_sent_len:
            extended_sentences = [[0 for _ in range(max_word_len)] for _ in
                                  range(max_sent_len - len(temp_index))]
            temp_index.extend(extended_sentences)

        temp_index = [sentences[:max_word_len] for sentences in temp_index][:max_sent_len]

        if len(temp_word_len) < max_sent_len:
            extended_word_len = [0 for _ in range(max_sent_len - len(temp_word_len))]
            temp_word_len.extend(extended_word_len)
        temp_word_len = temp_word_len[:max_sent_len]

        temp_index = torch.tensor(temp_index)
        temp_sent_len = torch.tensor(temp_sent_len)
        temp_word_len = torch.tensor(temp_word_len)

        temp_index = temp_index.unsqueeze(0).to(self.device)
        temp_sent_len = temp_sent_len.unsqueeze(0).to(self.device)
        temp_word_len = temp_word_len.unsqueeze(0).to(self.device)
        y_hat, weights, sent_weights = self.model(temp_index, temp_sent_len, temp_word_len)
        ps = torch.exp(y_hat)
        top_p, top_class = ps.topk(1, dim=1)
        weights = weights.tolist()

        return top_class, tokens, weights




