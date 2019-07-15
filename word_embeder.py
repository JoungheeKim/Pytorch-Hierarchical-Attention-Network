import csv
import gensim
from gensim.models import Word2Vec
import os
from argparse import ArgumentParser
import json
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

class Word2VecCorpus:
    def __init__(self, data_path, tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer
    def __iter__(self):
        with open(self.data_path, encoding="UTF8") as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                for sent in sent_tokenize(line[1]):
                    yield self.tokenizer.tokenize(sent)

class MyTokenizer():
    def __init__(self, tokenizer_name):
        self.tokenizer_name = tokenizer_name
        self.lemma = nltk.wordnet.WordNetLemmatizer()

    def tokenize(self, sent):
        if self.tokenizer_name == "gensim":
            return gensim.utils.simple_preprocess(sent)
        else:
            tokens = word_tokenize(sent)
            return [self.lemma.lemmatize(token) for token in tokens]


class EmbeddingGenerator():
    def __init__(self, data_path, save_path, tokenizer_name, config):
        self.data_path = data_path
        self.save_path = save_path
        self.tokenizer_name = tokenizer_name
        self.config = config
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        max_dir_num = 0
        all_subdir = [int(s) for s in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, str(s)))]
        if all_subdir:
            max_dir_num = max(all_subdir)
        max_dir_num += 1
        self.model_path = os.path.join(save_path, str(max_dir_num))
        os.mkdir(self.model_path)
        self.model_name = "word2vec.model"
        self.config_name = "config.json"

    def generate(self):
        word2vec_corpus = Word2VecCorpus(self.data_path, MyTokenizer(self.tokenizer_name))
        word2vec_model = Word2Vec(
            word2vec_corpus,
            size=self.config.size,
            alpha=self.config.alpha,
            window=self.config.window,
            min_count=self.config.min_count,
            sg=self.config.sg,
            negative=self.config.negative)

        word2vec_model.vocabulary

        word2vec_model.save(os.path.join(self.model_path, self.model_name))
        device = self.config.device
        self.config.device = "gpu"
        with open(os.path.join(self.model_path, self.config_name), 'w') as outfile:
            json.dump(vars(self.config), outfile)
        self.config.device = device

        return self.model_path

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--train_path", dest="train_path", default="data/train.csv")
    parser.add_argument("--dict_path", dest="dict_path", default="word2vec")

    parser.add_argument("--tokenizer_name", dest="tokenizer_name", default="word_tokenizer", help="Choose gensim, word_tokenizer")

    parser.add_argument("--size",dest="size", type=int, default=200)
    parser.add_argument("--alpha", dest="alpha", type=float, default=0.025)
    parser.add_argument("--window", dest="window", type=int, default=5)
    parser.add_argument("--min_count", dest="min_count", type=int, default=0)
    parser.add_argument("--sg", dest="sg", type=int, default=0)
    parser.add_argument("--negative", dest="negative", type=int, default=5)
    config = parser.parse_args()
    return config

if __name__ == "__main__":
    config = build_parser()
    generator = EmbeddingGenerator(config.train_path, config.dict_path, config.tokenizer_name, config)
    generator.generate()
