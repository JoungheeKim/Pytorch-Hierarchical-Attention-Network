from model import HierarchialAttentionNetwork
import torch
from word_embeder import MyTokenizer
import json
from nltk.tokenize import sent_tokenize
from data_loader import MyGensimModel
from IPython.display import Markdown, display

class Classifier():
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
        self.model = HierarchialAttentionNetwork(dictionary_size=self.word2vec_model.dict_size,
                                                 embedding_size=word2vec_config.size,
                                                 hidden_size=HAN_config.hidden_size,
                                                 attention_size=HAN_config.atten_size,
                                                 num_class=HAN_config.num_class,
                                                 n_layers=HAN_config.n_layers,
                                                 device=device
                                                 )
        self.model.set_embedding(self.word2vec_model.embedding)
        check_point = torch.load(HAN_mdoel_path)
        self.model.load_state_dict(check_point["model"])
        self.model.to(device)

    def analysis(self, doc):
        # |doc| = (doc)

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
        y_hat, sent_weights, word_weights = self.model(temp_index, temp_sent_len, temp_word_len)
        ps = torch.exp(y_hat)
        top_p, top_class = ps.topk(1, dim=1)

        sent_weights = sent_weights.squeeze()
        word_weights = word_weights.squeeze()

        return top_class, tokens, sent_weights, word_weights

    def view(self, doc):
        top_class, tokens, sent_weights, word_weights = self.analysis(doc)
        sent_weights = sent_weights.tolist()
        word_weights = word_weights.tolist()
        total_len = len(sent_weights)

        for sent, word_weight, sent_weight in zip(tokens, word_weights, sent_weights):
            temp_str = self.mk_weight_string(sent, word_weight, sent_weight, total_len)
            self.printmd(temp_str)


    def mk_weight_string(self, str_list, w_list, s_weight, total_len):
        temp_str = []
        for string, weight in zip(str_list, w_list):
            temp_str += ['<span style="background-color:rgba(255,0,0,' + str(weight) + ');  font-size: ' + str(
                int(total_len) * 10 * s_weight) + 'pt;">' + string + '</span>']
        return " ".join(temp_str)

    # Markdown Printer
    def printmd(self, string):
        display(Markdown(string))




