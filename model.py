import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence



class HierarchialAttentionNetwork(nn.Module):

    def __init__(self, dictionary_size, embedding_size, hidden_size, attention_size,
                 num_class, n_layers=1, dropout_p=0.05, device="cpu"):

        super(HierarchialAttentionNetwork, self).__init__()

        self.word_attention_model = WordAttention(dictionary_size=dictionary_size,
                                                  embedding_size=embedding_size,
                                                  hidden_size=hidden_size,
                                                  attention_size=attention_size,
                                                  n_layers=n_layers,
                                                  dropout_p=dropout_p,
                                                  device=device
                                                  )

        self.sentence_attention_model = SentenceAttention(input_size=hidden_size,
                                                          hidden_size=hidden_size,
                                                          attention_size=attention_size,
                                                          n_layers=n_layers,
                                                          dropout_p=dropout_p,
                                                          device=device
                                                          )

        self.device = device
        self.output = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, document, sentence_per_document, word_per_sentence):
        batch_size, max_sentence_length, max_word_length = document.size()
        # |document| = (batch_size, max_sentence_length, max_word_length)
        # |sentence_per_document| = (batch_size)
        # |word_per_sentence| = (batch_size, max_sentence_length)

        # Remove sentence-padding in document by using "pack_padded_sequence.data"
        packed_sentences = pack(document,
                                lengths=sentence_per_document.tolist(),
                                batch_first=True,
                                enforce_sorted=False)
        # |packed_sentences.data| = (sum(sentence_length), max_word_length)

        # Remove sentence-padding in word_per_sentence "pack_padded_sequence.data"
        packed_words_per_sentence = pack(word_per_sentence,
                                         lengths=sentence_per_document.tolist(),
                                         batch_first=True,
                                         enforce_sorted=False)
        # |packed_words_per_sentence.data| = (sum(sentence_length))

        # Get sentence vectors
        sentence_vecs, word_weights = self.word_attention_model(packed_sentences.data,
                                                                packed_words_per_sentence.data)
        # |sentence_vecs| = (sum(sentence_length), hidden_size)
        # |word_weights| = (sum(sentence_length, max(word_per_sentence))

        # "packed_sentences" have same information to recover PackedSequence for sentence
        packed_sentence_vecs = PackedSequence(data=sentence_vecs,
                                              batch_sizes=packed_sentences.batch_sizes,
                                              sorted_indices=packed_sentences.sorted_indices,
                                              unsorted_indices=packed_sentences.unsorted_indices)

        # Get document vectors
        doc_vecs, sentence_weights = self.sentence_attention_model(packed_sentence_vecs,
                                                                   sentence_per_document)
        # |doc_vecs| = (batch_size, hidden_size)
        # |sentence_weights| = (batch_size)

        y = self.softmax(self.output(doc_vecs))

        return y, sentence_weights, word_weights

    def set_embedding(self, embedding, requires_grad = True):
        self.word_attention_model.emb.weight.data.copy_(embedding)
        return True



class SentenceAttention(nn.Module):
    def __init__(self, input_size, hidden_size, attention_size, n_layers=1, dropout_p=0, device="cpu"):
        super(SentenceAttention, self).__init__()

        self.device=device
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=int(hidden_size / 2),
                          num_layers=n_layers,
                          dropout=dropout_p,
                          bidirectional=True,
                          batch_first=True
                          )

        self.attn = Attention(hidden_size=hidden_size,
                              attention_size=attention_size)


    def forward(self, packed_sentences, sentence_per_document):
        # |packed_sentences| = PackedSequence()

        # Apply RNN and get hiddens layers of each sentences
        last_hiddens, _ = self.rnn(packed_sentences)

        # Unpack ouput of rnn model
        last_hiddens, _ = unpack(last_hiddens, batch_first=True)
        # |last_hiddens| = (sentence_length, max(word_per_sentence), hidden_size)

        # Based on the length information, gererate mask to prevent that shorter sample has wasted attention.
        mask = self.generate_mask(sentence_per_document)
        # |mask| = (sentence_length, max(word_per_sentence))

        # Get attention weights and context vectors
        context_vectors, context_weights = self.attn(last_hiddens, mask)
        # |context_vectors| = (sentence_length, hidden_size)
        # |context_weights| = (sentence_length, max(word_per_sentence))

        return context_vectors, context_weights

    def generate_mask(self, length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples,
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat(
                    [torch.zeros((1, l), dtype=torch.uint8), torch.ones((1, (max_length - l)), dtype=torch.uint8)],
                    dim=-1)]
            else:
                # If the length of the sample equals to maximum length among samples,
                # set every value in mask to be 0.
                mask += [torch.zeros((1, l), dtype=torch.uint8)]

        mask = torch.cat(mask, dim=0).byte()

        return mask.to(self.device)

class WordAttention(nn.Module):
    def __init__(self, dictionary_size, embedding_size, hidden_size, attention_size, n_layers=1, dropout_p=0, device="cpu"):
        super(WordAttention, self).__init__()

        self.device=device
        self.emb = nn.Embedding(dictionary_size, embedding_size).to(device)
        self.rnn = nn.GRU(input_size=embedding_size,
                          hidden_size=int(hidden_size / 2),
                          num_layers=n_layers,
                          dropout=dropout_p,
                          bidirectional=True,
                          batch_first=True
                          )

        self.attn = Attention(hidden_size=hidden_size,
                              attention_size=attention_size)


    def forward(self, sentence, word_per_sentence):
        # |sentence| = (sentence_length, max_word_length)
        # |word_per_sentence| = (sentence_length)

        sentence = self.emb(sentence)
        # |sentence| = (sentence_length, max_word_length, embedding_size)

        # Pack sentence before insert rnn model.
        packed_sentences = pack(sentence,
                                lengths=word_per_sentence.tolist(),
                                batch_first=True,
                                enforce_sorted=False)

        # Apply RNN and get hiddens layers of each words
        last_hiddens, _ = self.rnn(packed_sentences)

        # Unpack ouput of rnn model
        last_hiddens, _ = unpack(last_hiddens, batch_first=True)
        # |last_hiddens| = (sentence_length, max(word_per_sentence), hidden_size)

        # Based on the length information, gererate mask to prevent that shorter sample has wasted attention.
        mask = self.generate_mask(word_per_sentence)
        # |mask| = (sentence_length, max(word_per_sentence))

        context_vectors, context_weights = self.attn(last_hiddens, mask)
        # |context_vectors| = (sentence_length, hidden_size)
        # |context_weights| = (sentence_length, max(word_per_sentence))

        return context_vectors, context_weights

    def generate_mask(self, length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples,
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat(
                    [torch.zeros((1, l), dtype=torch.uint8), torch.ones((1, (max_length - l)), dtype=torch.uint8)],
                    dim=-1)]
            else:
                # If the length of the sample equals to maximum length among samples,
                # set every value in mask to be 0.
                mask += [torch.zeros((1, l), dtype=torch.uint8)]

        mask = torch.cat(mask, dim=0).byte()

        return mask.to(self.device)

class Attention(nn.Module):

    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, attention_size, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        ## Context vector
        self.context_weight = nn.Parameter(torch.Tensor(attention_size, 1))
        self.context_weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, h_src, mask=None):
        # |h_src| = (batch_size, length, hidden_size)
        # |mask| = (batch_size, length)
        batch_size, length, hidden_size = h_src.size()

        # Resize hidden_vectors to generate weight
        weights = h_src.view(-1, hidden_size)
        weights = self.linear(weights)
        weights = self.tanh(weights)

        weights = torch.mm(weights, self.context_weight).view(batch_size, -1)
        # |weights| = (batch_size, length)

        if mask is not None:
            # Set each weight as -inf, if the mask value equals to 1.
            # Since the softmax operation makes -inf to 0, masked weights would be set to 0 after softmax operation.
            # Thus, if the sample is shorter than other samples in mini-batch, the weight for empty time-step would be set to 0.
            weights.masked_fill_(mask, -float('inf'))

        # Modified every values to (0~1) by using softmax function
        weights = self.softmax(weights)
        # |weights| = (batch_size, length)

        context_vectors = torch.bmm(weights.unsqueeze(1), h_src)
        # |context_vector| = (batch_size, 1, hidden_size)

        context_vectors = context_vectors.squeeze(1)
        # |context_vector| = (batch_size, hidden_size)

        return context_vectors, weights

