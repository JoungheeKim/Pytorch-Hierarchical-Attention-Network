import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

def hierIterator(tensor, tensor_len, batch_size):
    count = 0
    max_len = len(tensor_len)
    while True:
        if count + batch_size >= max_len:
            yield tensor[count:], tensor_len[count:]
            break

        yield tensor[count: count+batch_size], tensor_len[count: count+batch_size]
        count = count+batch_size


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_src, query, mask=None):
        # |h_src| = (batch_size, length, hidden_size)
        # |query| = (hidden_size, 1)
        # |mask| = (batch_size, length)

        batch_size = h_src.size(0)
        hidden_size = query.size(0)

        weight = torch.mm(h_src.view(-1, hidden_size), query).view(batch_size, -1)
        # |weight| = (batch_size, length)

        if mask is not None:
            # Set each weight as -inf, if the mask value equals to 1.
            # Since the softmax operation makes -inf to 0, masked weights would be set to 0 after softmax operation.
            # Thus, if the sample is shorter than other samples in mini-batch, the weight for empty time-step would be set to 0.
            weight.masked_fill_(mask, -float('inf'))
        weight = self.softmax(weight)

        context_vector = torch.bmm(weight.unsqueeze(1), h_src)
        # |context_vector| = (batch_size, 1, hidden_size)

        return context_vector, weight


class HierAttModel(nn.Module):

    def __init__(self,
                 input_size,
                 word_vec_dim,
                 hidden_size,
                 num_class,
                 n_layers=1,
                 dropout_p=0,
                 running_size=32,
                 device=torch.device("cpu")
                 ):
        super(HierAttModel, self).__init__()

        self.device = device
        self.word_model = AttModel(input_size,
                                   word_vec_dim,
                                   hidden_size,
                                   n_layers=n_layers,
                                   dropout_p=dropout_p,
                                   use_embed=True,
                                   device=device
                                   ).to(device)
        self.sent_model = AttModel(hidden_size,
                                   word_vec_dim,
                                   hidden_size,
                                   n_layers=n_layers,
                                   dropout_p=dropout_p,
                                   use_embed=False,
                                   device=device
                                   ).to(device)
        self.running_size = running_size
        self.output = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, ids, sent_len, word_len):
        batch_size, max_sent_len, max_word_len = ids.size()
        # |ids| = (batch_size, max_sent_len, max_word_len)

        ids = ids.view(-1, max_word_len)
        word_len = word_len.view(-1)
        # |ids| = (total_len, max_word_len)
        # |word_len| = (total_len)

        # Pick the indices having information
        selected_index = []
        for sent_idx, length in enumerate(sent_len):
            selected_index += [sent_idx * max_sent_len + idx for idx in range(length)]

        ids = ids[selected_index].view(-1, max_word_len)
        word_len = word_len[selected_index].view(-1)
        # |ids| = (total_len, max_word_len)
        # |word_len| = (total_len)

        sent_vecs = []
        word_weights = []
        for temp_ids, temp_word_len in hierIterator(ids, word_len, self.running_size):
            sent_vec, word_weight = self.word_model(temp_ids, temp_word_len)
            sent_vecs += [sent_vec]
            word_weights += [word_weight]
        sent_vecs = torch.cat(sent_vecs, dim=0)
        word_weights = torch.cat(word_weights, dim=0)

        hidden_size = sent_vecs.size(1)
        temp_vec = []
        max_length = max(sent_len)

        for l in sent_len:
            if max_length - l > 0:
                temp_vec += [torch.cat([sent_vecs[:l], torch.zeros(((max_length - l), hidden_size), device=self.device)]).unsqueeze(0)]
            else:
                temp_vec += [sent_vecs[:l].unsqueeze(0)]
            sent_vecs = sent_vecs[l:]
        sent_vecs = torch.cat(temp_vec, dim=0)

        doc_vecs = []
        sent_weights = []
        for temp_vec, temp_sent_len in hierIterator(sent_vecs, sent_len, self.running_size):
            doc_vec, sent_weight = self.sent_model(temp_vec, temp_sent_len)
            doc_vecs += [doc_vec]
            sent_weights += [sent_weight]
        doc_vecs = torch.cat(doc_vecs, dim=0)
        sent_weights = torch.cat(sent_weights, dim=0)

        y = self.softmax(self.output(doc_vecs))

        return y, word_weights, sent_weights

    def set_embedding(self, embedding):
        self.word_model.emb_src.weight.data.copy_(embedding)
        return True


class AttModel(nn.Module):
    def __init__(self,
                 input_size,
                 word_vec_dim,
                 hidden_size,
                 n_layers=1,
                 dropout_p=0,
                 use_embed=False,
                 device=torch.device("cpu")
                 ):
        super(AttModel, self).__init__()

        self.device=device
        self.use_embed = use_embed
        vec_dim = input_size
        if use_embed:
            self.emb_src = nn.Embedding(input_size, word_vec_dim).to(device)
            vec_dim = word_vec_dim
        self.rnn = nn.GRU(vec_dim,
                          int(hidden_size / 2),
                          num_layers=n_layers,
                          dropout=dropout_p,
                          bidirectional=True,
                          batch_first=True
                          ).to(device)

        self.attn = Attention(hidden_size).to(device)

        self.context_weight = nn.Parameter(torch.Tensor(hidden_size, 1)).to(device)
        self.context_weight.data.normal_(mean=0.0, std=0.02)
        # |context_weight| = (hidden_size, 1)

    def encode(self, emb):
        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True, enforce_sorted=False)
            # Below is how pack_padded_sequence works.
            # As you can see, PackedSequence object has information about mini-batch-wise information, not time-step-wise information.
            #
            # a = [torch.tensor([1,2,3]), torch.tensor([3,4])]
            # b = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
            # >>>>
            # tensor([[ 1,  2,  3],
            #     [ 3,  4,  0]])
            # torch.nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=[3,2]
            # >>>>PackedSequence(data=tensor([ 1,  3,  2,  4,  3]), batch_sizes=tensor([ 2,  2,  1]))
        else:
            x = emb

        y, h = self.rnn(x)
        # |y| = (batch_size, length, hidden_size)
        # |h[0]| = (num_layers * 2, batch_size, hidden_size / 2)

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)

        return y, h

    def forward(self, x, x_length):
        # |x| = (batch_size, length)
        # |x_length| = (batch_size)
        # 'length equals torch.max(x_length)
        batch_size= x.size(0)
        length = x.size(1)

        # Based on the length information, gererate mask to prevent that shorter sample has wasted attention.
        mask = self.generate_mask(x_length)
        # |mask| = (batch_size, length)

        if self.use_embed:
            # Get word embedding vectors for every time-step of input sentence.
            emb_src = self.emb_src(x)
            # |emb_src| = (batch_size, length, word_vec_dim)
        else:
            emb_src = x

        # The last hidden state of the encoder would be a initial hidden state of decoder.
        h_src, h_0_tgt = self.encode((emb_src, x_length))
        # |h_src| = (batch_size, 'length, hidden_size)   - Notice that 'length equals torch.max(x_length)
        # |h_0_tgt| = (n_layers * 2, batch_size, hidden_size / 2)

        context_vector, context_weight = self.attn(h_src, self.context_weight, mask)
        # |context_vector| = (batch_size, 1, hidden_size)
        # |context_weight| = (batch_size, 'length) - Notice that 'length equals torch.max(x_length)

        context_vector = context_vector.view(batch_size, -1)
        # |context_vector| = (batch_size, hidden_size)

        context_weight = torch.cat([context_weight, torch.zeros(batch_size, length - context_weight.size(1), device=self.device)], dim=1)

        return context_vector, context_weight

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