from argparse import ArgumentParser
import torch
from word_embeder import EmbeddingGenerator
from train import run
import pandas as pd

def build_parser():
    parser = ArgumentParser()

    ##Common option
    parser.add_argument("--device", dest="device", default="gpu")

    ##Loader option
    parser.add_argument("--train_path", dest="train_path", default="yahoo/train.csv")
    parser.add_argument("--valid_path", dest="valid_path", default="yahoo/test.csv")
    parser.add_argument("--save_path", dest="save_path", default=None)
    parser.add_argument("--max_sent_len", dest="max_sent_len", default=10, type=int)
    parser.add_argument("--max_word_len", dest="max_word_len", default=256, type=int)
    parser.add_argument("--tokenizer_name", dest="tokenizer_name", default="word_tokenizer",
                        help="Choose gensim, word_tokenizer")

    ##Model option
    parser.add_argument("--running_size", dest="running_size", default=32, type=int)
    parser.add_argument("--hidden_size", dest="hidden_size", default=512, type=int)
    parser.add_argument("--n_layers", dest="n_layers", default=1, type=int)

    ##Train option
    parser.add_argument("--n_epochs", dest="n_epochs", default=15, type=int)
    parser.add_argument("--early_stop", dest="early_stop", default=2, type=int)
    parser.add_argument("--batch_size", dest="batch_size", default=16, type=int)
    
    ##Embedding option
    parser.add_argument("--dict_path", dest="dict_path", default="word2vec")
    parser.add_argument("--size", dest="size", type=int, default=200)
    parser.add_argument("--alpha", dest="alpha", type=float, default=0.025)
    parser.add_argument("--window", dest="window", type=int, default=5)
    parser.add_argument("--min_count", dest="min_count", type=int, default=0)
    parser.add_argument("--sg", dest="sg", type=int, default=0)
    parser.add_argument("--negative", dest="negative", type=int, default=5)
    parser.add_argument("--num_class", dest="num_class", default=1, type=int)
    
    config = parser.parse_args()
    return config

def main(config):
    print("Experiment")
    size_list = [200,400,600]
    lr_list = [0.0001]
    hidden_list = [128, 256, 512]

    """
    dict_list = []
    ## Generating Embedding
    for size in size_list:
        print("Generating Embedding -- size:", str(size))
        config.size = size
        generator = EmbeddingGenerator(config.train_path, config.dict_path, config.tokenizer_name, config)
        dict_path = generator.generate()
        dict_list.append(dict_path)
    """

    dict_list = [
        "word2vec/1",
        "word2vec/2",
        "word2vec/3"
    ]


    history = {
        "train_loss": [],
        "valid_loss": [],
        "valid_correct": [],
        "epoch" : [],
        "lr" : [],
        "word2vec" : [],
        "hidden_size" : []
    }

    ## Testing multiple options
    for dict_path in dict_list:
        config.dict_path = dict_path
        for lr in lr_list:
            config.lr = lr
            for hidden_size in hidden_list:
                config.save_path = None
                config.hidden_size = hidden_size
                temp_history = run(config)
                epoch_size = len(temp_history["train_loss"])
                temp_epoch = [epoch for epoch in range(epoch_size)]
                temp_lr = [lr for epoch in range(epoch_size)]
                temp_word2vec = [dict_path for epoch in range(epoch_size)]
                temp_hidden_size = [hidden_size for epoch in range(epoch_size)]

                history["train_loss"].extend(temp_history["train_loss"])
                history["valid_loss"].extend(temp_history["valid_loss"])
                history["valid_correct"].extend(temp_history["valid_correct"])
                history["epoch"].extend(temp_epoch)
                history["lr"].extend(temp_lr)
                history["word2vec"].extend(temp_word2vec)
                history["hidden_size"].extend(temp_hidden_size)

    df = pd.DataFrame(history)
    df.to_csv("result.csv", encoding="UTF8", index=False)

if __name__ == "__main__":
    ##load config files
    config = build_parser()
    config.device = torch.device(
        "cuda" if torch.cuda.is_available() and (config.device == 'gpu' or config.device == 'cuda') else "cpu")
    main(config)