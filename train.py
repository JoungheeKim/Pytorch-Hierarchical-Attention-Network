import pandas as pd
import numpy
from argparse import ArgumentParser
from data_loader import MyDataLoader
from model import HierAttModel
from trainer import Trainer
from torch import nn
import torch
import logging
import tools
import os


def build_parser():
    parser = ArgumentParser()

    ##Common option
    parser.add_argument("--device", dest="device", default="gpu")

    ##Loader option
    parser.add_argument("--train_path", dest="train_path", default="source/train.csv")
    parser.add_argument("--valid_path", dest="valid_path", default="source/test.csv")
    parser.add_argument("--dict_path", dest="dict_path", default="word2vec/1")
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
    parser.add_argument("--lr", dest="lr", default=0.00001, type=int)
    parser.add_argument("--early_stop", dest="early_stop", default=1, type=int)
    parser.add_argument("--batch_size", dest="batch_size", default=16, type=int)

    config = parser.parse_args()
    return config


def run(config):
    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    _print_config(config)

    if not logging.getLogger() == None:
        for handler in logging.getLogger().handlers[:]:  # make a copy of the list
            logging.getLogger().removeHandler(handler)

    if not config.save_path and config.dict_path:
        all_subdir = [int(s) for s in os.listdir(config.dict_path) if os.path.isdir(os.path.join(config.dict_path, str(s)))]
        max_dir_num = 0
        if all_subdir:
            max_dir_num = max(all_subdir)
        max_dir_num += 1
        config.save_path = os.path.join(config.dict_path, str(max_dir_num))
        os.mkdir(config.save_path)

    logging.basicConfig(filename=os.path.join(config.save_path, 'train_log'),
                        level=tools.LOGFILE_LEVEL,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(tools.CONSOLE_LEVEL)
    logging.getLogger().addHandler(console)

    logging.info("##################### Start Training")
    logging.debug(vars(config))

    ##load data loader
    logging.info("##################### Load DataLoader")
    loader = MyDataLoader(train_path=config.train_path,
                          valid_path=config.valid_path,
                          dict_path=config.dict_path,
                          batch_size=config.batch_size,
                          tokenizer_name=config.tokenizer_name,
                          max_sent_len=config.max_sent_len,
                          max_word_len=config.max_word_len)

    train, valid, num_class = loader.get_train_valid()
    logging.info("##################### Train Dataset size : [" + str(len(train)) + "]")
    logging.info("##################### Valid Dataset size : [" + str(len(valid)) + "]")
    logging.info("##################### class size : [" + str(num_class) + "]")

    input_size = loader.get_dict_size()
    word_vec_dim = loader.get_dict_vec_dim()
    embedding = loader.get_embedding()

    logging.info("##################### Load 'HAN' Model")
    model = HierAttModel(input_size=input_size,
                         word_vec_dim=word_vec_dim,
                         hidden_size=config.hidden_size,
                         num_class=num_class,
                         running_size=config.running_size,
                         n_layers=config.n_layers,
                         device=config.device
                         ).to(config.device)
    model.set_embedding(embedding)

    crit = nn.NLLLoss()
    trainer = Trainer(model=model,
                      crit=crit,
                      config=config,
                      device=config.device)
    history = trainer.train(train, valid)
    return history

if __name__ == "__main__":
    ##load config files
    config = build_parser()
    config.device = torch.device(
        "cuda" if torch.cuda.is_available() and (config.device == 'gpu' or config.device == 'cuda') else "cpu")
    run(config)