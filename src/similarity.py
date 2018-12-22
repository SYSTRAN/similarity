# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import sys
from dataset import Dataset, Vocab
from model import Model
from config import Config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def main():
    config = Config(sys.argv)
    model = Model(config)
    model.build_graph()
    model.initialize_session()

    if config.trn:
        trn = Dataset(config.trn, config.voc_src, config.tok_src, config.voc_tgt, config.tok_tgt,
                      config.seq_size, config.max_sents, do_shuffle=True)
        dev = Dataset(config.dev, config.voc_src, config.tok_src, config.voc_tgt, config.tok_tgt,
                      seq_size=0, max_sents=0, do_shuffle=False)
        model.learn(trn, dev, config.n_epochs)
    if config.tst:
        tst = Dataset(config.tst, config.voc_src, config.tok_src, config.voc_tgt, config.tok_tgt,
                      seq_size=0, max_sents=0, do_shuffle=False)
        model.inference(tst, quiet=config.quiet)

    model.close_session()


if __name__ == "__main__":
    main()
