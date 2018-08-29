# similarity
Bilingual sentence similarity classifier based on optimising word alignments using Tensorflow.

This repo implements a sentence similarity classifier model using Tensorflow. Similarity classification is based on the ideas introduced by [Carpuat et al., 2017](http://aclweb.org/anthology/W17-3209) and similar to [Vyas et al., 2018](http://aclweb.org/anthology/N18-1136), [Schwenk, 2018](http://aclweb.org/anthology/P18-2037) and [Grégoire et al., 2018](http://www.aclweb.org/anthology/C18-1122). The code borrows many of the concepts and architecture presented in [Legrand et al., 2016](http://www.aclweb.org/anthology/W16-2207). 

Details on the implementation and experiments are published in:
* MinhQuang Pham, Josep Crego, Jean Senellart and François Yvon. [Fixing Translation Divergences in Parallel Corpora for Neural MT](). In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, EMNLP 2018. October 31–November 4. Brussels, Belgium.

The next picture, shows an example of similarity classification for the sentence pair:

``` What do you feel ? Not . ||| Que ressentez-vous ?```

<img src="https://github.com/jmcrego/similarity/blob/master/divergence_example.png" width="200" />

As it can be seen, the model outputs:
* a matrix with alignment scores,
* word aggregation scores (shown next to each source/target word) and
* an overall sentence similarity score (+0.1201).

In the previous paper we show that divergent sentences can be filtered out (using the overal similarity score) and that some divergences can be fixed (following alignment scores), guiding in both cases to outperform accuracy when compared to using the original corpora to learn a neural MT system. For our experiments we used the English-French [OpenSubtitles](http://www.lrec-conf.org/proceedings/lrec2016/pdf/947_Paper.pdf) and the English-German [Paracrawl](http://paracrawl.eu/) corpora.

# Preprocess

In order to learn our similarity model we better preprocess our training data with any tokenisation toolkit, basically aiming at reducing the vocabulary size. Any subtokenisation toolkit (such as BPE) can also be used. In our experiments we used the default tokenisation scheme implemented in [OpenNMT](http://opennmt.net) performing minimal tokenisation without subtokenisation.

## Vocabularies

After tokenisation, the most frequent |Vs| source and |Vt| target words are considered to be part of the source and target vocabularies respectively. The remaining will be mapped to a special UNK token. In our experiments we used |Vs| = |Vt| = 50,000 words.

## Pre-trained word embeddings

Any initialisation of source and target word embeddings can be used. In our experiments we initialised both source and target embeddings using [fastText](https://github.com/facebookresearch/fastText) with |Es| = |Et| = 256 cells. Embeddings were further refined using [MUSE](https://github.com/facebookresearch/MUSE). Note that word embeddings are not needed to learn the similarity model.

## Word alignments and Part-of-Speeches

To generate some training examples we will need to perform word alignments and POS-tagging of the source sentences. In our experiments we used [fast\_align](https://github.com/clab/fast_align) and [Freeling](https://github.com/TALP-UPC/FreeLing.git) to perform word alignment and English POS tagging respectively. Note that neither word alignments nor POS tags are needed to learn the similarity model.

Once the training parallel corpora is preprocessed we are ready to prepare our training examples:

```
python ./build_data.py -data FILE \
                       -mode purid \
                       -replace FILE
```

# Learning
```
python ./divergence_tagger.py -mdir DIR \
                              -dev FILE \
                              -trn FILE \
                              -wrd_dict FILE \
                              -tag_dict FILE \
                              -emb_size 100 \
                              -seq_size 100 \
                              -lstm_size 64 \
                              -batch_size 4 \
                              -lr_method sgd \
                              -lr 1.0 \
                              -lr_decay 0.8 \
                              -dropout 0.3 \
                              -n_epochs 15
```
# Inference
```
python ./divergence_tagger.py -model FILE \
                              -tst FILE \
                              -wrd_dict FILE \
                              -tag_dict FILE \
                              -emb_size 100 \
                              -seq_size 100 \
                              -lstm_size 64 \
                              -evaluate
```

## Visualize

# Fixing sentence pairs

