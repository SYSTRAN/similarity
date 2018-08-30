# similarity
Bilingual sentence similarity classifier based on optimising word alignments using Tensorflow.

This repo implements a sentence similarity classifier model using Tensorflow. Similarity classification is based on the ideas introduced by [Carpuat et al., 2017](http://aclweb.org/anthology/W17-3209) and similar to [Vyas et al., 2018](http://aclweb.org/anthology/N18-1136), [Schwenk, 2018](http://aclweb.org/anthology/P18-2037) and [Grégoire et al., 2018](http://www.aclweb.org/anthology/C18-1122). The code borrows many of the concepts and architecture presented in [Legrand et al., 2016](http://www.aclweb.org/anthology/W16-2207). 

Details on the implementation and experiments are published in:
* MinhQuang Pham, Josep Crego, Jean Senellart and François Yvon. [Fixing Translation Divergences in Parallel Corpora for Neural MT](http://emnlp2018.org/program/accepted/short-papers). In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, EMNLP 2018. October 31–November 4. Brussels, Belgium.

The next picture, shows an example of similarity classification for the sentence pair:

``` What do you feel ? Not . ||| Que ressentez-vous ?```

<img src="https://github.com/jmcrego/similarity/blob/master/divergence_example.png" width="200" />

As it can be seen, the model outputs:
* a matrix with alignment scores,
* word aggregation scores (shown next to each source/target word) and
* an overall sentence pair similarity score (+0.1201).

In the previous paper we show that divergent sentences can be filtered out (using the sentence similarity score) and that some divergences can be fixed (following alignment scores), guiding in both cases to outperform accuracy when compared to neural MT systems using the original corpora. For our experiments we used the English-French [OpenSubtitles](http://www.lrec-conf.org/proceedings/lrec2016/pdf/947_Paper.pdf) and the English-German [Paracrawl](http://paracrawl.eu/) corpora.

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
                       -mode STRING \
                       -replace FILE
```
The input data file contains one sentence pair per line, with the next fields separated by TABs:
* source sentence
* target sentence
* source/target alignments
* source part-of-speeches

 :point_right: Why wait for the Euro ? \ \ \ \  **Pourquoi attendre l ' Euro ?** 0-0 1-1 2-1 3-2 4-3 4-4 5-5 **WRB NNP IN DT NNP SYM**

(The last two fields are optional)

Available modes:
* 'p': Parallel sentences
 :point_right: Why wait for the Euro ?   **Pourquoi attendre l ' Euro ?**   -1.0 -1.0 -1.0 -1.0 -1.0 -1.0   **-1.0 -1.0 -1.0 -1.0 -1.0 -1.0**

* 'u': uneven sentences
 :point_right: Why wait for the Euro ?   **Cela peut donc se produire .**   1.0 1.0 1.0 1.0 1.0 1.0   **1.0 1.0 1.0 1.0 1.0 1.0**

* 'i': insert sentence
 :point_right: Why wait for the Euro ?   **Pourquoi attendre l ' Euro ? Il existe un précédant .**   -1.0 -1.0 -1.0 -1.0 -1.0 -1.0   **-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 1.0 1.0 1.0 1.0 1.0**

* 'd': delete sequence
 :point_right: Why wait for the Euro ?   **l ' Euro ?**   1.0 1.0 1.0 -1.0 -1.0 -1.0   **-1.0 -1.0 -1.0 -1.0**

 (needs word alignments in input FILE)

* 'r': replace sequences with equivalent part-of-speech
 :point_right: Where wait for the Euro ?   **Pourquoi attendre l ' Euro ?**   1.0 -1.0 -1.0 -1.0 -1.0 -1.0   **1.0 -1.0 -1.0 -1.0 -1.0 -1.0**

 (needs word alignments and source POS-tags in -data FILE and equivalent sequences in -replace FILE)

# Learning
```
python ./divergence_tagger.py -mdir DIR -dev FILE -trn FILE -data_mode $DATAMODE -net_mode $NETMODE -max_sents 1000000 -src_voc $REMOTEDIR/$sdic -tgt_voc $REMOTEDIR/$tdic -src_emb $REMOTEDIR/$semb -tgt_emb $REMOTEDIR/$temb -batch_size $BATCH -n_epochs $NEPOCHS -seq_size $SEQ -lr_method $METHOD -lr $LR -lr_decay $DECAY -dropout $DROP -aggr $AGGR -src_lstm_size $LSTM -tgt_lstm_size $LSTM
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
python -u similarity.py -mdir DIR -epoch N -tst FILE
```

## Visualize

# Fixing sentence pairs

```
python -u ./fix.py -use_punct < FILE_WITH_ALIGNMENTS
```
