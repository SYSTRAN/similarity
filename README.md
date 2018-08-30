# similarity
Bilingual sentence similarity classifier based on optimising word alignments using Tensorflow.

This repo implements a sentence similarity classifier model using Tensorflow. Similarity classification is based on the ideas introduced by [Carpuat et al., 2017](http://aclweb.org/anthology/W17-3209) and similar to [Vyas et al., 2018](http://aclweb.org/anthology/N18-1136), [Schwenk, 2018](http://aclweb.org/anthology/P18-2037) and [Grégoire et al., 2018](http://www.aclweb.org/anthology/C18-1122). The code borrows many of the concepts and architecture presented in [Legrand et al., 2016](http://www.aclweb.org/anthology/W16-2207). 

Details on the implementation and experiments are published in:
* MinhQuang Pham, Josep Crego, Jean Senellart and François Yvon. [Fixing Translation Divergences in Parallel Corpora for Neural MT](http://emnlp2018.org/program/accepted/short-papers). In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, EMNLP 2018. October 31–November 4. Brussels, Belgium.

The next picture, shows an example of similarity classification for the sentence pair:

```What do you feel ? Not . ||| Que ressentez-vous ?```

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
python -u build_data.py
   -seq_size       INT : sentences larger than this number of src/tgt words are filtered out [50]
   -max_sents      INT : Consider this number of sentences per batch (0 for all) [0]
   -seed           INT : seed for randomness [1234]
   -shuffle            : shuffle data
   -debug              : debug mode
   -h                  : this help

*  -data          FILE : training data
   -mode        STRING : how data examples are generated (p: parallel, u:uneven, i:insert, r:replace d:delete) [p]
   -replace       FILE : equivalent sequences (needed when -data_mode contains r)

+ Options marked with * must be set. The rest have default values.
```
The input data file contains one sentence pair per line, with the next fields separated by TABs:
* source sentence
* target sentence
* source/target alignments
* source part-of-speeches

 <pre>Why wait for the Euro ?   Pourquoi attendre l' Euro ?   0-0 1-1 2-1 3-2 4-3 5-4   WRB VB IN DT NNP .</pre>

(The last two fields are optional)

Available modes:
* 'p': parallel sentences
 <pre>Why wait for the Euro ?   Pourquoi attendre l' Euro ?   -1.0 -1.0 -1.0 -1.0 -1.0 -1.0   -1.0 -1.0 -1.0 -1.0 -1.0</pre>

* 'u': uneven sentences
 <pre>Why wait for the Euro ?   Cela peut donc se produire .   1.0 1.0 1.0 1.0 1.0 1.0   1.0 1.0 1.0 1.0 1.0 1.0</pre>

* 'i': insert sentence
 <pre>Why wait for the Euro ?   Pourquoi attendre l' Euro ? Il existe un précédant .   -1.0 -1.0 -1.0 -1.0 -1.0 -1.0   -1.0 -1.0 -1.0 -1.0 -1.0 1.0 1.0 1.0 1.0 1.0</pre>

* 'd': delete sequence
 <pre>Why wait for the Euro ?   l' Euro ?   1.0 1.0 1.0 -1.0 -1.0 -1.0   -1.0 -1.0 -1.0</pre>

 (needs word alignments in input FILE)

* 'r': replace sequence with equivalent part-of-speech
 <pre>Where wait for the Euro ?  Pourquoi attendre l ' Euro ?  1.0 -1.0 -1.0 -1.0 -1.0 -1.0  1.0 -1.0 -1.0 -1.0 -1.0 -1.0</pre>

 (needs word alignments and source POS-tags in -data FILE and equivalent sequences in -replace FILE)

# Learning
```
python -u similarity.py
*  -mdir          FILE : directory to save/restore models
   -seq_size       INT : sentences larger than this number of src/tgt words are filtered out [50]
   -batch_size     INT : number of examples per batch [32]
   -seed           INT : seed for randomness [1234]
   -debug              : debug mode
 [LEARNING OPTIONS]
*  -trn           FILE : training data
   -dev           FILE : validation data
   -src_voc       FILE : vocabulary of src words (needed to initialize learning)
   -tgt_voc       FILE : vocabulary of tgt words (needed to initialize learning)
   -src_emb       FILE : embeddings of src words (needed to initialize learning)
   -tgt_emb       FILE : embeddings of tgt words (needed to initialize learning)
   -src_lstm_size  INT : hidden units for src bi-lstm [256]
   -tgt_lstm_size  INT : hidden units for tgt bi-lstm [256]
   -lr           FLOAT : initial learning rate [1.0]
   -lr_decay     FLOAT : learning rate decay [0.9]
   -lr_method   STRING : GD method either: adam, adagrad, adadelta, sgd, rmsprop [adagrad]
   -aggr          TYPE : aggregation operation: sum, max, lse [lse]
   -r            FLOAT : r for lse [1.0]
   -dropout      FLOAT : dropout ratio [0.3]
   -mode        STRING : mode (alignment, sentence) [alignment]
   -max_sents      INT : Consider this number of sentences per batch (0 for all) [0]
   -n_epochs       INT : train for this number of epochs [1]
   -report_every   INT : report every this many batches [1000]

+ Options marked with * must be set. The rest have default values.
+ If -mdir exists in learning mode, learning continues after restoring the last model
+ Training data is shuffled at every epoch
```
# Inference
```
python -u similarity.py
*  -mdir          FILE : directory to save/restore models
   -batch_size     INT : number of examples per batch [32]
   -seed           INT : seed for randomness [1234]
   -debug              : debug mode
 [INFERENCE OPTIONS]
*  -epoch          INT : epoch to use ([mdir]/epoch[epoch] must exist)
*  -tst           FILE : testing data
   -show_matrix        : output formatted alignment matrix (mode must be alignment)
   -show_svg           : output alignment matrix using svg-like html format (mode must be alignment)
   -show_align         : output source/target alignment matrix (mode must be alignment)
   -show_last          : output source/target last vectors
   -show_aggr          : output source/target aggr vectors

+ Options marked with * must be set. The rest have default values.
+ -show_last, -show_aggr and -show_align can be used at the same time
```

# Fixing sentence pairs

```
python -u ./fix.py [-tau INT] [-nbest INT] [-max_sim FLOAT] [-use_punct] < FILE_WITH_ALIGNMENTS
```
