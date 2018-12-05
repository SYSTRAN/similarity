# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import sys
import os
import time
from random import randint
from config import Config
from dataset import minibatches
from visualize import Visualize

class Score():
    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.A = 0.0
        self.P = 0.0
        self.R = 0.0
        self.F = 0.0

    def add(self, p, r): ### prediction, reference
        # when r < 0 => positive example, alignment exists (parallel sentence)
        # when p > 0 => alignment exists in matrix (similarity is high)
        if p*r <= 0:
            if p >= 0:
                self.TP += 1 #alignment predicted
            else:
                self.TN += 1 #alignment not predicted
        else:
            if p >= 0: 
                self.FP += 1
            else: 
                self.FN += 1
        #print("Pred:{} Ref:{}, TP:{} TN:{} FP:{} FN:{}".format(p, r, self.TP, self.TN, self.FP, self.FN))

    def add_batch_tokens(self, p, r, l):
        for s in range(len(l)): ### sentence s of batch
            for w in range(l[s]): ### all words in sentence s (length is l[s])
                self.add(p[s][w],r[s][w])

    def add_batch(self, p, r): 
        for s in range(len(p)): ### all sentences in batch
            self.add(p[s],r[s])



    def update(self):
        self.A, self.P, self.R, self.F = 0.0, 0.0, 0.0, 0.0
        if (self.TP + self.FP) > 0: self.P = 1. * self.TP / (self.TP + self.FP) #true positives out of all that were predicted positive
        if (self.TP + self.FN) > 0: self.R = 1. * self.TP / (self.TP + self.FN) #true positives out of all that were actually positive
        if (self.P + self.R) > 0.0: self.F = 2. * self.P * self.R / (self.P + self.R)
        if (self.TP + self.TN + self.FP + self.FN) > 0: self.A = 1.0 * (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

class Model():
    def __init__(self, config):
        self.config = config
        self.sess = None

    def embedding_initialize(self,NS,ES,embeddings):
        if embeddings is not None: 
            m = embeddings.matrix
        else:
            sys.stderr.write("embeddings randomly initialized\n")
            m = tf.random_uniform([NS, ES], minval=-0.1, maxval=0.1)
        return m

###################
### build graph ###
###################

    def add_placeholders(self):
        self.input_src  = tf.placeholder(tf.int32, shape=[None,None], name="input_src")  # Shape: batch_size x |Fj|  (all sentences Fj are equally sized (padded if needed))  
        self.input_tgt  = tf.placeholder(tf.int32, shape=[None,None], name="input_tgt")  # Shape: batch_size x |Ei|  (all sentences Ej are equally sized (padded if needed))  
        self.sign_src   = tf.placeholder(tf.float32, shape=[None,None], name="sign_src") # Shape: batch_size x |Ei| 
        self.sign_tgt   = tf.placeholder(tf.float32, shape=[None,None], name="sign_src") # Shape: batch_size x |Fi| 
        self.sign       = tf.placeholder(tf.float32, shape=[None], name="sign")          # Shape: batch_size (sign of each sentence: {1,-1}) 
        self.len_src    = tf.placeholder(tf.int32, shape=[None], name="len_src")
        self.len_tgt    = tf.placeholder(tf.int32, shape=[None], name="len_tgt")
        self.lr         = tf.placeholder(tf.float32, shape=[], name="lr")

    def add_model(self):
        BS = tf.shape(self.input_src)[0] #batch size
        KEEP = 1.0-self.config.dropout   # keep probability for embeddings dropout Ex: 0.7
#        print("KEEP={}".format(KEEP))

        ###
        ### src-side
        ###
        NW = self.config.src_voc_size #src vocab
        ES = self.config.src_emb_size #src embedding size
        L1 = self.config.src_lstm_size #src lstm size
        #print("SRC NW={} ES={}".format(NW,ES))
        with tf.device('/cpu:0'), tf.variable_scope("embedding_src"):
            self.LT_src = tf.get_variable(initializer = self.embedding_initialize(NW, ES, self.config.emb_src), dtype=tf.float32, name="embeddings_src")
            self.embed_src = tf.nn.embedding_lookup(self.LT_src, self.input_src, name="embed_src")
            self.embed_src = tf.nn.dropout(self.embed_src, keep_prob=KEEP)
            if self.config.share: #shared parameters
                self.embed_tgt = tf.nn.embedding_lookup(self.LT_src, self.input_tgt, name="embed_tgt")
                self.embed_tgt = tf.nn.dropout(self.embed_tgt, keep_prob=KEEP)

        with tf.variable_scope("lstm_src"):
            #print("SRC L1={}".format(L1))
            cell_fw = tf.contrib.rnn.LSTMCell(L1, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(L1, state_is_tuple=True)
            (output_src_fw, output_src_bw), (last_src_fw, last_src_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embed_src, sequence_length=self.len_src, dtype=tf.float32)
            if self.config.share: #shared parameters
                (output_tgt_fw, output_tgt_bw), (last_tgt_fw, last_tgt_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embed_tgt, sequence_length=self.len_tgt, dtype=tf.float32)

        ### divergence
        self.last_src = tf.concat([last_src_fw[1], last_src_bw[1]], axis=1)
        self.last_src = tf.nn.dropout(self.last_src, keep_prob=KEEP)
        ### alignment
        self.out_src = tf.concat([output_src_fw, output_src_bw], axis=2)
        self.out_src = tf.nn.dropout(self.out_src, keep_prob=KEEP)

#        sys.stderr.write("Total src parameters: {}\n".format(sum(variable.get_shape().num_elements() for variable in tf.trainable_variables())))

        ###
        ### tgt-side
        ###
        if not self.config.share:
            NW = self.config.tgt_voc_size #tgt vocab
            ES = self.config.tgt_emb_size #tgt embedding size
            L1 = self.config.tgt_lstm_size #tgt lstm size
            #print("TGT NW={} ES={}".format(NW,ES))
            with tf.device('/cpu:0'), tf.variable_scope("embedding_tgt"):
                self.LT_tgt = tf.get_variable(initializer = self.embedding_initialize(NW, ES, self.config.emb_tgt), dtype=tf.float32, name="embeddings_tgt")
                self.embed_tgt = tf.nn.embedding_lookup(self.LT_tgt, self.input_tgt, name="embed_tgt")
                self.embed_tgt = tf.nn.dropout(self.embed_tgt, keep_prob=KEEP)
            
            with tf.variable_scope("lstm_tgt"):
                #print("TGT L1={}".format(L1))
                cell_fw = tf.contrib.rnn.LSTMCell(L1, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(L1, state_is_tuple=True)
                (output_tgt_fw, output_tgt_bw), (last_tgt_fw, last_tgt_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embed_tgt, sequence_length=self.len_tgt, dtype=tf.float32)


        ### divergence
        self.last_tgt = tf.concat([last_tgt_fw[1], last_tgt_bw[1]], axis=1)                
        self.last_tgt = tf.nn.dropout(self.last_tgt, keep_prob=KEEP)
        ### alignment
        self.out_tgt = tf.concat([output_tgt_fw, output_tgt_bw], axis=2)
        self.out_tgt = tf.nn.dropout(self.out_tgt, keep_prob=KEEP)

#        sys.stderr.write("Total src/tgt parameters: {}\n".format(sum(variable.get_shape().num_elements() for variable in tf.trainable_variables())))
#        for variable in tf.trainable_variables():
#            sys.stderr.write("var {} params={}\n".format(variable,variable.get_shape().num_elements()))


        # next is a tensor containing similarity distances (one for each sentence pair) using the last vectors
        self.cos_similarity = tf.reduce_sum(tf.nn.l2_normalize(self.last_src, dim=1) * tf.nn.l2_normalize(self.last_tgt, dim=1), axis=1) ### +1:similar -1:divergent
        ###
        ### sentence (Grégoire and Langlais, 2017) https://arxiv.org/pdf/1709.09783.pdf
        ###
        if self.config.mode == "sentence":
            U = 256
            with tf.name_scope("sentence"):
                lastS_Dif_lastT = tf.abs(tf.subtract(self.last_src, self.last_tgt)) ### absolute element-wise difference
                lastS_Dot_lastT = self.last_src * self.last_tgt ### element-wise product
                lastS_DotDif_lastT = tf.concat([lastS_Dot_lastT, lastS_Dif_lastT], axis=1)
                self.output = tf.layers.dense(lastS_DotDif_lastT, U, activation=tf.nn.tanh, use_bias=True, kernel_initializer = tf.glorot_uniform_initializer())
                self.output = tf.layers.dense(self.output, 1, use_bias=True, kernel_initializer = tf.glorot_uniform_initializer())
                self.output = tf.reshape(self.output,[tf.shape(self.output)[0]])
        ###
        ### alignment (Legrand, Auli, Collobert, 2016) https://arxiv.org/pdf/1606.09560
        ###
        if self.config.mode == "alignment":
            R = self.config.r
#            print("R={}".format(R))
            with tf.name_scope("align"):
                self.align = tf.map_fn(lambda (x,y): tf.matmul(x,tf.transpose(y)), (self.out_src, self.out_tgt), dtype = tf.float32, name="align")
            with tf.name_scope("aggregation"):
                if self.config.aggr == "lse":
                    self.aggregation_src = tf.divide(tf.log(tf.map_fn(lambda (x,l) : tf.reduce_sum(x[:l,:],0), (tf.exp(tf.transpose(self.align,[0,2,1]) * R), self.len_tgt) , dtype=tf.float32)), R, name="aggregation_src")
                    self.aggregation_tgt = tf.divide(tf.log(tf.map_fn(lambda (x,l) : tf.reduce_sum(x[:l,:],0), (tf.exp(self.align * R), self.len_src) , dtype=tf.float32)), R, name="aggregation_tgt")
                elif self.config.aggr == "sum":
                    self.aggregation_src = tf.map_fn(lambda (x,l) : tf.reduce_sum(x[:l,:],0), (tf.transpose(self.align,[0,2,1]), self.len_tgt), dtype=tf.float32, name="aggregation_src")
                    self.aggregation_tgt = tf.map_fn(lambda (x,l) : tf.reduce_sum(x[:l,:],0), (self.align, self.len_src), dtype=tf.float32, name="aggregation_tgt")
                elif self.config.aggr == "max":
                    self.aggregation_src = tf.map_fn(lambda (x,l) : tf.reduce_max(x[:l,:],axis=0), (tf.transpose(self.align,[0,2,1]) , self.len_tgt), dtype=tf.float32, name="aggregation_src")
                    self.aggregation_tgt = tf.map_fn(lambda (x,l) : tf.reduce_max(x[:l,:],axis=0), (self.align , self.len_src), dtype=tf.float32, name="aggregation_tgt")
                else:
                    sys.stderr.write("error: bad aggregation option '{}'\n".format(self.config.aggr))
                    sys.exit()
                self.output_src = tf.log(1 + tf.exp(self.aggregation_src * self.sign_src))
                self.output_tgt = tf.log(1 + tf.exp(self.aggregation_tgt * self.sign_tgt))


    def add_loss(self):
        with tf.name_scope("loss"):
            if self.config.mode == "sentence": 
                ###cos_similarity: +1:similar, -1:opposite(divergence)
                ###sign: +1:divergence, -1:similar
                self.loss = tf.reduce_sum(tf.log(1 + tf.exp(self.output * self.sign)))
#                self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.sign, logits=self.output))
            else:
                self.loss_src = tf.reduce_mean(tf.map_fn(lambda (x,l): tf.reduce_sum(x[:l]), (self.output_src, self.len_src), dtype=tf.float32))
                self.loss_tgt = tf.reduce_mean(tf.map_fn(lambda (x,l): tf.reduce_sum(x[:l]), (self.output_tgt, self.len_tgt), dtype=tf.float32))
                self.loss = self.loss_tgt + self.loss_src

    def add_train(self):
        if   self.config.lr_method == 'adam':     optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.config.lr_method == 'adagrad':  optimizer = tf.train.AdagradOptimizer(self.lr)
        elif self.config.lr_method == 'sgd':      optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.config.lr_method == 'rmsprop':  optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.config.lr_method == 'adadelta': optimizer = tf.train.AdadeltaOptimizer(self.lr)
        else:
            sys.stderr.write("error: bad lr_method option '{}'\n".format(self.config.lr_method))
            sys.exit()

        self.train_op = optimizer.minimize(self.loss)
#        tvars = tf.trainable_variables()
#        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),1.0)
#        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def build_graph(self):
        self.add_placeholders()
        self.add_model()  
        if self.config.tst is None: 
            self.add_loss()
            self.add_train()

###################
### feed_dict #####
###################

    def get_feed_dict(self, input_src, input_tgt, sign_src, sign_tgt, sign, len_src, len_tgt, lr):
        feed = { 
            self.input_src: input_src,
            self.input_tgt: input_tgt,
            self.sign_src: sign_src,
            self.sign_tgt: sign_tgt,
            self.sign: sign,
            self.len_src: len_src,
            self.len_tgt: len_tgt,
            self.lr: lr
        }
        return feed

###################
### learning ######
###################

    def run_epoch(self, train, dev, lr):

        #######################
        # learn on trainset ###
        #######################
        nbatches = (len(train) + self.config.batch_size - 1) // self.config.batch_size
        curr_epoch = self.config.last_epoch + 1
        TLOSS = 0.0 # training loss
        ILOSS = 0.0 # intermediate loss (average over [config.report_every] iterations)
        tscore = Score()
        iscore = Score()
        ini_time = time.time()
        for iter, (src_batch, tgt_batch, raw_src_batch, raw_tgt_batch, sign_src_batch, sign_tgt_batch, sign_batch, len_src_batch, len_tgt_batch) in enumerate(minibatches(train, self.config.batch_size)):
            fd = self.get_feed_dict(src_batch, tgt_batch, sign_src_batch, sign_tgt_batch, sign_batch, len_src_batch, len_tgt_batch, lr)
            if self.config.mode == "sentence":
                _, loss, out = self.sess.run([self.train_op, self.loss, self.output], feed_dict=fd)
                tscore.add_batch(out,sign_batch)
                iscore.add_batch(out,sign_batch)
            else:
                _, loss, aggr_src, aggr_tgt, last_src, last_tgt = self.sess.run([self.train_op, self.loss, self.aggregation_src, self.aggregation_tgt, self.last_src, self.last_tgt], feed_dict=fd)
#                print("src_batch is {}".format(src_batch[0]))
#                print("tgt_batch is {}".format(tgt_batch[0]))
#                print("loss is {}".format(loss))
#                print("aggr_src is {}".format(aggr_src))
#                print("aggr_tgt is {}".format(aggr_tgt))
#                print("last_src is {}".format(last_src[0]))
#                print("last_tgt is {}".format(last_tgt[0]))
#                if iter==2: sys.exit()
                tscore.add_batch_tokens(aggr_src, sign_src_batch, len_src_batch)
                tscore.add_batch_tokens(aggr_tgt, sign_tgt_batch, len_tgt_batch)
                iscore.add_batch_tokens(aggr_src, sign_src_batch, len_src_batch)
                iscore.add_batch_tokens(aggr_tgt, sign_tgt_batch, len_tgt_batch)
            TLOSS += loss
            ILOSS += loss

            if (iter+1)%self.config.report_every == 0:
                curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
                iscore.update()
                ILOSS = ILOSS/self.config.report_every
                sys.stdout.write('{} Epoch {} Iteration {}/{} loss:{:.4f} (A{:.4f},P{:.4f},R{:.4f},F{:.4f})\n'.format(curr_time,curr_epoch,iter+1,nbatches,ILOSS,iscore.A,iscore.P,iscore.R,iscore.F))
                ILOSS = 0.0
                iscore = Score()

        TLOSS = TLOSS/nbatches
        tscore.update()
        curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        sys.stdout.write('{} Epoch {} TRAIN loss={:.4f} (A{:.4f},P{:.4f},R{:.4f},F{:.4f}) lr={:.4f}'.format(curr_time,curr_epoch,TLOSS,tscore.A,tscore.P,tscore.R,tscore.F,lr))
        unk_src = float(100) * train.nunk_src / train.nsrc
        unk_tgt = float(100) * train.nunk_tgt / train.ntgt
        div_src = float(100) * train.ndiv_src / train.nsrc
        div_tgt = float(100) * train.ndiv_tgt / train.ntgt
        sys.stdout.write(' Train set: words={}/{} %div={:.2f}/{:.2f} %unk={:.2f}/{:.2f}\n'.format(train.nsrc,train.ntgt,div_src,div_tgt,unk_src,unk_tgt))

        ##########################
        # evaluate over devset ###
        ##########################
        VLOSS = 0.0
        if dev is not None:
            nbatches = (len(dev) + self.config.batch_size - 1) // self.config.batch_size
            # iterate over dataset
            VLOSS = 0
            vscore = Score()
            for iter, (src_batch, tgt_batch, raw_src_batch, raw_tgt_batch, sign_src_batch, sign_tgt_batch, sign_batch, len_src_batch, len_tgt_batch) in enumerate(minibatches(dev, self.config.batch_size)):
                fd = self.get_feed_dict(src_batch, tgt_batch, sign_src_batch, sign_tgt_batch, sign_batch, len_src_batch, len_tgt_batch, 0.0)
                if self.config.mode == "sentence":
                    loss, out = self.sess.run([self.loss, self.output], feed_dict=fd)
                    vscore.add_batch(out, sign_batch)
                else:
                    loss, aggr_src, aggr_tgt = self.sess.run([self.loss, self.aggregation_src, self.aggregation_tgt], feed_dict=fd)
                    vscore.add_batch_tokens(aggr_src, sign_src_batch, len_src_batch)
                    vscore.add_batch_tokens(aggr_tgt, sign_tgt_batch, len_tgt_batch)
                VLOSS += loss # append single value which is a mean of losses of the n examples in the batch
            vscore.update()
            VLOSS = VLOSS/nbatches
            sys.stdout.write('{} Epoch {} VALID loss={:.4f} (A{:.4f},P{:.4f},R{:.4f},F{:.4f})'.format(curr_time,curr_epoch,VLOSS,vscore.A,vscore.P,vscore.R,vscore.F))
            unk_src = float(100) * dev.nunk_src / dev.nsrc
            unk_tgt = float(100) * dev.nunk_tgt / dev.ntgt
            div_src = float(100) * dev.ndiv_src / dev.nsrc
            div_tgt = float(100) * dev.ndiv_tgt / dev.ntgt
            sys.stdout.write(' Valid set words={}/{} %div={:.2f}/{:.2f} %unk={:.2f}/{:.2f}\n'.format(dev.nsrc,dev.ntgt,div_src,div_tgt,unk_src,unk_tgt,VLOSS,vscore.A,vscore.P,vscore.R,vscore.F))

        #################################
        #keep record of current epoch ###
        #################################
        self.config.tloss = TLOSS
        self.config.tA = tscore.A
        self.config.tP = tscore.P
        self.config.tR = tscore.R
        self.config.tF = tscore.F
        self.config.time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        self.config.seconds = "{:.2f}".format(time.time() - ini_time)
        self.config.last_epoch += 1
        self.save_session(self.config.last_epoch)
        if dev is not None:
            self.config.vloss = VLOSS
            self.config.vA = vscore.A
            self.config.vP = vscore.P
            self.config.vR = vscore.R
            self.config.vF = vscore.F
        self.config.write_config()
        return VLOSS, curr_epoch


    def learn(self, train, dev, n_epochs):
        lr = self.config.lr
        curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        sys.stdout.write("{} Training with {} sentence pairs: {} batches with up to {} examples each.\n".format(curr_time,len(train),(len(train)+self.config.batch_size-1)//self.config.batch_size,self.config.batch_size))
        best_score = 0
        best_epoch = 0
        for iter in range(n_epochs):
            score, epoch = self.run_epoch(train, dev, lr)  ### decay when score does not improve over the best
            curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
            if iter == 0 or score <= best_score: 
                best_score = score
                best_epoch = epoch
            else:
                lr *= self.config.lr_decay # decay learning rate

###################
### inference #####
###################

    def inference(self, tst):

        if self.config.show_svg: print "<html>\n<body>"
        nbatches = (len(tst) + self.config.batch_size - 1) // self.config.batch_size
        score = Score()
        n_sents = 0
        for iter, (src_batch, tgt_batch, raw_src_batch, raw_tgt_batch, sign_src_batch, sign_tgt_batch, sign_batch, len_src_batch, len_tgt_batch) in enumerate(minibatches(tst, self.config.batch_size)):
            fd = self.get_feed_dict(src_batch, tgt_batch, sign_src_batch, sign_tgt_batch, sign_batch, len_src_batch, len_tgt_batch, 0.0) 

            if self.config.mode == "sentence":
                out_batch, last_src_batch, last_tgt_batch = self.sess.run([self.output, self.last_src, self.last_tgt], feed_dict=fd)
                if tst.annotated: score.add_batch(out_batch, sign_batch)
                for i_sent in range(len(out_batch)):
                    n_sents += 1
                    v = Visualize(n_sents,raw_src_batch[i_sent],raw_tgt_batch[i_sent],out_batch[i_sent])
                    last_src = []
                    last_tgt = []
                    if self.config.show_last:
                        last_src = last_src_batch[i_sent]
                        last_tgt = last_tgt_batch[i_sent]
                    v.print_vectors(last_src,last_tgt,aggr_src=[],aggr_tgt=[],align=[])
            else:   
                align_batch, aggr_src_batch, aggr_tgt_batch, out_src_batch, out_tgt_batch, last_src_batch, last_tgt_batch, sim_batch = self.sess.run([self.align, self.aggregation_src, self.aggregation_tgt, self.out_src, self.out_tgt, self.last_src, self.last_tgt, self.cos_similarity], feed_dict=fd)
                if tst.annotated: 
                    score.add_batch_tokens(aggr_src_batch, sign_src_batch, len_src_batch)
                    score.add_batch_tokens(aggr_tgt_batch, sign_tgt_batch, len_tgt_batch)
                for i_sent in range(len(align_batch)):
                    n_sents += 1
                    v = Visualize(n_sents,raw_src_batch[i_sent],raw_tgt_batch[i_sent],sim_batch[i_sent])
                    if self.config.show_svg: 
                        v.print_svg(aggr_src_batch[i_sent],aggr_tgt_batch[i_sent],align_batch[i_sent])
                    elif self.config.show_matrix: 
                        v.print_matrix(aggr_src_batch[i_sent],aggr_tgt_batch[i_sent],align_batch[i_sent])
                    else:
                        last_src = []
                        last_tgt = []
                        aggr_src = []
                        aggr_tgt = []
                        align = []
                        if self.config.show_last: 
                            last_src = last_src_batch[i_sent]
                            last_tgt = last_tgt_batch[i_sent]
                        if self.config.show_aggr: 
                            aggr_src = aggr_src_batch[i_sent]
                            aggr_tgt = aggr_tgt_batch[i_sent]
                        if self.config.show_align: 
                            align = align_batch[i_sent]
                        v.print_vectors(last_src,last_tgt,aggr_src,aggr_tgt,align)

        if tst.annotated:
            score.update()
            unk_s = float(100) * tst.nunk_src / tst.nsrc
            unk_t = float(100) * tst.nunk_tgt / tst.ntgt
            div_s = float(100) * tst.ndiv_src / tst.nsrc
            div_t = float(100) * tst.ndiv_tgt / tst.ntgt
            sys.stdout.write('TEST words={}/{} %div={:.2f}/{:.2f} %unk={:.2f}/{:.2f} (A{:.4f},P{:.4f},R{:.4f},F{:.4f}) (TP:{},TN:{},FP:{},FN:{})\n'.format(tst.nsrc,tst.ntgt,div_s,div_t,unk_s,unk_t,score.A,score.P,score.R,score.F,score.TP,score.TN,score.FP,score.FN))

        if self.config.show_svg: print "</body>\n</html>"


###################
### session #######
###################

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=20)

        if self.config.epoch is not None: ### restore a file for testing
            fmodel = self.config.mdir + '/epoch' + self.config.epoch
            sys.stderr.write("Restoring model: {}\n".format(fmodel))
            self.saver.restore(self.sess, fmodel)
            return

        if self.config.mdir: ### initialize for training or restore previous
            if not os.path.exists(self.config.mdir + '/checkpoint'): 
                sys.stderr.write("Initializing model\n")
                self.sess.run(tf.global_variables_initializer())
            else:
                sys.stderr.write("Restoring previous model: {}\n".format(self.config.mdir))
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.config.mdir))

    def save_session(self,e):
        if not os.path.exists(self.config.mdir): os.makedirs(self.config.mdir)
        file = "{}/epoch{}".format(self.config.mdir,e)
        self.saver.save(self.sess, file) #, max_to_keep=4, write_meta_graph=False) # global_step=step, keep_checkpoint_every_n_hours=2

    def close_session(self):
        self.sess.close()


