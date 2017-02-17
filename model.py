# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap
from copy import deepcopy
import cPickle as pickle
import os, time
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

import tensorflow as tf
import numpy as np

from utils import train_data_iterator, sample

with open('data_files/val_image_id2feature.pkl', 'r') as f:
    val_image_id2feature = pickle.load(f)

class Config(object):
    img_dim = 1024
    hidden_dim = embed_dim = 512
    max_epochs = 50
    batch_size = 256
    keep_prob = 0.75
    layers = 3
    model_name = 'model_keep=%.2f_batch=%d_hidden_dim=%d_embed_dim=%d_layers=%d' % (keep_prob, batch_size, hidden_dim, embed_dim, layers)

class Model(object):

    def __init__(self, config):
        self.config = config
        self.load_data()
        self.vocab_size = len(self.index2token)  #8834

        # placeholders
        self._sent_placeholder = tf.placeholder(tf.int32, shape=[self.config.batch_size, None], name='sent_ph')  #None代表n_steps
        self._img_placeholder = tf.placeholder(tf.float32, shape=[self.config.batch_size, self.config.img_dim], name='img_ph') # lstm unfold的第一维就是图片
        self._targets_placeholder = tf.placeholder(tf.int32, shape=[self.config.batch_size, None], name='targets')
        self._dropout_placeholder = tf.placeholder(tf.float32, name='dropout_placeholder')


        # Input layer
        with tf.variable_scope('CNN'):
            W_i = tf.get_variable('W_i', shape=[self.config.img_dim, self.config.embed_dim])
            b_i = tf.get_variable('b_i', shape=[self.config.batch_size, self.config.embed_dim])
            img_input = tf.expand_dims(tf.nn.sigmoid(tf.matmul(self._img_placeholder, W_i) + b_i), 1)  # 256*1*512 256表示batch_size 512表示输入向量维度
            print 'Img:', img_input.get_shape()
        with tf.variable_scope('sent_input'):
            # self.vocab_size 表示单词大小 self.config.embed_dim 表示一个单词转换为特征向量的维度   _sent_placeholder将单词转为向量维度
            word_embeddings = tf.get_variable('word_embeddings', shape=[self.vocab_size, self.config.embed_dim])
            sent_inputs = tf.nn.embedding_lookup(word_embeddings, self._sent_placeholder)
            print 'Sent:', sent_inputs.get_shape()   # 256×?*512 其中256是batch_size ?表示n_steps-1  512表示输入向量维数
        with tf.variable_scope('all_input'):
            all_inputs = tf.concat(1, [img_input, sent_inputs])    # 256*?*512  img_input 和sent_inputs连在一起了
            print 'Combined:', all_inputs.get_shape()


        # LSTM layer
        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim,forget_bias=1,input_size=self.config.embed_dim)
        lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=self._dropout_placeholder, output_keep_prob=self._dropout_placeholder)
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_dropout] * self.config.layers)
        initial_state = stacked_lstm.zero_state(self.config.batch_size, tf.float32)
        # outputs 256*?*512  其中256是batch_size ?是n_steps 512是维数，因为n_steps大小不固定，所以是dynamic_rnn
        outputs, final_state = tf.nn.dynamic_rnn(stacked_lstm, all_inputs, initial_state=initial_state, scope='LSTM')
        output = tf.reshape(outputs, [-1, self.config.hidden_dim]) # for matrix multiplication
        # 将每个样例的n_step放在了一起，变成了(256*?) * 512  其中每n_steps个512 表示一个样例所有的ouputs
        self._final_state = final_state
        print 'Outputs (raw):', outputs.get_shape()
        print 'Final state:', final_state.get_shape()
        print 'Output (reshaped):', output.get_shape()


        # Softmax layer
        with tf.variable_scope('softmax'):   #每个unfolding的w和b共享权重
            softmax_w = tf.get_variable('softmax_w', shape=[self.config.hidden_dim, self.vocab_size])
            softmax_b = tf.get_variable('softmax_b', shape=[self.vocab_size])
            logits = tf.matmul(output, softmax_w) + softmax_b   #结果为(256*?)*vocab_size  ?表示n_steps个数，前n_steps行表示一个样例所对应的全部输出， 共享softmax_w和softmx_b
        print 'Logits:', logits.get_shape()


        # Predictions
        self.logits = logits
        self._predictions = predictions = tf.argmax(logits,1)
        # 返回每一行最大元素所在的下标，这样得到的维数就是 (256*?)*1
        # 内容为最大元素所对应的下标， 故前n_steps行表示的是第一个样例所对应的每个概率单词的id
        print 'Predictions:', predictions.get_shape()


        # Minimize Loss
        targets_reshaped = tf.reshape(self._targets_placeholder,[-1])  # 将target转为(256*?)*1
        print 'Targets (raw):', self._targets_placeholder.get_shape()
        print 'Targets (reshaped):', targets_reshaped.get_shape()
        with tf.variable_scope('loss'):
            # _targets is [-1, ..., -1] so that the first and last logits are not used
            # these correspond to the img step and the <eos> step
            # see: https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#sparse_softmax_cross_entropy_with_logits
            self.loss = loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets_reshaped, name='ce_loss'))
            print 'Loss:', loss.get_shape()
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(loss)
    

    def load_data(self, type='train'):
        if type == 'train':
            with open('data_files/index2token.pkl','r') as f:
                self.index2token = pickle.load(f) #id对应的字典 如同{0:u'raining', 1:u'writing', 2:'yellow', ...8841:'awake', '8842:'<EOS>'}
            with open('data_files/preprocessed_train_captions.pkl','r') as f:
                #train_captions: {sentence长度:train_caption_id}
                self.train_captions, self.train_caption_id2sentence, self.train_caption_id2image_id = pickle.load(f)
            with open('data_files/train_image_id2feature.pkl','r') as f:
                self.train_image_id2feature = pickle.load(f)


    def run_epoch(self, session, train_op):
        # 一次epoch迭代所有样本次数，total_step为迭代完所有样本需要的mini_batch次数
        total_steps = sum(1 for x in train_data_iterator(self.train_captions, self.train_caption_id2sentence, self.train_caption_id2image_id, self.train_image_id2feature, self.config))
        total_loss = []
        if not train_op:   # 不是训练过程
            train_op = tf.no_op()
        start = time.time()

        for step, (sentences, images, targets) in enumerate(train_data_iterator(self.train_captions, self.train_caption_id2sentence, self.train_caption_id2image_id, self.train_image_id2feature, self.config)):

            feed = {self._sent_placeholder: sentences,
                    self._img_placeholder: images,
                    self._targets_placeholder: targets,
                    self._dropout_placeholder: self.config.keep_prob}
            loss, _ = session.run([self.loss, train_op], feed_dict=feed)
            total_loss.append(loss)

            if (step % 50) == 0:
                print '%d/%d: loss = %.2f time elapsed = %d' % (step, total_steps, np.mean(total_loss) , time.time() - start)

        print 'Total time: %ds' % (time.time() - start) 
        return total_loss


    def generate_caption(self, session, img_feature,toSample=False):
        dp = 1
        img_template = np.zeros([self.config.batch_size, self.config.img_dim])
        img_template[0,:] = img_feature

        sent_pred = np.ones([self.config.batch_size, 1])*3591 # <SOS>
        while sent_pred[0,-1] != 3339 and (sent_pred.shape[1] - 1) < 50:
           # sent_pred[0,-1]表示第0行最后一个词不是id=33339对应的词语(表示结束)，那么生成的caption小于50才执行
            feed = {self._sent_placeholder: sent_pred,    # lstm unfold 每循环一次增长一次
                    self._img_placeholder: img_template,
                    self._targets_placeholder: np.ones([self.config.batch_size,1]), # dummy variable
                    self._dropout_placeholder: dp}

            idx_next_pred = np.arange(1, self.config.batch_size + 1)*(sent_pred.shape[1] + 1) - 1

            if toSample:
                logits = session.run(self.logits, feed_dict=feed)
                next_logits = logits[idx_next_pred,:]
                raw_predicted = []
                for row_idx in range(next_logits.shape[0]):
                    idx = sample(next_logits[row_idx,:])
                    raw_predicted.append(idx)
                raw_predicted = np.array(raw_predicted)
            else:
                raw_predicted = session.run(self._predictions, feed_dict=feed)
                raw_predicted = raw_predicted[idx_next_pred]

            next_pred = np.reshape(raw_predicted, (self.config.batch_size,1))
            sent_pred = np.concatenate([sent_pred, next_pred], 1)

        predicted_sentence = ' '.join(self.index2token[idx] for idx in sent_pred[0,1:-1])
        return predicted_sentence

def generate_captions_val(session, model, epoch, valSize=200, debug=False):
    total_images = len(val_image_id2feature)
    results_list = []
    val_set = val_image_id2feature.items()
    for step, (img_id, img_feature) in enumerate(val_set[:valSize]):
        generated_caption = model.generate_caption(session, img_feature)
        line = {}
        line['image_id'] = img_id
        line['caption'] = generated_caption
        results_list.append(line)

        print '%d/%d imgid %d: %s' % (step, valSize, img_id, generated_caption)

    results_dir = '%s/results' % model.config.model_name
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    fn = '%s/val_res_%d.json' % (results_dir, epoch)
    with open(fn, 'w') as f:
        json.dump(results_list, f, sort_keys=True, indent=4)
    print 'json results dumped in %s' % fn

def evaluateModel(model_json):
    cocoRes = coco.loadRes(model_json)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds() # to evaluate only on subset of images    
    cocoEval.evaluate()
    results = {}
    for metric, score in cocoEval.eval.items():
        results[metric] = score
    del cocoRes
    del cocoEval
    return results

def main():
    config = Config()   # lstm配置信息
    with tf.variable_scope('CNNLSTM') as scope:
        model = Model(config)    #创建model
    loss_history = []
    all_results_json = {}
    init = tf.initialize_all_variables()    #初始化所有的变量
    saver = tf.train.Saver()      # 会话保存信息

    with tf.Session() as session:
        session.run(init)
        for epoch in range(config.max_epochs):    #执行epoch迭代
            ## train model
            print 'Epoch %d' % (epoch+1)
            total_loss =  model.run_epoch(session, model.train_op)
            loss_history.extend(total_loss)
            print 'Average loss: %.1f' % np.mean(total_loss)

            if not os.path.exists(config.model_name):
                os.mkdir(config.model_name)
            if not os.path.exists('%s/weights' % config.model_name):
                os.mkdir('%s/weights' % config.model_name)
            saver.save(session, '%s/weights/model' % config.model_name, global_step=epoch)

            if not os.path.exists('%s/loss' % config.model_name):
                os.mkdir('%s/loss' % config.model_name)
            pickle.dump(loss_history, open('%s/loss/loss_history.pkl' % (config.model_name),'w'))

            ## evaluate on 200 validation images
            generate_captions_val(session, model, epoch)
            resFile = '%s/results/val_res_%d.json' % (config.model_name, epoch)
            # results = evaluateModel(resFile)
            # all_results_json[epoch] = results
            print '-'*30

            # with open('%s/results/evaluation_val.json' % config.model_name, 'w') as f:
                # json.dump(all_results_json, f, sort_keys=True, indent=4)

if __name__ == '__main__':
    main()
