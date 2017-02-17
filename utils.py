from __future__ import division

import random

import numpy as np

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def get_batches(train_captions, batch_size):
    train_batches = []
    # train_caption 为{id1:set1, id2:set2,...}  其中id为set中id(对应sentence)的数量，长度为id的sentence的id的结合为set
    for sent_length, caption_set in train_captions.items():
        caption_set = list(caption_set)
        random.shuffle(caption_set)
        num_captions = len(caption_set)
        num_batches = num_captions // batch_size   # 必须选择caption中id对应的set集合在256以上才加入，这样才能保证minbatch=256
        for i in range(num_batches+1):
            end_idx = min((i+1)*batch_size, num_captions)
            new_batch = caption_set[(i*batch_size):end_idx]
            if len(new_batch) == batch_size:
                train_batches.append((new_batch, sent_length))
    random.shuffle(train_batches)
    return train_batches

def formatPlaceholder(batch_item, batch_size, img_dim, train_caption_id2sentence, train_caption_id2image_id, train_image_id2feature):
    (caption_ids, sent_length) = batch_item   # caption_ids为sentence的对应的id，是一个列表，他们拥有相同的长度sent_length
    num_captions = len(caption_ids)   # 等于batch_size的大小，每个batch_size中短语的长度是一致的，等于lstm中n_step-1
    sentences = np.array([train_caption_id2sentence[k] for k in caption_ids])
    images = np.array([train_image_id2feature[train_caption_id2image_id[k]] for k in caption_ids])  # 变成了256×img_dim
    targets = sentences[:,1:]
    
    sentences_template = np.zeros([batch_size, sent_length])
    images_template = np.zeros([batch_size, img_dim])
    targets_template = -np.ones([batch_size, sent_length + 1])
    
    sentences_template[range(num_captions),:] = sentences
    images_template[range(num_captions),:] = images
    targets_template[range(num_captions), 1:sent_length] =  targets
    assert (targets_template[:,[0,-1]] == -1).all() # front and back should be padded with -1
    
    return (sentences_template, images_template, targets_template)

def train_data_iterator(train_captions, train_caption_id2sentence, train_caption_id2image_id, train_image_id2feature, config):
    batch_size = config.batch_size
    img_dim = config.img_dim
    # 返回的是[(batch_size1,对应短语长度1)，(batch_size2, 对应短语长度2)，..] batch的长度为256，是一个列表，每个短语对应一个id，通过self.caption_id2sentence能找到对应的sentence是什么
    train_batches = get_batches(train_captions, batch_size)
    for batch_item in train_batches:
        sentences, images, targets = formatPlaceholder(batch_item, batch_size, img_dim, train_caption_id2sentence, train_caption_id2image_id, train_image_id2feature,)
        yield (sentences, images, targets)
