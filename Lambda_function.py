#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import *
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
import tensorflow_addons as tfa
import io
import boto3

s3 = boto3.client('s3')

BUCKET_NAME = 'mbti-predict-s3' 
OBJECT_NAME = ['mbti_model.h5'] 
PATH_NAME = '/tmp/'

def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)

download_s3_folder(BUCKET_NAME, '/cache','/tmp')
        
for obj in OBJECT_NAME:
  s3.download_file(BUCKET_NAME, obj, PATH_NAME+obj)

os.system("pwd")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SEQ_LEN = 512
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir='./tmp')

def create_mbti_bert():
  model = TFBertModel.from_pretrained("bert-base-multilingual-cased", from_pt=True, cache_dir='./tmp')
  token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
  mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
  segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')
  bert_outputs = model([token_inputs, mask_inputs, segment_inputs])
  bert_outputs = bert_outputs[1]
  mbti_first = tf.keras.layers.Dense(16, activation='softmax', kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02))(bert_outputs)
  mbti_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], mbti_first)
  mbti_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
      metrics=['sparse_categorical_accuracy'])
  return mbti_model

import sys
mod = sys.modules[__name__]
mbti_model = create_mbti_bert()

mbti_model.load_weights("/tmp/mbti_model.h5")

setattr(mod, 'model', mbti_model)

def mean_answer_label(*preds):
  preds_sum = np.zeros(preds[0].shape[0])
  for pred in preds:
    preds_sum += np.argmax(pred, axis=-1)
  return np.round(preds_sum/len(preds), 0).astype(int)


mod = sys.modules[__name__]

def sentence_convert_data(data):
    global tokenizer
    
    tokens, masks, segments = [], [], []
    token = tokenizer.encode(data, max_length=SEQ_LEN, padding='max_length', truncation=True)
    
    num_zeros = token.count(0) 
    mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros 
    segment = [0]*SEQ_LEN

    tokens.append(token)
    segments.append(segment)
    masks.append(mask)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    return [tokens, masks, segments]

def api_predict(sentence):
    cat_dict = {'0':'enfj','1':'enfp','2':'entj','3':'entp','4':'esfj','5':'esfp','6':'estj','7':'estp',
             '8':'infj','9':'infp','10':'intj','11':'intp','12':'isfj','13':'isfp','14':'istj','15':'istp'}
    result_dict = {}
    global mod
    sentence = re.sub("[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…]+", "", sentence)
    sentence = re.sub("\\n+", " ", sentence)
    sentence = re.sub("\\t+", " ", sentence)
    data_x = sentence_convert_data(sentence)   
    setattr(mod, 'chung', model.predict(data_x, batch_size=1))
    preds = str(mean_answer_label(chung).item())
    intChung = []
    for key, value in cat_dict.items():
        appendInt = int(np.round(chung[0,int(key)]*100))
        intChung.append(appendInt)
        result_dict.update({value:appendInt})
    
    x = np.arange(16)
    predi = cat_dict[preds]
    result_dict.update({'mbti':predi})

    return result_dict

def handler(event, context):
  text = str(event.get('text'))
  result = api_predict(text)
  return {
    "statusCode": 200,
    "headers": {
      "Content-Type": "application/json"
      },
    "body": json.dumps(result)
    }
