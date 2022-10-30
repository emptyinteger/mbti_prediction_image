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
from pathlib import Path


def get_file_folders(s3_client, bucket_name, prefix=""):
    file_names = []
    folders = []

    default_kwargs = {
        "Bucket": bucket_name,
        "Prefix": prefix
    }
    next_token = ""

    while next_token is not None:
        updated_kwargs = default_kwargs.copy()
        if next_token != "":
            updated_kwargs["ContinuationToken"] = next_token

        response = s3_client.list_objects_v2(**default_kwargs)
        contents = response.get("Contents")

        for result in contents:
            key = result.get("Key")
            if key[-1] == "/":
                folders.append(key)
            else:
                file_names.append(key)

        next_token = response.get("NextContinuationToken")

    return file_names, folders


def download_files(s3_client, bucket_name, local_path, file_names, folders):

    local_path = Path(local_path)

    for folder in folders:
        folder_path = Path.joinpath(local_path, folder)
        folder_path.mkdir(parents=True, exist_ok=True)

    for file_name in file_names:
        file_path = Path.joinpath(local_path, file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            bucket_name,
            file_name,
            str(file_path)
        )


def download_s3():
    client = boto3.client("s3")

    file_names, folders = get_file_folders(client, "mbti-predict-s3")
    download_files(
        client,
        "mbti-predict-s3"
        "/tmp",
        file_names,
        folders
    )
    
download_s3()
    
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SEQ_LEN = 512
tokenizer = BertTokenizer.from_pretrained('temp/bert-base-multilingual-cased/vocab.txt',local_files_only=True)

def create_mbti_bert():
  model = TFBertModel.from_pretrained('/temp/bert-base-multilingual-cased/pytorch_model.bin', local_files_only=True)
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
