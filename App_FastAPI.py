#!/usr/bin/env python
# coding: utf-8

# In[1]:

from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI

# In[2]:

import sys
import pickle
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, TextClassificationPipeline
from arabert.preprocess import ArabertPreprocessor

# In[3]:

model_name = "aubmindlab/bert-base-arabertv02-twitter"
arabert_prep = ArabertPreprocessor(model_name=model_name, keep_emojis=False)
arabert_tokenizer = AutoTokenizer.from_pretrained(model_name)
label_dict = {'LABEL_0' : 'AE', 'LABEL_1' : 'BH', 'LABEL_2' : 'DZ', 'LABEL_3' : 'EG', 'LABEL_4' : 'IQ', 'LABEL_5' : 'JO', 'LABEL_6' : 'KW', 'LABEL_7' : 'LB', 'LABEL_8' : 'LY',
              'LABEL_9' : 'MA', 'LABEL_10' : 'OM', 'LABEL_11' : 'PL', 'LABEL_12' : 'QA', 'LABEL_13' : 'SA', 'LABEL_14' : 'SD', 'LABEL_15' : 'SY', 'LABEL_16' : 'TN', 'LABEL_17' : 'YE'}

# In[4]:

def tokenize(text):
    tokens = arabert_tokenizer.tokenize(text)
    return tokens

# In[5]:

class InputText(BaseModel):
    text : str

# In[6]:

def server(tfidf, machine_learning_model, deep_learning_model):
    app = FastAPI()

    
    @app.get("/")
    def root():
        return {"Hello" : 'User'}

    @app.post("/predict_ml/")
    def predict(input:InputText):
        print(input.text)
    
        sentence = input.text
        sentence_prep = arabert_prep.preprocess(sentence)
        X_tf = tfidf.transform([sentence_prep])
        predicted_class = machine_learning_model.predict(X_tf)
        return {"output" : predicted_class[0]}
    
    @app.post("/predict_dl/")
    def predict(input:InputText):
        print(input.text)
    
        sentence = input.text
        sentence_prep = arabert_prep.preprocess(sentence)
        predicted_class = label_dict[deep_learning_model(sentence_prep)[0]['label']]
        return {"output" : predicted_class}
    
    return app

# In[7]:

def run_server(tfidf_path, machine_learning_model_path, deep_learning_model_path, host, port):
    tfidf = pickle.load(open(tfidf_path, 'rb'))
    machine_learning_model = pickle.load(open(machine_learning_model_path, 'rb'))
    
    deep_learning_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=18)
    deep_learning_model.load_state_dict(torch.load(deep_learning_model_path))
    pipe = TextClassificationPipeline(model=deep_learning_model, tokenizer=arabert_tokenizer)

    app = server(tfidf, machine_learning_model, pipe)
    uvicorn.run(app, host=host, port=port)

# In[8]:

def cli(sys_argv):
    parser = argparse.ArgumentParser()
    
    # ----------------
    # Model parameters
    # ----------------
    parser.add_argument(
        '-t',
        '--tfidf_path',
        help='tfidf to load',
        required=True
    )

    parser.add_argument(
        '-ml',
        '--machine_learning_model_path',
        help='model to load',
        required=True
    )

    parser.add_argument(
        '-dl',
        '--deep_learning_model_path',
        help='model to load',
        required=True
    )

    # ----------------
    # Server parameters
    # ----------------
    parser.add_argument(
        '-p',
        '--port',
        help='port for server (default: 8000)',
        default=8000,
        type=int,
    )

    parser.add_argument(
        '-H',
        '--host',
        help='host for server (default: 0.0.0.0)',
        default='0.0.0.0'
    )
    args = parser.parse_args(sys_argv)
    run_server(args.tfidf_path, args.machine_learning_model_path, args.deep_learning_model_path, args.host, args.port)

# In[9]:

if __name__ =="__main__":
    cli(sys.argv[1:])



