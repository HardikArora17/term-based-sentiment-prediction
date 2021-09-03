#IMPORTING THE NECESSARY LIBRARIES

import pandas as pd
import numpy as np
import os
import pickle
import torch
import random
import re
import nltk
from tqdm import trange
nltk.download('punkt',quiet=True)
from tqdm.autonotebook import tqdm
from transformers import BertTokenizer,AutoModel
import torch.nn as nn
import numpy as np

#CHECKING FOR GPU DEVICE

if torch.cuda.is_available():        
    device = torch.device("cuda:0")

else:
  print("GPU NOT AVAILABLE USING CPU")
  device = torch.device("cpu")



model_name='bert-base-uncased'
tokenizer=BertTokenizer.from_pretrained(model_name)


#LOADING GLOVE EMBEDDINGS AND VOCABULARY
with open("models/embeddings_train_100.pickle",'rb') as out:
  embeddings=pickle.load(out)

with open("models/vocab_train_100.pickle",'rb') as out:
  vocab=pickle.load(out)

#CREATING VOCAB DICTIONARY
word_to_ix={word:i for i,word in enumerate(vocab)}
VOCAB_LEN=len(word_to_ix)

ix_to_word={i:word for word, i in word_to_ix.items()}

#  tokenizing a sentence
def test_convert(sentence,name):
  sentence=nltk.word_tokenize(sentence.lower())
  sen=[word_to_ix.get(i,1) for i in sentence]
  n=[word_to_ix.get(name.lower(),1)]

  return sen,n

#FUNCTION FOR REMOVING BASIC CHARACTERS
def preprocess_text(string:str):
    string=string.lower()
    punctuations = '''!()-[]{};:'"\<>/?@#$^&*_~'''
    string=string.replace('â€™'," ")
    string=string.replace('\n',"")
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, " ") 

    return string


#MAIN-MODEL
class BertModel(nn.Module):
    def __init__(self,out_features):

        super(BertModel, self).__init__()
        weights_matrix=torch.from_numpy(embeddings).float().to(device)

        self.embedding =nn.Embedding.from_pretrained(weights_matrix,freeze=False)
        self.out_features = out_features    
        self.lstm =nn.LSTM(100,50//2,batch_first=True,num_layers=1,bidirectional=True)
        self.flatten=nn.Flatten()
        self.dropout1=nn.Dropout(0.1)

        self.linear1=nn.Linear(150,32*2)
        self.linear2=nn.Linear(32*2,16*2)
        self.linear3=nn.Linear(16*2,8*2)

        self.last_dense = nn.Linear(8*2, self.out_features)
        self.relu = nn.ReLU()

    def forward(self, t,t1):
        
        encoding=self.embedding(t)  
        encoding_name=self.embedding(t1).squeeze(dim=1) 
        l=self.lstm(encoding)[0]
        l=torch.mean(l,dim=1) 
        l=torch.cat((encoding_name,l),dim=1) 
        l = self.relu(self.linear1(l)) 
        l = self.dropout1(l)
        l = self.relu(self.linear2(l)) 
        l = self.relu(self.linear3(l)) 
        l = self.dropout1(l)

        model_output = torch.nn.functional.softmax(self.last_dense(l),dim=1)
        
        return model_output

text_model = BertModel(3)
text_model.to(device)

#LOADING MODEL_DICT
text_model.load_state_dict(torch.load("models/intern_model_state_dict.pt",map_location=device))
text_model.eval()


#FUNCTION FOR PREDICTING SENTIMENT
def predict_result(sen1,name1):
    sen,n=test_convert(sen1,name1)
    data={'ids':torch.tensor([sen]),
        'ids_name':torch.tensor([n])}

    ids = data['ids'].to(device,dtype = torch.long)
    ids_name = data['ids_name'].to(device,dtype = torch.long)
    out_val= text_model(ids,ids_name)
    pred_sent=torch.max(out_val.detach().cpu(),dim=1)[1].item()
    return pred_sent


#LOADING ,CALCULATING AND SAVING PREDICTIONS FOR TEST DATA
tqdm.pandas()

test_data=pd.DataFrame()
x_test=pd.read_csv('data/test_data.csv')


sentences=list(map(lambda x: preprocess_text(x),list(x_test['text'])))
aspects=list(map(lambda x: preprocess_text(x),list(x_test['aspect'])))

test_data['text']=x_test['text']
test_data['aspect']=x_test['aspect']
predictions=[predict_result(sentences[i],aspects[i]) for i in trange(len(x_test['text']))]
test_data['label']=predictions

test_data.to_csv('data/results/test.csv')