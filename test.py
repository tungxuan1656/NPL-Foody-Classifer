#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import glob
import re
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


def remove_stop_words(data):
    out = []
    for d in data:
        o = re.sub(r"""[0-9!"\[\]#$%&()*+,-./:;<=>?@^`{|}~\\\n–\']""", '', d).replace('_', ' ')
        out.append(o)
    return out


def load_comments(path):
    paths = glob.glob(path)
    comments = []
    for path in paths:
        with open(path, encoding="utf-8") as file:
            text = file.read()
            text_lower = text.lower()
            comments.append(text_lower)
        file.close()
    print(len(comments))
    return remove_stop_words(comments)


# In[3]:


train_neg = load_comments('data_train/train/neg/*.txt')
train_pos = load_comments('data_train/train/pos/*.txt')
val_neg = load_comments('data_train/test/neg/*.txt')
val_pos = load_comments('data_train/test/pos/*.txt')
test_neg = load_comments('data_test/test/neg/*.txt')
test_pos = load_comments('data_test/test/pos/*.txt')


# In[20]:


import pickle


# In[24]:


with open('train_neg.dt', 'wb') as datafile:
    pickle.dump(train_neg, datafile)
with open('train_pos.dt', 'wb') as datafile:
    pickle.dump(train_pos, datafile)
with open('val_neg.dt', 'wb') as datafile:
    pickle.dump(val_neg, datafile)
with open('val_pos.dt', 'wb') as datafile:
    pickle.dump(val_pos, datafile)
with open('test_neg.dt', 'wb') as datafile:
    pickle.dump(test_neg, datafile)
with open('test_pos.dt', 'wb') as datafile:
    pickle.dump(test_pos, datafile)

