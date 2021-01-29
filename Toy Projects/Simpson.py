#!/usr/bin/env python
# coding: utf-8

# In[110]:


import re 
import pandas as pd 
import numpy as np
from time import time  #
from collections import defaultdict

import spacy

import logging  
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


# In[ ]:


df = pd.read_csv('simpsons_dataset.csv')
df.shape


# In[10]:


df.isnull().sum()


# ## 전처리 (nltk)

# In[12]:


#결측치 제거

df = df.dropna().reset_index(drop=True)
df.isnull().sum()


# In[130]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

texts = df['spoken_words'].to_list()
stop_words = set(stopwords.words('english'))

tokens = [word_tokenize(txt.lower()) for txt in texts]
tokens_wo_stopword = [[word for word in token if not word in stop_words] for token in tokens]
tokens2 = [[word for word in token if len(word) >= 3] for token in tokens]


# In[131]:


tokens2


# ## Word2Vec

# ### 1. Gensim

# In[43]:


import multiprocessing

from gensim.models import Word2Vec


# In[44]:


cores = multiprocessing.cpu_count()


# In[144]:


w2v_model = Word2Vec(tokens2,
                     min_count=20,
                     window=2,
                     iter=30,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)


# In[145]:


w2v_model.init_sims(replace=True)


# In[146]:


w2v_model.wv.most_similar(positive=["homer"])


# In[160]:


w2v_model.wv.most_similar(positive=["marge"])


# In[168]:


w2v_model.wv.most_similar(positive=["donut"])


# ## 시각화

# In[148]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[156]:


vocab = list(w2v_model.wv.vocab)
X = w2v_model[vocab]

print(len(X))
print(X[0][:10])
tsne = TSNE(n_components=2)

# 100개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(X[:100, :])


# In[158]:


df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])
df.shape


# In[159]:


fig = plt.figure()
fig.set_size_inches(48, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.show()


# In[136]:


#================교재처럼 전처리 하려 했으나 실패=======================

from common.util import create_contexts_target
contexts, target = create_contexts_target(corpus[0], window_size=1)
for i in range(len(corpus)):
    a, b = create_contexts_target(corpus[i+1], window_size=1)
    contexts = np.concatenate((contexts,a))
    target = np.concatenate((target,b))

