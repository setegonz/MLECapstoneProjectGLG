#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import spacy
import pandas as pd
import numpy as np

nlp = spacy.load("en_core_web_sm")


# ### Import Data

# import pickle
# p0 = pickle.load( open( "data_part_0.pickle", "rb" ) )
# p0 = p0.iloc[:, 3:].copy()
# 
# p0

# In[2]:


'''
Import CSV file
'''

p0 = pd.read_csv('chunk0.csv')

print(p0.shape)
print()

p0 = p0.iloc[:, 2:].copy()
p0.head()


# ### Preprocess data:
# 
# * Filter out unwanted columns
# * Number of article and length of each article

# In[3]:


# Unnecessary for the csv file
cols = p0.columns.values.tolist()
# key = cols[:6] + cols[7] + cols[9]
# key
# key = cols[:6]
# key.append(cols[7])
# key.append(cols[9])
key = cols[:5] + cols[-2:]
print(key)

df = p0[key].copy()


# In[4]:


p0


# In[5]:


import statistics as stat

articles = p0.loc[:, 'article'].values.tolist()
articlesLen = [len(str(article)) for article in articles]
len(articlesLen), round(stat.mean(articlesLen), 3)


# ### NER and get Document Embedding
# 
# **Steps**
# 1. Run NER on each article
# 1. For each recognized entities in NER: get embedding for the query and entity. Do elementwise multiplication to get combined embedding of NER results. 
# 1. Get embedding of the document using Spacy. 
# 1. Concat the embeddings of the document together to get final embedding of the documents

# In[6]:


## For each article
def entityPerArticle(article):
    debug = False
    doc = nlp(article)

    d = {}
    for ent in doc.ents:
        d[ent.text] = ent.label_

    if debug: print(d)
    return d


# In[7]:


p0.loc[0, :]


## For test purpose
# In[ ]:


# entities = []
# i = 0
# for article in articles:
#     ent = entityPerArticle(str(article))
#     entities.append(ent)
#     i+=1
#     if i <=3:
#         print(ent)
#         print()
    


# # In[ ]:


# p0['entities'] = entities
# key.append('entities')
# df = p0[key]
# df


# # In[ ]:


# df.to_csv('chunk0_ner.csv', index=False)


# # In[ ]:


# for sent in doc.sents:
#     for token in sent:
#         print(token, token.pos_)


# # In[ ]:


# import re
# import shutil
# import string
# import tensorflow as tf

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
# from tensorflow.keras.layers import TextVectorization


# # In[9]:


a = articles[0]
# print(a)


# In[8]:


def getArticleNEREmbedding(article, article_meta):
    '''
    Get embedding of the article. 
    Use NER to get known entities, get embedding of queries and entiteis from NER.
    Get embeddings of article metadata and article itself. 
    Return the final embedding. 
    
    Input: 
        article: String containing article
        article_meta: List of String containing author, section, publication
        
    Output:
        Array of float containing final embedding of the documents
    '''
    
    debug = False
    
    # Perform NER
    entities = entityPerArticle(article)

    # Get the embeddings of entities from NER
    i, M, N = 0, len(entities.keys()), 96
    if debug: print(M, N)

    embd = np.zeros((M, N))   # Numpy array to store embeddings
    for (item, key) in entities.items():
        query = nlp(item)
        key = nlp(key)

        q2v = query.vector
        k2v = key.vector

        vec = q2v*k2v
        embd[i] = vec

    #     if i == 0:
    #         print(vec.shape)
    #         print(vec)

        i += 1

    avg_embd = np.average(embd, axis=0).flatten()
    if debug: print(avg_embd.shape)
    
#     # Embedding of the article meta
#     i, M = 0, 3
#     embd = np.zeros((M, N))
#     for meta in article_meta:
#         if meta !=None or meta != '':
#             doc = nlp(meta)
#             vec = doc.vector
#             print(meta)
#             embd[i]= vec
#         i += 1
    
#     meta_embd = np.average(embd, axis=0)
    
    # Embedding of the document
#     doc_embd = np.zeros((1, 96))
#     e = nlp(article).vector
#     doc_embd[:e.shape[0]] = e
    doc_embd = nlp(article).vector
    if doc_embd.shape[0]<96:
        print(article)
    if debug: print(doc_embd.shape)

    embd_final = np.concatenate([avg_embd, doc_embd])
    embd_final = embd_final 
    return embd_final


# In[98]:


print(entityPerArticle(a))


# In[10]:


y = nlp(a).vector
x = np.zeros((1, 96))
x[:y.shape[0]] = y


# ### Run embedding process for each article and prepare data for clustering

# In[11]:


p0.fillna('', inplace=True)
p0 = p0[p0['article']!=''].reset_index(drop=True)


# In[37]:


m, n = p0.shape

embd_array = np.zeros((m, 96*2))

p0.fillna('', inplace=True)
k=0
for (i, row) in p0.iterrows():
    if k%100==0: print(f'k = {k}...')
    
#     print(i)
    
    article = row['article']
    author, sec, pub = row['author'], row['section'], row['publication']
    
#     print(i, author, sec, pub)
    
    embd = getArticleNEREmbedding(article, [author, sec, pub])
    if embd.shape[0]<192: print(article)
    
    embd_array[k]=embd
    
    k+=1
#     if k == 500: break


# In[38]:


import pickle

pickle.dump(embd_array, open("chunk0_embd.pkl", "wb"))

print(embd_array)


# In[36]:


# embd_array_1 = pickle.load(open("chunk0_embd.pkl", 'rb'))


# ### Clusterng

# #### Agglomerative Clustering

# In[18]:


import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram


# In[50]:


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# In[40]:


embd_array_1 = np.nan_to_num(embd_array, 0)
embd_array_1[np.isinf(embd_array_1)]=0
embd_array = embd_array_1


# **Ward Linkage**

# In[91]:


model = AgglomerativeClustering(distance_threshold=0.01, n_clusters=None)
model = model.fit(embd_array)


# In[92]:


plt.figure(figsize=(20, 10))
plt.title("Hierarchical Clustering Dendrogram (Ward)")

# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of article in cluster (or index of point if no parenthesis).")
plt.show()


# In[93]:


y = model.labels_

# print(np.unique(y, return_counts=True))
y.shape


# **Complete Linkage**

# In[55]:


embd_array.shape


# In[96]:


model = AgglomerativeClustering(distance_threshold=0.01, linkage='complete', n_clusters=None)
model = model.fit(embd_array)


# In[97]:


plt.figure(figsize=(20, 10))
plt.title("Hierarchical Clustering Dendrogram (Complete)")

# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of article in cluster (or index of point if no parenthesis).")
plt.show()


# In[70]:


y = model.labels_

# print(np.unique(y, return_counts=True))
y.shape


# #### DBScan

# In[71]:


from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# In[78]:


# Compute DBSCAN
db = DBSCAN(eps=0.5, min_samples=3).fit(embd_array)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_


# In[79]:


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
# print(
#     "Adjusted Mutual Information: %0.3f"
#     % metrics.adjusted_mutual_info_score(labels_true, labels)
# )
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))


# In[80]:


# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = embd_array[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = embd_array[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()


# In[ ]:




