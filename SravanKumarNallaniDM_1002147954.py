#!/usr/bin/env python
# coding: utf-8

# In[1]:


## necessary packages
import nltk,math
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
from nltk.corpus import stopwords
stpwords = set(stopwords.words('english'))
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from math import log10,sqrt
import collections
from collections import Counter
import os
## operating system & Text documents
corpus = "C:/Users/gjwpr/Desktop/dmpr/US_Inaugural_Addresses"
## settings and dependencies
vectors = {}                             # all documents tf-idf vectors
df = Counter()                           # document frequency storage
tfs = {}                                 # permanent storage for tfs of all tokens in all documents
lengths = Counter()                      # used to calculate lenght of documents
postingslist = {}                        # posting list storage for each token in the corp
for filename in os.listdir(corpus):
    if filename.startswith('0') or filename.startswith('1') or filename.startswith('2') or filename.startswith('3'):
        with open(os.path.join(corpus, filename), "r", encoding='windows-1252') as file:
            doc = file.read().lower()
        tokens = tokenizer.tokenize(doc)
        tokens = [stemmer.stem(token) for token in tokens if token not in stpwords]
        tf = Counter(tokens)
        df += Counter(set(tokens))
        tfs[filename] = tf.copy()

### document frequencies
m = len(tfs)                            # # number of documents

# normalize the tf-idf score
def cos_normalize(filename, tf_idf_score):
    norm_sum = 0

    for token in tfs[filename]:
        idf = getidf(token, False)
        t_freq = tfs[filename][token]
        tf = 1 + math.log10(t_freq)
        norm_sum += (idf * tf)**2
    det = math.sqrt(norm_sum)
    return tf_idf_score/det


def getidf(token, stem_need = True):
    if stem_need:
        stem_word = stemmer.stem(token)
    else:
        stem_word = token

    if df[stem_word] == 0:                    #df[token] document token's frequency
        return -1

    return math.log10(m / df[stem_word])

# weight & log of term frequencies
def getweight(filename, token,stem_need = True):
    if stem_need:
        stem_word = stemmer.stem(token)
    else:
        stem_word = token
    #not normalized document's token weights
    idf = getidf(stem_word, False)
    tf = tfs[filename][stem_word]
    #print(tf)
    if tf == 0:
        return 0   #return 0 if token doesn't exist in document

    tf_idf_score = (1 + math.log10(tfs[filename][stem_word])) * idf
    #print('tf_idf_score', tf_idf_score)
    tf_idf_norm = cos_normalize(filename, tf_idf_score)
    #print('tf_idf_norm', tf_idf_norm)
    return tf_idf_norm  #multiple level dict logs of term frequencies for all documents


def query(qstring):
    qstring = qstring.lower()
    qtokens = [stemmer.stem(token) for token in tokenizer.tokenize(qstring)]
    qtf = Counter(qtokens)
    qlength = sqrt(sum((1 + math.log10(count))**2 for count in qtf.values()))
    best_match = (None, 0)

    for filename, word in tfs.items():
        cos_sim = sum(((1 + math.log10(qtf[token])) / qlength) * getweight(filename, token, False) for token in qtokens)

        if cos_sim > best_match[1]:
            best_match = (filename, cos_sim)

    if best_match[1] == 0:
        return 0

    return best_match

# test
print("%.12f" % getidf('children'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('people'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
print("%.12f" % getweight('23_hayes_1877.txt','public'))
print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
print("%.12f" % getweight('05_jefferson_1805.txt','press'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("cuba government"))


# In[ ]:




