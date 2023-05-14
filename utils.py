import contractions
import re
import numpy as np
import distance
import pickle

from wordcloud import STOPWORDS
from fuzzywuzzy import fuzz

STOP_WORDS = STOPWORDS
cv_file=open("cv.pkl" , 'rb')
cv = pickle.load(cv_file)

def preprocess(q):
  q = str(q).lower().strip()
  q = contractions.fix(q) ## correcting the contractions
  q = re.sub(r"https?://\S+|www\.\S+", "", q) ## remove the urls from string
  html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")  ##removing the html tags
  q = re.sub(html, "", q)
  #replace certain special characters
  q = q.replace('%', ' percent')
  q = q.replace('$', ' dollar')
  q = q.replace('₹', ' rupee ')
  q = q.replace('@', ' at')
  q = q.replace('€', ' euro')
  q = q.replace('[math]','')
  q = re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", q) ##puntuation
  return q

def common_words(q1,q2):
  w1 = set(map(lambda word : word.lower().strip(), q1.split(' ')))
  w2 = set(map(lambda word : word.lower().strip(), q2.split(' ')))
  return len(w1&w2)
def total_words(q1,q2):
  w1 = set(map(lambda word : word.lower().strip(), q1.split(' ')))
  w2 = set(map(lambda word : word.lower().strip(), q2.split(' ')))
  return (len(w1) + len(w2))

def fetch_token_features(q1,q2):
  NUM_TOKEN_FEATURE = 8
  SAFE_DIV = 0.0001

  token_features = [0.0]*NUM_TOKEN_FEATURE
  ##converting token to features
  q1_tokens = q1.split()
  q2_tokens = q2.split()

  if len(q1_tokens)==0 or len(q2_tokens)==0:
    return token_features

  ##getting the non stopwords form the questions
  q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
  q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

  ##getting the stop words in the tokens
  q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
  q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

  ##common word count
  common_word_count = len(q1_words.intersection(q2_words))
  ##common stopwords count
  common_stopword_count = len(q1_stops.intersection(q2_stops))
  #common token count
  common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
  # cwc-min - common words/min(words(q1,q2))
  token_features[0] = common_word_count / (min(len(q1_words),len(q2_words)) + SAFE_DIV)
  # cwc-max - common words/max(words(q1,q2))
  token_features[1] = common_word_count / (max(len(q1_words),len(q2_words)) + SAFE_DIV)
  # csc-min - common stopwords/min(words(q1,q2))
  token_features[2] = common_stopword_count / (min(len(q1_stops),len(q2_stops)) + SAFE_DIV)
  # csc-max - common stopwords/max(words(q1,q2))
  token_features[3] = common_stopword_count / (max(len(q1_stops),len(q2_stops)) + SAFE_DIV)
  #ctc-min  common tokens/min(words(q1,q2))
  token_features[4] = common_token_count / (min(len(q1_tokens) , len(q2_tokens)) + SAFE_DIV)
  #ctc-max  common tokens/max(words(q1,q2))
  token_features[5] = common_token_count / (max(len(q1_tokens) , len(q2_tokens)) + SAFE_DIV)
  ##check first word is similar
  token_features[6] = int(q1_tokens[0]==q2_tokens[0])
  ##check last word is similar
  token_features[7] = int(q1_tokens[-1]==q2_tokens[-1])

  return token_features

def fetch_length_features(q1 ,q2):

  length_features = [0.0]*3

  q1_tokens = q1.split()
  q2_tokens = q2.split()

  if len(q1_tokens)==0 or len(q2_tokens)==0:
    return length_features

  ##absolute length features diff
  length_features[0] = abs(len(q1_tokens) - len(q2_tokens))

  ##avg token length of both questions
  length_features[1] = (len(q1_tokens) + len(q2_tokens))/2

  strs = list(distance.lcsubstrings(q1,q2))
  length_features[2] = len(strs[0]) / (min(len(q1),len(q2))+1)

  return length_features

def fetch_fuzzy_feature(q1,q2):
  
  fuzzy_features = [0.0]*4
  #fuzz Qratio
  fuzzy_features[0] = fuzz.QRatio(q1,q2)
  #fuzz partial ratio
  fuzzy_features[1] = fuzz.partial_ratio(q1,q2)
  #token sort ratio
  fuzzy_features[2] = fuzz.token_sort_ratio(q1,q2)
  ##token set ratio
  fuzzy_features[3] = fuzz.token_set_ratio(q1,q2)

  return fuzzy_features

def query_point_creator(q1,q2):
  
  input_query = []
  
  #preprocess
  q1 = preprocess(q1)
  q2 = preprocess(q2)

  #basic features
  input_query.append(len(q1))
  input_query.append(len(q2))

  input_query.append(len(q1.split(' ')))
  input_query.append(len(q2.split(' ')))

  input_query.append(common_words(q1,q2))
  input_query.append(total_words(q1,q2))

  input_query.append(round(common_words(q1,q2)/total_words(q1,q2) ,2))

  ##token features
  token_features = fetch_token_features(q1,q2)
  input_query.extend(token_features)

  ##length featurers
  length_features = fetch_length_features(q1,q2)
  input_query.extend(length_features)

  ##fuzzy features
  fuzzy_features = fetch_fuzzy_feature(q1,q2)
  input_query.extend(fuzzy_features)

  ##cbow features of q1
  q1_bow = cv.transform([q1]).toarray()

  ##cbow features of q2
  q2_bow = cv.transform([q2]).toarray()

  return np.hstack((np.array(input_query).reshape(1,22),q1_bow,q2_bow))

  