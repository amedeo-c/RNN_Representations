
import json
import re
import numpy as np
import pandas as pd
import torchtext.data as data
import torch.nn.functional as F

def save_dictionary(dict, path):
    """
    takes a python dictionary object and save it as a json file.
    """
    with open(path, "w") as f:
        js = json.dumps(dict)
        f.write(js)
        f.close()

def cleaner(text):
    text = text.lower()
    text = re.sub(r'@\S+', 'USERNAME', text)
    text = re.sub(r'&\S+', '', text) # emoticons
    text = re.sub(r'\#\S+', 'HASHTAG', text)
    text = re.sub(r'http\S+', '', text) # hyperlinks
    text = re.sub(r'\srt\s', '', text) # retweet tag
    text = re.sub(r'[0-9]+', 'NUMBER', text)

    text = re.sub(r'w/', 'with', text)
    text = re.sub(r'\su\s', 'you', text)
    text = re.sub(r'\sur\s', 'your', text)

    text = re.sub(r'[!\)\(\]\[;|]', '. ', text)
    text = re.sub(r'\.', '. ', text)
    text = re.sub(r'[,:\-_"]', ' ', text)
    text = re.sub(r'\S*[*#/%$£=^ðâ]\S*', '', text) # words containing..
    text = re.sub(r'\s+', ' ', text) # multiple spaces
    text = re.sub(r'\.+', '.', text) # multiple dots
    text = re.sub(r'(\.\s\.\s)+', '. ', text)

    return text

def k_nearest(word, k, embeddings):
    """
    returns the row indexes in embeddings matrix of the k vectors with the highest cosine_similarity with
    the vector for "word" (excluding the index of "word" itself, which clearly has similarity = 1).
    the input 'word' is an index in the dictionary
    """
    distances = []
    for i in range(embeddings.size()[0]):
        distances.append(F.cosine_similarity(embeddings[word], embeddings[i], dim = -1))
    distances = np.array(distances)
    max_indexes = np.argpartition(distances, -k)
    return max_indexes[-(k+1):-1]

def dataset_from_text(text, field):
    """
    takes a single string of text and returns a torchtext dataframe with a single example.
    the field is supposed to be the usual TEXT field.
    """
    ex = data.example.Example()
    setattr(ex, 'text', field.preprocess(text))
    dataset = data.Dataset([ex], [('text', field)])
    return dataset

def extract_tweets():
    """
    we collected two datsets in csv format, each with its own structure and annotation.
    this function return a single string containing all the tweets labeled as "hate speech", "offensive language"
    or "racist/sexist".
    """
    text = ""
    df = pd.read_csv("twitter.data/hate.csv", encoding="latin1") # davidson et al.
    for idx, row in df.iterrows():
        if row['class'] == 0 or row['class'] == 1:
            text += row['tweet']
    df = pd.read_csv("twitter.data/sentiment.csv", encoding="latin1") # twitter sentiment analysis (kaggle)
    for idx, row in df.iterrows():
        if row['label'] == 1:
            text += row['tweet']
    return cleaner(text)
