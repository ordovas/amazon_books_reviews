#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:08:30 2021

@author: ordovas
"""

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')

import re

# Function to lemmatize the text of the review
def lemmatizer_text(text):
    # Load Lemmatizer
    lemmatizer=WordNetLemmatizer()
    # Obtain a list with each word
    words = nltk.word_tokenize(text)
    # Returns an string joining the lemmatized words
    return ' '.join([lemmatizer.lemmatize(w) for w in words])

def cleaning_review(texts):
    #Remove numbers, punctuation and lowercase everything
    res=re.sub("[^A-Za-z]+", " ", texts.lower()) 
    #Lemmatize the text of the review
    res=lemmatizer_text(res)
    #Remove rest of stop words
    for stopword in stopwords.words('english'): 
        sw=stopword.replace("'","")
        res=res.replace(f" {sw} "," ")
    return res
