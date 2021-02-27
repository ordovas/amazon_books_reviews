#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:08:30 2021

@author: ordovas
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import pandas as pd

# Returns a DataFrame with the TF-IDF vectors for each review
# and with the column names matching the feature names
def tfidf(data,min_df=5):
    v = TfidfVectorizer(min_df=min_df)
    x = v.fit_transform(data).toarray()
    return pd.DataFrame(x, columns=v.get_feature_names())


def var_pca(data,comps=np.arange(2,20)):
    variance = []
    for nc in comps:

        pca = PCA(n_components=nc,random_state=1)
        pca.fit(data)
        variance.append(pca.explained_variance_ratio_.sum())
        
    return comps,variance


def var_svc(data,step=50,ini_nc=300):
    variance = []
    t_var=0
    nc=ini_nc
    i=0
    comps=[]
    while t_var < 0.8:
        comps.append(nc+i*step)
        svd = TruncatedSVD(n_components=nc+i*step,random_state=1)
        svd.fit(data)
        t_var = svd.explained_variance_ratio_.sum()
        print(f"Nº comp={nc+i*step}  =>  Var. ratio={t_var}")
        i+=1
        variance.append(t_var)
        
    return comps,variance,svd


