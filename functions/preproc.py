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

def tfidf(data_train, data_test, min_df=5):
    v=TfidfVectorizer(min_df=min_df)
    x_train = v.fit_transform(data_train).toarray()
    x_test = v.transform(data_test).toarray()
    return pd.DataFrame(x_train, columns=v.get_feature_names()),pd.DataFrame(x_test, columns=v.get_feature_names())


def var_pca(data,comps=np.arange(2,20)):
    variance = []
    for nc in comps:

        pca = PCA(n_components=nc,random_state=1)
        pca.fit(data)
        variance.append(pca.explained_variance_ratio_.sum())
        
    return comps,variance


def var_svc(data,step=250,ini_nc=500,var=0.8):
    variance = []
    t_var=0
    nc=ini_nc
    i=0
    comps=[]
    while t_var < var:
        comps.append(nc+i*step)
        svd = TruncatedSVD(n_components=nc+i*step,random_state=1)
        svd.fit(data)
        t_var = svd.explained_variance_ratio_.sum()
        print(f"Nº comp={nc+i*step}  =>  Var. ratio={t_var}")
        i+=1
        variance.append(t_var)
        
    return comps,variance,svd

def svc_dimred(data,comps=500):
   
    svd = TruncatedSVD(n_components=comps,random_state=1)
    svd.fit(data)
    t_var = svd.explained_variance_ratio_.sum()
        
    return comps,t_var,svd
