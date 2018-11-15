# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:39:50 2018

@author: bcheung
"""
import pandas as pd
import numpy as np
import nltk
from multiprocessing import Pool

def parallelize_dataframe(df, func,num_partitions=4,num_cores=4):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(processes=num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return(df)

def split_sentences(df,id_col,notes_col):
    """
    Given a dataframe with CLAIM_SK and NOTES_TX, this splits the the NOTES_TX 
    into all the sentences and return a expanded dataframe
    """
    
    claim_sk_list = []
    notes_sent = []
    for i in range(len(df)):
        claim_sk = df[id_col].iloc[i]
        sent_text = nltk.sent_tokenize(df[notes_col].iloc[i].lower())
        [notes_sent.append(s) for s in sent_text]
        [claim_sk_list.append(claim_sk) for s in sent_text]
        
    df = pd.DataFrame({id_col:claim_sk_list,
                       'sentence':notes_sent})
    return(df)
    

