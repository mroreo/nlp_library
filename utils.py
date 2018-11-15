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
    

