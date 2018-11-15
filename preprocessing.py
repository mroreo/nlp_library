# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:15:13 2018

@author: bcheung
"""
import pandas as pd
import numpy as np
import re
import json
from html.parser import HTMLParser

from sklearn.base import BaseEstimator, TransformerMixin

class LowerCase(BaseEstimator, TransformerMixin):
    
    def fit(self,X,y=None):
        return(self)
        
    def transform(self,X,y=None):
        if type(X) == np.ndarray or type(X) == list:
            return(list(map(lambda x: x.lower(),X)))
        elif type(X) == str:
            return(X.lower())
        else:
            raise ValueError('Cannot transform the X type. Please check if it is ndarray or str')

class PadSpaces(BaseEstimator, TransformerMixin):
    
    def __init__(self,charPads):
        self.charPads = charPads
        
    def fit(self,X,y=None):
        return(self)
        
    def transform(self,X,y=None):
        if type(X) == np.ndarray or type(X) == list:
            return(list(map(lambda x: re.sub(self.charPads,r' \1 ',x),X)))
        elif type(X) == str:
            return(re.sub(self.charPads,r' \1 ',X))
        else:
            raise ValueError('Cannot transform the X type. Please check if it is ndarray or str')
        

class StrReplace(BaseEstimator, TransformerMixin):
    """
    Removes all characters that are punctuations and numeric
    """
    
    def __init__(self,replace,substitute):
        self.replace=replace
        self.substitute = substitute
        
    def fit(self,X,y=None):
        return(self)
    
    def transform(self,X,y=None):
        if type(X) == np.ndarray or type(X) == list:
            return(list(map(lambda x: re.sub(self.replace,self.substitute,x),X)))
        elif type(X) == str:
            return(re.sub(self.replace,self.substitute,X))
        else:
            raise ValueError('Cannot transform the X type. Please check if it is ndarray or str')
            
class DictReplacement(BaseEstimator, TransformerMixin):
    
    def __init__(self,dict_mapping):
        self.dict_mapping = dict_mapping
        
        self.pattern_dict = {r'\b{}s?\b'.format(k):v for k, v in self.dict_mapping.items()}
        self.pattern_dict = self.dictinvert(self.pattern_dict)
        
    def dictinvert(self,d):
        inv = {}
        for k, v in d.items():
            keys = inv.setdefault(v, [])
            keys.append(k)
            
        for k,v in inv.items():
            inv[k] = '|'.join(v)
        
        return(inv)    
        
    def replace_strings(self,x):
        for w in self.pattern_dict.keys():
            x = re.sub(self.pattern_dict[w],w,x)
        return(x)
    
    def fit(self,X,y=None):
        return(self)
    
    def transform(self,X,y=None):
        if type(X) == np.ndarray or type(X) == list:
            return(list(map(lambda x: self.replace_strings(x),X)))
        elif type(X) == str:
            return(self.replace_strings(X))
        else:
            raise ValueError('Cannot transform the X type. Please check if it is ndarray or str')

class RemoveStopWords(BaseEstimator, TransformerMixin):
    
    def __init__(self,stopwords):
        self.stopwords = stopwords
        
        self.pattern = [r'\b{}s\b'.format(x) for x in self.stopwords]
        self.pattern = '|'.join(self.pattern)
        
    def fit(self,X,y=None):
        return(self)
    
    def transform(self,X,y=None):
        if type(X) == np.ndarray or type(X) == list:
            return(list(map(lambda x: re.sub(self.pattern,'',x),X)))
        elif type(X) == str:
            return(re.sub(self.pattern,'',X))
        else:
            raise ValueError('Cannot transform the X type. Please check if it is ndarray or str')
        
       
class StemWord(BaseEstimator, TransformerMixin):
    """
    Stem the words by removing the past tense, the plural form and the ing
    """
    def __init__(self,words):
        self.words = words
        self.sub_words = {k:self._clean_word(k) for k in self.words}
    
    def _determine_singular(self,word):
        """
        This finds the plural forms of the words and converts them to the singular form.
        
        Parameters
        ----------
        word : string
              the singular word to use as the stem word.
        
        Returns
        ----------
        regex_pattern : regex
              the pattern for the plural version of the word. 
        
        Examples
        ----------
        >>> sentence1 = 'the heaters are not warm enough'
        >>> word1 = 'heater' 
        >>> plural1 = _determine_singular(word1)
        >>> print(re.sub(plural1,word1,sentence1))
        the heater are not warm enough
                
        >>> sentence2 = 'the entries do not make sense'
        >>> word2 = 'entry'
        >>> plural2 = _determine_singular(word2)
        >>> print(re.sub(plural2,word2,sentence2))
        the entry do not make sense
                
        >>> sentence3 = 'the couches have a flower pattern'
        >>> word3 = 'couch'
        >>> plural3 = _determine_singular(word3)
        >>> print(re.sub(plural3,word3,sentence3))
        the couch have a flower pattern
        """
        if word[-1] == 'h':
            return(r'\b{}es?\b'.format(word))
        elif word[-1] == 'y':
            word = word[:-1]
            return(r'\b{}ies?\b'.format(word))
        else:
            return(r'\b{}s?\b'.format(word))
            
    def _determine_present(self,word):
        """
        This finds the past tense form of a word and converts them to the present form.
        
        Parameters
        ----------
        word : string
              the present tense word to use as the stem word.
        
        Returns
        ----------
        regex_pattern : regex
              the pattern for the past tense version of the word. 
        
        Examples
        ----------
        >>> sentence1 = 'the boy and girl are married'
        >>> word1 = 'marry'
        >>> present1 = _determine_present(word1)
        >>> print(re.sub(present1,word1,sentence1))
        the boy and girl are marry
        
        >>> sentence2 = 'they bagged the groceries in the supermarket'
        >>> word2 = 'bag'
        >>> present2 = _determine_present(word2)
        >>> print(re.sub(present2,word2,sentence2))
        they bag the groceries in the supermarket
        """
        if word[-2:] == 'ee':
            return(r'\b{}ed?\b'.format(word))
        elif word[-1] == 'e':
            return(r'\b{}d?\b'.format(word))
        elif word[-1] == 'g':
            word = word + 'g'
            return(r'\b{}ed?\b'.format(word))
        elif word[-1] == 'y':
            word = word[:-1]
            return(r'\b{}ied?\b'.format(word))
        else:
            return(r'\b{}ed?\b'.format(word))
       
    def _determine_no_ing(self,word):
        """
        This finds the words ending with ing and converts them to the original form.
        
        Parameters
        ----------
        word : string
              the word with no ing to use as the stem word.
        
        Returns
        ----------
        regex_pattern : regex
              the pattern for the word ending with ing. 
        
        Examples
        ----------
        >>> sentence1 = 'the boy and girl are cleaning'
        >>> word1 = 'clean'
        >>> no_ing1 = _determine_no_ing(word1)
        >>> print(re.sub(no_ing1,word1,sentence1))
        the boy and girl are clean
        
        >>> sentence2 = 'the baseball player is hitting homeruns'
        >>> word2 = 'hit'
        >>> no_ing2 = _determine_no_ing(word2)
        >>> print(re.sub(no_ing2,word2,sentence2))
        the baseball player is hit homeruns
    
        """
        if word[-1] == 'e':
            word = word[:-1]
            return(r'\b{}ing?\b'.format(word))
        elif word[-1] == 't':
            return(r'\b{}ting?\b'.format(word))
        else:
            return(r'\b{}ing?\b'.format(word))
    
    def _clean_word(self,word):
        """
        This cleans the word by remove the past tense, the plural form and the ing from a stem word.
        """
        present_word = self._determine_present(word)
        singular_word =  self._determine_singular(word)
        no_ing_word = self._determine_no_ing(word)
        return('{}|{}|{}'.format(present_word,singular_word,no_ing_word))
    
    def replace_strings(self,x):
        for w in self.sub_words.keys():
            x = re.sub(self.sub_words[w],w,x)
        return(x)
    
    def fit(self,X,y=None):
        return(self)
    
    def transform(self,X,y=None):
        if type(X) == np.ndarray or type(X) == list:
            return(list(map(lambda x: self.replace_strings(x),X)))
        elif type(X) == str:
            return(self.replace_strings(X))
        else:
            raise ValueError('Cannot transform the X type. Please check if it is ndarray or str')
            

class MLStripper(HTMLParser):
    
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
        
    def handle_data(self, d):
        self.fed.append(d)
        
    def get_data(self):
        return ''.join(self.fed)
    
class StripHTMLTags(BaseEstimator, TransformerMixin):
    
    def __init__(self,mlstripper_obj = MLStripper()):
        self.mlstripper_obj = mlstripper_obj
    
    def _add_space(self,st):
        tt=st[:]
        tt=re.sub(':',' : ',tt)
        tt=re.sub('\.',' . ',tt)
        tt=re.sub(',',' , ',tt)
        tt=re.sub('\n','  ',tt)
        tt=re.sub('\r','  ',tt)
        tt=re.sub('\b','  ',tt)
        return(tt)
        
    def _remove_tags(self,X):
        self.mlstripper_obj.feed(X)
        X = self.mlstripper_obj.get_data()
        return(self.add_space(X))
    
    def fit(self,X,y=None):
        return(self)
    
    def transform(self,X,y=None):
        if type(X) == np.ndarray or type(X) == list:
            return(list(map(lambda x: self._remove_tags(x),X)))
        elif type(X) == str:
            return(self._remove_tags(X))
        else:
            raise ValueError('Cannot transform the X type. Please check if it is ndarray or str')
        
def apply_prep_pipe(df,pipe,column):
    df[column] = pipe.fit_transform(df[column].values)
    return(df)