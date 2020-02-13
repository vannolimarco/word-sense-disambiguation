import preprocessing
from lxml import etree
import io
import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import re
import csv
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
import  tensorflow as tf
import numpy as np
import nltk
from keras import backend as keras
import keras as k
import tensorflow as tf
import tensorflow.keras as K
from nltk.corpus import wordnet
import numpy as np
import torch
from keras.models import load_model
from sklearn.model_selection import train_test_split
import sys
from sklearn.model_selection import train_test_split
import pickle
from keras import optimizers
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import math
import io
import os
from gensim.models import Word2Vec
import preprocessing
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
import pathconfig
import nltk

########################################################################################
###################### RUN FOR CREATE GOLD DATA FOR EVALUATION  ########################
########################################################################################

""" in this file py, is possible run all preprocessing in order to elaborate data for the evaluation.
    It use the class Preprocessing and its relative sub-class."""

paths = pathconfig.paths()

Preprocessing = preprocessing.Preprocessing()

#all type of task which can be called from methods
babelnet_task = 'babelnet'
domains_taks = 'domains'
lexnames_task = 'lexnames'

# create Gold file for semval2013
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=babelnet_task,path_gold_key_par=paths.SEMEVAL2013_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_2013_GOLD_BABELNET_PATH)
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=domains_taks,path_gold_key_par=paths.SEMEVAL2013_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_2013_GOLD_DOMAINS_PATH,path_mapping_task=paths.MAPPING_BN2WNDOM_PATH)
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=lexnames_task,path_gold_key_par=paths.SEMEVAL2013_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_2013_GOLD_LEXNAMES_PATH,path_mapping_task=paths.MAPPING_BN2LEX_PATH)

# create Gold file for semval2007
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=babelnet_task,path_gold_key_par=paths.SEMEVAL2007_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_2007_GOLD_BABELNET_PATH)
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=domains_taks,path_gold_key_par=paths.SEMEVAL2007_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_2007_GOLD_DOMAINS_PATH,path_mapping_task=paths.MAPPING_BN2WNDOM_PATH)
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=lexnames_task,path_gold_key_par=paths.SEMEVAL2007_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_2007_GOLD_LEXNAMES_PATH,path_mapping_task=paths.MAPPING_BN2LEX_PATH)

#create Gold file for semval2015
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=babelnet_task,path_gold_key_par=paths.SEMEVAL2015_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_2015_GOLD_BABELNET_PATH)
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=domains_taks,path_gold_key_par=paths.SEMEVAL2015_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_2015_GOLD_DOMAINS_PATH,path_mapping_task=paths.MAPPING_BN2WNDOM_PATH)
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=lexnames_task,path_gold_key_par=paths.SEMEVAL2015_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_2015_GOLD_LEXNAMES_PATH,path_mapping_task=paths.MAPPING_BN2LEX_PATH)

#create Gold file for semval2
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=babelnet_task,path_gold_key_par=paths.SENSEVAL2_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_2_GOLD_BABELNET_PATH)
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=domains_taks,path_gold_key_par=paths.SENSEVAL2_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_2_GOLD_DOMAINS_PATH,path_mapping_task=paths.MAPPING_BN2WNDOM_PATH)
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=lexnames_task,path_gold_key_par=paths.SENSEVAL2_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_2_GOLD_LEXNAMES_PATH,path_mapping_task=paths.MAPPING_BN2LEX_PATH)

#create Gold file for semval3
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=babelnet_task,path_gold_key_par=paths.SENSEVAL3_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_3_GOLD_BABELNET_PATH)
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=domains_taks,path_gold_key_par=paths.SENSEVAL3_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_3_GOLD_DOMAINS_PATH,path_mapping_task=paths.MAPPING_BN2WNDOM_PATH)
Preprocessing.EvaluationData.save_gold_key_true(self=None,type_task_par=lexnames_task,path_gold_key_par=paths.SENSEVAL3_GOLD_KEY_PATH,path_save_evaluation=paths.EVALUATION_3_GOLD_LEXNAMES_PATH,path_mapping_task=paths.MAPPING_BN2LEX_PATH)