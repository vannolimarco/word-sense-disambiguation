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
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import backend as keras
import keras as k
import tensorflow as tf
import tensorflow.keras as K
from nltk.corpus import wordnet
import numpy as np
import torch
from keras.models import load_model
import sys
from sklearn.model_selection import train_test_split
import pickle
from keras import optimizers
import matplotlib.pyplot as plt
import math
import preprocessing
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
import pathconfig
import nltk


"""README:
 !(they are slightly slow in the process. It takes a little more time for large data such as senseval2 and senseval3)
 The fucntions were used with paths class which allow us to call its property to refer all paths available:
 For examples:
 code:

 if want to use the class pathconfig: add some path in the class and then call it in the following way on predict functions:
 
 #!!!remember that all the paths that are in the pathconfig class have been set following the paths of the Raganato framework!!!

 -paths = pathconfig.paths()
 -predict_babelnet(paths.SEMEVAL2007_PATH, paths.PREDICT_2007_BABELNET_PATH, '..\\resources')
 -predict_wordnet_domains(paths.SEMEVAL2007_PATH, paths.PREDICT_2007_DOMAINS_PATH, '..\\resources')
 -predict_lexicographer(paths.SEMEVAL2007_PATH, paths.PREDICT_2007_LEXNAMES_PATH, '..\\resources')
 
 -predict_babelnet(paths.SEMEVAL2013_PATH, paths.PREDICT_2013_BABELNET_PATH, '..\\resources')
 -predict_wordnet_domains(paths.SEMEVAL2013_PATH, paths.PREDICT_2013_DOMAINS_PATH, '..\\resources')
 -predict_lexicographer(paths.SEMEVAL2013_PATH, paths.PREDICT_2013_LEXNAMES_PATH, '..\\resources')
 -...

 or add only the paths as parameter:

 -predict_babelnet('..\\path_input', '..\\path_output', '..\\resources')
 -predict_wordnet_domains('..\\path_input', '..\\path_output', '..\\resources')
 -predict_lexicographer('..\\path_input', '..\\path_output', '..\\resources')
"""

def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    path_model = resources_path + '\\model.json'                                  # path of model
    path_weights = resources_path + '\\weights.h5'                                # path of weights
    path_model_word_embeddings = resources_path + '\\model_word_embeddings.json'  # path for modeof word embeddings
    path_Mapping_babelnet2wordnet = resources_path + '\\babelnet2wordnet.tsv'     # path of mapping bn-wn provided

    training = preprocessing.Preprocessing.TrainingData()      # create an instance from sub class Trainingdata
    prediction = preprocessing.Preprocessing.PredictionData()  # create an instance from sub class Predictiondata
    corpus = preprocessing.Preprocessing.Corpus()              # create an instance from sub class Corpus

    LOSS_LIST = ["sparse_categorical_crossentropy", 'sparse_categorical_crossentropy',
                 'sparse_categorical_crossentropy']                                     # the loss list that is necessary to pass to the model

    METRICS = {'babelnet': 'accuracy', 'domains': 'accuracy', 'lexnames': 'accuracy'}   # the accuracy list that is necessary to pass to the model

    input_x = 'input_x'    #tag input
    babelnet = 'babelnet'  #tag babelnet task

    corpus, vocabolary_WSD = corpus.get_corpus_for_prediction(path_xml_par=input_path,
                                                              return_dict_WSDid_lemma=True)  # create corpus from input path
                                                                                             # and return a dict of WSD id and lemmas
                                                                                             # The vocabulary is created in order to obtain
                                                                                             # all needs informations from it:
                                                                                             # lemma%POS:pos%ID:WSD's id

    input_data, vocabolary_word_embeddings = prediction.get_input_x(corpus=corpus, type_task=input_x,
                                                                    path_mapping_for_vocab=path_model_word_embeddings) # create the data for preidciton from corpus
                                                                                                                       # return all vocabulary for input( word embeddings
    vocabolary_babelnet = training.get_single_vocabolary(path_mapping_task=path_Mapping_babelnet2wordnet,
                                                         type_vocab_par=babelnet)  # create vocabulary for the output task, in this case babelnet
    json_file = open(path_model, 'r')                           # open file of model
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = K.models.model_from_json(loaded_model_json)  # load the model
    loaded_model.load_weights(path_weights)                     # load the weights
    loaded_model.compile(loss=LOSS_LIST, optimizer=K.optimizers.Adam(lr=0.001, epsilon=None, decay=0.0, amsgrad=False),
                         metrics=METRICS)  # compile the model
    predictions = loaded_model.predict([input_data], verbose=2)  # does the precition. All probability tensor are obtained from prediction
    tensor_babelnet = predictions[0]                             # take only the tensor about babelnet output
    print('Start prediction for %s' % babelnet)
    with open(output_path, 'a') as file_predict:                 # open output file
        for sentence in corpus:                                  # for each sentence of corpus
            for word in sentence[:20]:                           # for each word, until padding value (20)
                if word in vocabolary_WSD:                       # if word is a WSD word then
                    lemma = word[:word.index('%POS:')]                       # obtain the lemma
                    pos = word[word.index('%POS:') + 5:word.index('%ID:')]   # obtaine the pos
                    id = vocabolary_WSD[word]                                # obtain the id of WSD
                    list_prob_index = []
                    list_index_candidate, list_babelnet_candidates = prediction.get_list_candidates(type_task=babelnet,
                                                                                                    lemma_par=lemma,
                                                                                                    pos_par=pos,
                                                                                                    path_mapping_bn2wn=path_Mapping_babelnet2wordnet,
                                                                                                    vocabolary_lemma=vocabolary_word_embeddings,
                                                                                                    vocabolary_task=vocabolary_babelnet)     # get the list of candidates for eah WSD word and its list of index
                    for index in list_index_candidate:                                              # for each index on list of index
                        prob = tensor_babelnet[corpus.index(sentence)][sentence.index(word)][index] # take from tensor each probability
                        list_prob_index.append(prob)                                                # add to list each probability
                    file_predict.write("%s %s\n" % (id, list_babelnet_candidates[np.argmax(list_prob_index)]))  # write on file
                                                                                                                # the babelnet synset most likely
                    file_predict.flush()
        file_predict.close()
        print('Predict file for %s was saved to the file : %s' % (babelnet, output_path))
    pass


def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    path_model = resources_path + '\\model.json'                                  # path of model
    path_weights = resources_path + '\\weights.h5'                                # path of weights
    path_model_word_embeddings = resources_path + '\\model_word_embeddings.json'  # path for modeof word embeddings
    path_Mapping_babelnet2wordnet = resources_path + '\\babelnet2wordnet.tsv'     # path of mapping bn-wn provided
    path_Mapping_babelnet2domains = resources_path + '\\babelnet2wndomains.tsv'   # path of mapping bn-dm provided

    training = preprocessing.Preprocessing.TrainingData()         # create an instance from sub class Trainingdata
    prediction = preprocessing.Preprocessing.PredictionData()     # create an instance from sub class Predictiondata
    corpus = preprocessing.Preprocessing.Corpus()                 # create an instance from sub class Corpus

    LOSS_LIST = ["sparse_categorical_crossentropy", 'sparse_categorical_crossentropy',
                 'sparse_categorical_crossentropy']                                   # the loss list that is necessary to pass to the model

    METRICS = {'babelnet': 'accuracy', 'domains': 'accuracy', 'lexnames': 'accuracy'} # the accuracy list that is necessary to pass to the model

    input_x = 'input_x'  # tag input
    domains = 'domains'  # tag babelnet task


    corpus, vocabolary_WSD = corpus.get_corpus_for_prediction(path_xml_par=input_path,
                                                              return_dict_WSDid_lemma=True)  # create corpus from input path
                                                                                             # and return a dict of WSD id and lemmas
                                                                                             # The vocabulary is created in order to obtain
                                                                                             # all needs informations from it:
                                                                                             # lemma%POS:pos%ID:WSD's id
    input_data, vocabolary_word_embeddings = prediction.get_input_x(corpus=corpus, type_task=input_x,
                                                                    path_mapping_for_vocab=path_model_word_embeddings)
                                                                                             # create the data for prediction from corpus
                                                                                             # return all vocabulary for input( word embeddings
    vocabolary_domains = training.get_single_vocabolary(path_mapping_task=path_Mapping_babelnet2domains,
                                                        type_vocab_par=domains)            # create vocabulary for the output task, in this case domains
    json_file = open(path_model, 'r')                          # open file of model
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = K.models.model_from_json(loaded_model_json)  # load the model
    loaded_model.load_weights(path_weights)                     # load the weights
    loaded_model.compile(loss=LOSS_LIST, optimizer=K.optimizers.Adam(lr=0.001, epsilon=None, decay=0.0, amsgrad=False),
                         metrics=METRICS)                       # compile the model
    predictions = loaded_model.predict([input_data],
                                       verbose=2)               # does the precition. All probability tensor are obtained from prediction
    tensor_domains = predictions[1]                             # take only the tensor about domain output

    print('Start prediction for %s' % domains)
    with open(output_path, 'a') as file_predict:                # open output file
        for sentence in corpus:                                 # for each sentence of corpus
            for word in sentence[:20]:                          # for each word, until padding value (20)
                if word in vocabolary_WSD:                      # if word is a WSD word then
                    lemma = word[:word.index('%POS:')]                     # obtain the lemma
                    pos = word[word.index('%POS:') + 5:word.index('%ID:')] # obtain the pos
                    id = word[word.index('%ID:') + 4:]                     # obtain the WSD'id
                    list_prob_index = []
                    list_index_candidate, list_domains_candidates = prediction.get_list_candidates(type_task=domains,
                                                                       lemma_par=lemma,
                                                                       pos_par=pos,
                                                                       path_mapping_bn2wn=path_Mapping_babelnet2wordnet,
                                                                       path_mapping_task=path_Mapping_babelnet2domains,
                                                                       vocabolary_lemma=vocabolary_word_embeddings,
                                                                       vocabolary_task=vocabolary_domains) # get the list of candidates for eah WSD word and its list of index
                    for index in list_index_candidate:                                             # for each index on list of index
                        prob = tensor_domains[corpus.index(sentence)][sentence.index(word)][index] # take from tensor each probability
                        list_prob_index.append(prob)                                               # add to list each probability
                    domains_candidates = list_domains_candidates[np.argmax(list_prob_index)]
                    file_predict.write("%s %s\n" % (id, "\t".join(domains_candidates.split('/')))) # write on file the domain synset most likely (the domains is considered in its hierarchy)
                    file_predict.flush()
    file_predict.close()
    print('Predict file for %s was saved to the file : %s' % (domains, output_path))
    pass


def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    path_model = resources_path + '\\model.json'                                 # path of model
    path_weights = resources_path + '\\weights.h5'                               # path of weights
    path_model_word_embeddings = resources_path + '\\model_word_embeddings.json' # path for modeof word embeddings
    path_Mapping_babelnet2wordnet = resources_path + '\\babelnet2wordnet.tsv'    # path of mapping bn-wn provided
    path_Mapping_babelnet2lexnames = resources_path + '\\babelnet2lexnames.tsv'  # path of mapping bn-lex provided

    training = preprocessing.Preprocessing.TrainingData()          # create an instance from sub class Trainingdata
    prediction = preprocessing.Preprocessing.PredictionData()      # create an instance from sub class Predictiondata
    corpus = preprocessing.Preprocessing.Corpus()                  # create an instance from sub class Corpus

    LOSS_LIST = ["sparse_categorical_crossentropy", 'sparse_categorical_crossentropy',
                 'sparse_categorical_crossentropy']                                    # the loss list that is necessary to pass to the model

    METRICS = {'babelnet': 'accuracy', 'domains': 'accuracy', 'lexnames': 'accuracy'}  # the accuracy list that is necessary to pass to the model

    input_x = 'input_x'    # tag input
    lexnames = 'lexnames'  # tag lexnames task

    corpus, vocabolary_WSD = corpus.get_corpus_for_prediction(path_xml_par=input_path,
                                                              return_dict_WSDid_lemma=True)  # create corpus from input path
                                                                                             # and return a dict of WSD id and lemmas
                                                                                             # The vocabulary is created in order to obtain
                                                                                             # all needs informations from it:
                                                                                             # lemma%POS:pos%ID:WSD's id
    input_data, vocabolary_word_embeddings = prediction.get_input_x(corpus=corpus,
                                                                    type_task=input_x,
                                                                    path_mapping_for_vocab=path_model_word_embeddings)# create the data for prediction from corpus
                                                                                                                      # return all vocabulary for input( word embeddings
    vocabolary_lexnames = training.get_single_vocabolary(path_mapping_task=path_Mapping_babelnet2lexnames,
                                                         type_vocab_par=lexnames)           # create vocabulary for the output task, in this case lexnames
    json_file = open(path_model, 'r')                           # open file of model
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = K.models.model_from_json(loaded_model_json)  # load the model
    loaded_model.load_weights(path_weights)                     # load the weights
    loaded_model.compile(loss=LOSS_LIST, optimizer=K.optimizers.Adam(lr=0.001, epsilon=None, decay=0.0, amsgrad=False),
                         metrics=METRICS)                       # compile the model
    predictions = loaded_model.predict([input_data],
                                       verbose=2)  # does the precition. All probability tensor are obtained from prediction
    tensor_lexnames = predictions[2]               # take only the tensor about lexnames output
    print('Start prediction for %s' % lexnames)
    with open(output_path, 'a') as file_predict:      # open output file
        for sentence in corpus:                       # for each sentence of corpus
            for word in sentence[:20]:                # for each word, until padding value (20)
                if word in vocabolary_WSD:            # if word is a WSD word then
                    lemma = word[:word.index('%POS:')]                      # obtain the lemma
                    pos = word[word.index('%POS:') + 5:word.index('%ID:')]  # obtain the pos
                    id = word[word.index('%ID:') + 4:]                      # obtain the WSD's id
                    list_prob_index = []
                    list_index_candidate, list_lexnames_candidates = prediction.get_list_candidates(type_task=lexnames,
                                                                        lemma_par=lemma,
                                                                        pos_par=pos,
                                                                        path_mapping_bn2wn=path_Mapping_babelnet2wordnet,
                                                                        path_mapping_task=path_Mapping_babelnet2lexnames,
                                                                        vocabolary_lemma=vocabolary_word_embeddings,
                                                                        vocabolary_task=vocabolary_lexnames) # get the list of candidates for eah WSD word and its list of index
                    for index in list_index_candidate:                                              # for each index on list of index
                        prob = tensor_lexnames[corpus.index(sentence)][sentence.index(word)][index] # take from tensor each probability
                        list_prob_index.append(prob)                                                # add to list each probability
                    file_predict.write("%s %s\n" % (id, list_lexnames_candidates[np.argmax(list_prob_index)])) # write on file the domain synset most likely
                    file_predict.flush()
    file_predict.close()
    print('Predict file for %s was saved to the file : %s' % (lexnames, output_path))
    pass

