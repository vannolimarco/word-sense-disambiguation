from lxml import etree
import io
import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import re
import csv
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import WordNetError
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pickle
import numpy as np
from collections import OrderedDict
from operator import itemgetter
from keras.preprocessing.text import Tokenizer
import  tensorflow as tf
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
import pathconfig

###############################################################
##################  GLOBAL VARIABLES ##########################
###############################################################

paths = pathconfig.paths()   #create instance from class paths in the file pathconfig.py

######################################################################
###################### CLASS FOR PREPROCESSING  ######################
######################################################################

class Preprocessing(object):
    """ This is a class for Preprocessing phase. It expose a several methods used for preprocessing,
    preprocessing for prediction and evaluation. It is called wherever there is the necessary to
    run some method for data preprocessing,  data prediction and evaluation.
    """
    def synset_from_sense_key(self,sense_key_par: str):
        """ this method is important to prelevate from synset key the correspond wordent senset.
        Take the sense key as parameter and return the wordnet synset"""
        sense_key_regex = r"(.*)\%(.*):(.*):(.*):(.*):(.*)"
        synset_types = {1: 'n', 2: 'v', 3: 'a', 4: 'r', 5: 's'}
        lemma, ss_type, lex_num, lex_id, head_word, head_id = re.match(sense_key_regex, sense_key_par).groups()
        ss_idx = '.'.join([lemma, synset_types[int(ss_type)], lex_id])
        return wn.synset(ss_idx)

    def get_sentence_tokenize_from_path_corpus(self,path_corpus_par: str):
        """ this method is used to return each sentence tokenize from a corpus given as parameter. The coprus in this case
        is structurated such that each sentence are broken if it goes beyond the twentieth word"""
        with open(path_corpus_par, 'r', encoding='utf-8') as file:
            corpus = file.readlines()
            sentences = []
            for sentence in corpus:
                sentences.append(sentence.replace('\n', ''))
            return sentences

    class Mapping(object):
          """ This is a sub-class for mapping phase. It expose a several methods used to return
          all data provided by mapping file.
          """
          def getDictAllIdKey(self,data_gold_key_path: str):
                  """ This method is called whenever is necessary to return the mapping between sensey key and
                  WSD's id from file data.gold.key. In order to used it, is called giving it the path of file in
                  which there are the mapping amoung sensey key and WSD's id.
                  """
                  with open(data_gold_key_path, 'r', encoding='utf-8') as file:
                      list_id_synset = file.readlines()
                      list_id = []
                      list_key = []
                      for id_key in list_id_synset:  # iterates in all lines of file bn2wn_mapping.txt
                          list_id.append(id_key[:14])  # takes only the babelnet synset
                          list_key.append(id_key[15:].replace('\n',''))
                      dictionary_id_synset = dict(zip(list_id, list_key))  # create dict with babelnet synset as key and number enumerate as value
                      return dictionary_id_synset  # return dictionary

          def getDictAllBnWn(self, mapping_bn2wn_path:str):
              """ This method is called whenever is necessary to return the mapping between babelnet and
                  wordnet synsets from file tsv. In order to used it, is called giving it the path of file in
                  which there are the mapping amoung babelnet and wordnet synsets.
              """
              with open(mapping_bn2wn_path) as tsvfile:
                  tsvreader = csv.reader(tsvfile, delimiter="\t")
                  list_bn = []
                  list_wn = []
                  for bn, wn in tsvreader:
                      list_bn.append(bn)
                      list_wn.append(wn)
                  dictionary_wn_bn = dict(zip(list_wn, list_bn))
                  return dictionary_wn_bn

          def getDictAllBnLex(self, mapping_bn2lex_path:str):
              """ This method is called whenever is necessary to return the mapping between babelnet and
                  lexnames from file tsv. In order to used it, is called giving it the path of file in
                  which there are the mapping amoung babelnet and lexnames.
               """

              with open(mapping_bn2lex_path) as tsvfile:
                  tsvreader = csv.reader(tsvfile, delimiter="\t")
                  list_bn = []
                  list_lex = []
                  for bn, lex in tsvreader:
                      list_bn.append(bn)
                      list_lex.append(lex)
                  dictionary_wn_lex = dict(zip(list_bn, list_lex))
                  return dictionary_wn_lex

          def getDictAllBnDomain(self, mapping_bn2dom_path):
                  """ This method is called whenever is necessary to return the mapping between babelnet and
                       domains from file tsv. In order to used it, is called giving it the path of file in
                       which there are the mapping amoung babelnet and domains. The domains whose value is composed
                       hierarchically, are splitted by '\' in order to consider it completely.
                       Example: (\t -> coloum of file tsv)
                           domain -> bn:00000019n religion \t time_period ->
                           from method -> key = bn:00000019n value = religion\time_period
                 """
                  with open(mapping_bn2dom_path) as tsvfile:
                      tsvreader = csv.reader(tsvfile, delimiter="\t")
                      list_bn = []
                      list_domains = []
                      for row in tsvreader:
                          list_row = row
                          list_bn.append(list_row[0])
                          list_domains.append("/".join(map(str, list_row[1:] )))  #add '\' in the hierarchical domains
                      dictionary_wn_domains = dict(zip(list_bn, list_domains)) #create dictionary from values obtained
                      return dictionary_wn_domains


    class Corpus(object):
        """ This is a sub-class for the creation and saving of corpus for preprocessing/prediction. It exposes a several methods used
            to create, save and return all type of corpus utiled to create training sets and test sets.
        """

        def create_and_save_corpus_x(self, path_xml_par: str, path_save_corpus_par: str, break_sentence = 20):
            """ This method is called whenever is necessary to create and save corpus for input training data in
            the path file given as parameter. the method needs to take three parameters: one is the path of file Raganato
            in xml format, one for the path where will save the corpus and finally one inter parameter that tells to the method
            where each sentence of corpus has to broke if it goes beyond the "break_sentence" word. This last solution
            wis adopted to avoid the issue about missing WSd words. The corpus is created following this rules:
            - for each word no WSD (wf notation), its value string is taken
            - for each word WSD, its lemma is taken
            - if sentence goes beyond twentieth word (default), it must broken. Otherwise it is broken when finish sentence
            (\sentence notation)
            """

            xml_file = etree.iterparse(path_xml_par)
            print('#####################################')
            print('Beginning of the creating of the corpus for data x from the file xml : %s' % path_xml_par)
            cont_sentence_padding = 0
            for event, element in xml_file:
                with open(path_save_corpus_par, "a", encoding='utf-8') as file_corpus:
                    list_item = element.items()
                    try:
                        if (element.tag == "wf"):
                            if (cont_sentence_padding == break_sentence - 1):
                                if (list_item[1][1] != '.'):
                                    word = str(element.text) + '\n'
                                    file_corpus.write(word)
                                    element.clear()
                                    cont_sentence_padding = 0
                                else:
                                    file_corpus.write('\n')
                                    element.clear()
                                    cont_sentence_padding = 0
                            else:
                                if (list_item[1][1] != '.'):
                                    word = str(element.text) + ' '
                                    file_corpus.write(word)
                                    element.clear()
                                    cont_sentence_padding += 1
                                else:
                                    element.clear()
                        if(element.tag == "instance"):
                            if (cont_sentence_padding == break_sentence-1):
                                word = str(list_item[1][1]) + '\n'
                                file_corpus.write(word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                                element.clear()
                                cont_sentence_padding = 0
                            else:
                                word = str(list_item[1][1]) + ' '
                                file_corpus.write(word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                                element.clear()
                                cont_sentence_padding += 1
                        if (element.tag == "sentence"):
                            if(cont_sentence_padding == 0):
                                file_corpus.close()
                                element.clear()
                            else:
                                file_corpus.write('\n')
                                cont_sentence_padding = 0
                                file_corpus.close()
                                element.clear()
                    except Exception as e:
                        print(e)
            print('End of the creating of the corpus..')
            print('The corpus was saved in the file : %s' % path_save_corpus_par)

        def create_and_save_corpus_babelnet_y(self, path_xml_par: str,path_gold_key_par: str, path_save_corpus_par: str, break_sentence=20)-> None:
            """ This method is called whenever is necessary to create and save corpus for training sets about babelnet (fine-grained) in
                the path file given as parameter. the method needs to take three parameters: one is the path of file Raganato
                in xml format, one for the path where will save the corpus and finally one inter parameter that tells to the method
                where each sentence of corpus has to broke if it goes beyond the "break_sentence" word. This last solution
                wis adopted to avoid the issue about missing WSd words. The corpus is created following this rules:
                - for each word no WSD (wf notation), its value string is taken,
                - for each word WSD, its lemma is taken; once the wordnet sysnet is computed,
                then the babelnet synset is taken from mapping provided (wn-bn)
                - if sentence goes beyond twentieth word (default), it must broken. Otherwise it is broken when finish sentence
                (\sentence notation)
            """

            xml_file = etree.iterparse(path_xml_par)
            print('#####################################')
            print('Beginning of the creating of the corpus labels babelnet from the file xml : %s' % path_xml_par)
            mapping = Preprocessing.Mapping()
            dict_id_key = mapping.getDictAllIdKey(path_gold_key_par)
            dict_bn_wn = mapping.getDictAllBnWn(mapping_bn2wn_path=paths.MAPPING_BN2WN_PATH)
            cont_sentence_padding = 0
            for event, element in xml_file:
                with open(path_save_corpus_par, "a", encoding='utf-8') as file_corpus:
                    list_item = element.items()
                    try:
                        if (element.tag == "wf"):   #no WSD word
                            if (cont_sentence_padding == break_sentence - 1):
                                if (list_item[1][1] != '.'):
                                    word = str(element.text) + '\n'
                                    file_corpus.write(word)
                                    element.clear()
                                    cont_sentence_padding = 0
                                else:
                                    file_corpus.write('\n')
                                    element.clear()
                                    cont_sentence_padding = 0
                            else:
                                if (list_item[1][1] != '.'):
                                    word = str(element.text) + ' '
                                    file_corpus.write(word)
                                    element.clear()
                                    cont_sentence_padding += 1
                                else:
                                    element.clear()
                        if (element.tag == "instance"):  #WSD word
                            word_from_id = dict_id_key[list_item[0][1]] #lemma
                            if ' ' in word_from_id:                # if there are two synset keys
                                list_wn_id = word_from_id.split()  # split them
                                synset = Preprocessing.synset_from_sense_key(self,list_wn_id[0]) # take only the first
                                synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos() # compute its wn synset
                            else:                                  # else takes the only one
                                synset = Preprocessing.synset_from_sense_key(self,word_from_id)
                                synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
                            bn_synset = dict_bn_wn[synset_id]
                            if (cont_sentence_padding == break_sentence - 1):
                                wn_word = str(bn_synset) + '\n'
                                file_corpus.write(wn_word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                                element.clear()
                                cont_sentence_padding = 0
                            else:
                                wn_word = str(bn_synset) + ' '
                                file_corpus.write(wn_word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                                element.clear()
                                cont_sentence_padding += 1
                        if(element.tag == "sentence"):
                            if (cont_sentence_padding == 0):
                                file_corpus.close()
                                element.clear()
                            else:
                                file_corpus.write('\n')
                                cont_sentence_padding = 0
                                file_corpus.close()
                                element.clear()
                    except WordNetError:
                        if (cont_sentence_padding == break_sentence - 1):
                            word = str(list_item[1][1]) + '\n'
                            file_corpus.write(word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                            element.clear()
                            cont_sentence_padding = 0
                        else:
                            word = str(list_item[1][1]) + ' '
                            file_corpus.write(word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                            element.clear()
                            cont_sentence_padding += 1
            print('End of the creating of the corpus..')
            print('The corpus was saved in the file : %s' % path_save_corpus_par)

        def create_and_save_corpus_domains_y(self, path_corpus_babelnet_par:str, path_save_corpus_par: str) -> None:
            """ This method is called whenever is necessary to create and save corpus for training sets about domains (coarse-grained) in
                the path file given as parameter. the method needs to take three parameters: one is the path of file bebelnet corpus
                and one for the path where will save the corpus. In this case,the  inter parameter "break_sentence" is not necessary
                because the bebelnet corpus is already worked. The corpus is created following the rules used to create babelnet corpus:
                - for each word no WSD (wf notation), its value string is taken,
                - for each word WSD, its lemma is taken and its relative domain from mapping provided (bn-domain),
                - if sentence goes beyond twentieth word (default), it must broken. Otherwise it is broken when finish sentence
                (\sentence notation)
            """

            mapping = Preprocessing.Mapping()
            dict_bn_domains = mapping.getDictAllBnDomain(mapping_bn2dom_path=paths.MAPPING_BN2WNDOM_PATH)
            with open(path_save_corpus_par, 'a', encoding='utf-8') as file_save:
                        corpus = Preprocessing.get_sentence_tokenize_from_path_corpus(self,path_corpus_babelnet_par)
                        for sentence in corpus:
                            for word in sentence.split():
                                    if('bn:' in word ):  #if in the word there is bn sysnet
                                       try:
                                           domain = dict_bn_domains[word]
                                           file_save.write(domain + ' ')
                                       except KeyError:
                                           file_save.write('factotum' + ' ')
                                    else:
                                           file_save.write(word+ ' ')
                            file_save.write('\n')


            print('End of the creating of the corpus..')
            print('The corpus was saved in the file : %s' % path_save_corpus_par)

        def create_and_save_corpus_lexnames_y(self, path_xml_par: str,path_gold_key_par:str, path_save_corpus_par: str, break_sentence = 20) -> None:
            """ This method is called whenever is necessary to create and save corpus for training sets about lexnames (coarse-grained) in
                the path file given as parameter. the method needs to take three parameters: one is the path of file Raganato
                in xml format, one for the path where will save the corpus and finally one inter parameter that tells to the method
                where each sentence of corpus has to broke if it goes beyond the "break_sentence" word. This last solution
                wis adopted to avoid the issue about missing WSd words. The corpus is created following this rules:
                - for each word no WSD (wf notation), its value string is taken,
                - for each word WSD, its lemma is taken and its relative babelnet synset and
                then its relative lexnames form mapping provided (bn-lexname),
                - if sentence goes beyond twentieth word (default), it must broken. Otherwise it is broken when finish sentence
                (\sentence notation)
            """

            xml_file = etree.iterparse(path_xml_par)
            print('#####################################')
            print('Beginning of the creating of the corpus labels lexnames from the file xml :' + path_xml_par)
            mapping = Preprocessing.Mapping()
            dict_id_key = mapping.getDictAllIdKey(path_gold_key_par)
            dict_bn_wn = mapping.getDictAllBnWn(mapping_bn2wn_path=paths.MAPPING_BN2WN_PATH)
            dict_bn_lex = mapping.getDictAllBnLex(mapping_bn2lex_path=paths.MAPPING_BN2LEX_PATH)
            cont_sentence_padding = 0
            for event, element in xml_file:
                with open(path_save_corpus_par, "a", encoding='utf-8') as file_corpus:
                    list_item = element.items()
                    try:
                        if (element.tag == "wf"):
                            if (cont_sentence_padding == break_sentence - 1):
                                if (list_item[1][1] != '.'):
                                    word = str(element.text) + '\n'
                                    file_corpus.write(word)
                                    element.clear()
                                    cont_sentence_padding = 0
                                else:
                                    file_corpus.write('\n')
                                    element.clear()
                                    cont_sentence_padding = 0
                            else:
                                if (list_item[1][1] != '.'):
                                    word = str(element.text) + ' '
                                    file_corpus.write(word)
                                    element.clear()
                                    cont_sentence_padding += 1
                                else:
                                    element.clear()
                        if (element.tag == "instance"):
                            word_from_id = dict_id_key[list_item[0][1]]   #lemma
                            if ' ' in word_from_id:  # if there are two synset
                                list_wn_id = word_from_id.split()
                                synset = Preprocessing.synset_from_sense_key(self,list_wn_id[0])
                                synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()  #wordnet synset
                            else:
                                synset = Preprocessing.synset_from_sense_key(self,word_from_id)
                                synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
                            bn_synset = dict_bn_wn[synset_id]
                            lexname = dict_bn_lex[bn_synset]
                            if (cont_sentence_padding == break_sentence - 1):
                                wn_word = str(lexname) + '\n'
                                file_corpus.write(wn_word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                                element.clear()
                                cont_sentence_padding = 0
                            else:
                                wn_word = str(lexname) + ' '
                                file_corpus.write(wn_word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                                element.clear()
                                cont_sentence_padding += 1
                        if (element.tag == "sentence"):
                            if (cont_sentence_padding == 0):
                                file_corpus.close()
                                element.clear()
                            else:
                                file_corpus.write('\n')
                                cont_sentence_padding = 0
                                file_corpus.close()
                                element.clear()
                    except WordNetError:
                        if (cont_sentence_padding == break_sentence - 1):
                            word = str(list_item[1][1]) + '\n'
                            file_corpus.write(word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                            element.clear()
                            cont_sentence_padding = 0
                        else:
                            word = str(list_item[1][1]) + ' '
                            file_corpus.write(word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                            element.clear()
                            cont_sentence_padding += 1

            print('End of the creating of the corpus..')
            print('The corpus was saved in the file : %s' % path_save_corpus_par)

        def get_corpus_for_prediction(self, path_xml_par: str, return_dict_WSDid_lemma: bool, break_sentence=20) -> object:

            """ This method is called whenever is necessary to retunr corpus for prediction.the method needs to take three parameters: one is the path of file Raganato
               in xml format, one of kind boolean to return also dict amoung lemma and WSD's id and finally one inter parameter that tells to the method
               where each sentence of corpus has to broke if it goes beyond the "break_sentence" word. This last solution
               wis adopted to avoid the issue about missing WSd words. The corpus is created following this rules:
               - for each word no WSD (wf notation), its value string is taken following by its pos (%POS: pos)
               - for each word WSD, its lemma is taken following by its pos (%POS: pos) and its ID (%ID: id)
               - if sentence goes beyond twentieth word (default), it must broken. Otherwise it is broken when finish sentence(\sentence notation)
            """

            xml_file = etree.iterparse(path_xml_par)
            print('#####################################')
            print('Beginning of the creating of the corpus for data x from the file xml : %s' % path_xml_par)
            corpus_sentence = []
            corpus = []
            dictionary_id_lemma = dict()
            cont_sentence_padding = 0
            for event, element in xml_file:
                    list_item = element.items()
                    try:
                        if (element.tag == "wf"):
                            if (cont_sentence_padding == break_sentence - 1):
                                if (list_item[1][1] != '.'):
                                    word = str(element.text + '%POS:' + list_item[1][1])   #text + pos
                                    corpus_sentence.append(word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                                    corpus.append(corpus_sentence)
                                    corpus_sentence = []
                                    element.clear()
                                    cont_sentence_padding = 0
                                else:
                                    corpus.append(corpus_sentence)
                                    corpus_sentence = []
                                    element.clear()
                                    cont_sentence_padding = 0
                            else:
                                if(list_item[1][1] != '.'):
                                    word = str(element.text + '%POS:' + list_item[1][1]) # lemma + pos
                                    corpus_sentence.append(word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                                    element.clear()
                                    cont_sentence_padding += 1
                                else:
                                    element.clear()
                        if (element.tag == "instance"):
                            if (cont_sentence_padding == break_sentence - 1):
                                    word = str(list_item[1][1] + '%POS:' + list_item[2][1] + '%ID:' + list_item[0][1])  # lemma + pos + id
                                    corpus_sentence.append(word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                                    dictionary_id_lemma[word] = list_item[0][1]
                                    corpus.append(corpus_sentence)
                                    corpus_sentence = []
                                    element.clear()
                                    cont_sentence_padding = 0
                            else:
                                    word = str(list_item[1][1] + '%POS:' + list_item[2][1] + '%ID:' + list_item[0][1])  #lemma + pos + id
                                    corpus_sentence.append(word.replace('&apos;&apos;', "''").replace('&apos;', "'"))
                                    dictionary_id_lemma[word] = list_item[0][1]
                                    element.clear()
                                    cont_sentence_padding += 1
                        if (element.tag == "sentence"):
                            if (cont_sentence_padding == 0):
                                element.clear()
                            else:
                                corpus.append(corpus_sentence)
                                corpus_sentence = []
                                element.clear()
                                cont_sentence_padding = 0
                    except Exception as e:
                        print(e)
            if(return_dict_WSDid_lemma):
                return  corpus, dictionary_id_lemma
            else:
                return  corpus

    class TrainingData(object):
        """ This is a sub-class for the creation of Training set and own vacabulary."""

        def switch_type_task(self,type_task):
             """ This method is called whenever is necessary to return the type of task.It is adopted this solution
             to develop a directional method which does the same function but for different task"""
             switcher = {
                "input_x": 1,
                "babelnet": 2,
                "domains": 3,
                "lexnames": 4
             }
             return switcher.get(type_task, "Invalid Type Task")

        def create_and_save_single_vocabolary(self,path_save_file_par: str, type_vocab_par: str):
                """ This method is called whenever is necessary to create and so save a single vocbulary. It was used
                   to load vocabulary on google colab.
                   Its parameters are:
                   -the path where the vocabulary will be saved
                   -the type of task: input_x,babelnet,domains or lexnames"""

                print('Start creating and saving Vocabolary from corpus...')
                mapping = Preprocessing.Mapping()
                index_type_vocab = Preprocessing.TrainingData.switch_type_task(self,type_task=type_vocab_par)
                if(index_type_vocab != 'Invalid Type Task'):
                    list_all_words_for_vocab = []
                    if (index_type_vocab == 1):  # input task
                        model = KeyedVectors.load(paths.MODEL_WORD_EMBEDDINGS, mmap='r') #load model words embeddings
                        list_all_words_for_vocab = list(model.wv.vocab)                 #creates a list from word emebeddings
                    if(index_type_vocab == 2):   # babelnet task
                        dict_bn = mapping.getDictAllBnWn(mapping_bn2wn_path=paths.MAPPING_BN2WN_PATH)  #compute dict bn-wn
                        for key,value in dict_bn.items():
                            list_all_words_for_vocab.append(value)
                        list_all_words_for_vocab = list(set((list_all_words_for_vocab))) #create a list form mapping whitout duplicates
                    if (index_type_vocab == 3):  # domains task
                        dict_domain = mapping.getDictAllBnDomain(mapping_bn2dom_path=paths.MAPPING_BN2WNDOM_PATH) #compute dict bn-dm
                        for key, value in dict_domain.items():
                            list_all_words_for_vocab.append(value)
                        list_all_words_for_vocab = list(set((list_all_words_for_vocab))) #create a list form mapping whitout duplicates
                    if (index_type_vocab == 4):  #lexnames task
                        dict_lex = mapping.getDictAllBnLex(mapping_bn2lex_path=paths.MAPPING_BN2LEX_PATH) #compute dict bn-lex
                        for key, value in dict_lex.items():
                            list_all_words_for_vocab.append(value)
                        list_all_words_for_vocab = list(set((list_all_words_for_vocab))) #create a list form mapping whitout duplicates
                    list_all_words_for_vocab.sort()
                    vocabolary = dict()
                    vocabolary['UNK'] = 1
                    index = 2
                    for word in list_all_words_for_vocab:
                        vocabolary[word] = index
                        index += 1
                    with open(path_save_file_par, 'wb') as save_file:
                         pickle.dump(vocabolary, save_file)
                         return print('single vocabolary created %s and saved in to the file : %s' % (type_vocab_par,path_save_file_par))
                else:
                     print('The Parameter of task%s type is not valid: please choose among: %s, %s, %s, %s' % ("'s","input_x","babelnet","domains","lexnames") )

        def get_single_vocabolary(self,type_vocab_par: str, path_mapping_task: str):
                """ This method is called whenever is necessary to return the single vocabulary
                   depending on task.Its parameters are:
                   -the path of mapping file with which the vacabulary can be created
                   -the type of task: input_x,babelnet,domains or lexnames"""

                mapping = Preprocessing.Mapping()
                index_type_vocab = Preprocessing.TrainingData.switch_type_task(self,type_task=type_vocab_par)
                if(index_type_vocab != 'Invalid Type Task'):  #task not valid
                    list_all_words_for_vocab = []
                    if (index_type_vocab == 1):     #input task
                        model = KeyedVectors.load(path_mapping_task, mmap='r')  #load model word emenddings
                        list_all_words_for_vocab = list(model.wv.vocab)    #create a list from word embeddings
                    if(index_type_vocab == 2):     #babelnet task
                        dict_bn = mapping.getDictAllBnWn(mapping_bn2wn_path=path_mapping_task) #return the dictionary from mapping bn-wn
                        for key,value in dict_bn.items():    #iterate over it (key , value)
                            list_all_words_for_vocab.append(value)
                        list_all_words_for_vocab = list(set((list_all_words_for_vocab))) #create a list form mapping whitout duplicates
                    if (index_type_vocab == 3):   #domains task
                        dict_domain = mapping.getDictAllBnDomain(mapping_bn2dom_path=path_mapping_task) #return the dictionary from mapping bn-dm
                        for key, value in dict_domain.items(): #iterate over it (key, value)
                            list_all_words_for_vocab.append(value)
                        list_all_words_for_vocab = list(set((list_all_words_for_vocab))) #create a list form mapping whitout duplicates
                    if (index_type_vocab == 4):   #lexnames task
                        dict_lex = mapping.getDictAllBnLex(mapping_bn2lex_path=path_mapping_task) #return the dictionary from mapping bn-lex
                        for key, value in dict_lex.items(): #iterate over it  (key,value)
                            list_all_words_for_vocab.append(value)
                        list_all_words_for_vocab = list(set((list_all_words_for_vocab))) #create a list form mapping whitout duplicates
                    list_all_words_for_vocab.sort()  #order the list composed by all element needs whithout duplicate
                    vocabolary = dict()
                    vocabolary['UNK'] = 1  #the fist element is the UNK notation
                    index = 2              #start from 2
                    for word in list_all_words_for_vocab:  #for each word in the list
                        vocabolary[word] = index           #create vocabulary: value:index and key:word
                        index += 1
                    return  vocabolary                     #return vocabulary
                else:
                     print('The Parameter of task%s type is not valid: please choose among: %s, %s, %s, %s' % ("'s","input_x","babelnet","domains","lexnames") )

        def save_data_from_corpus_file(self,path_corpus_par :str,path_save_training_data_par : str,type_vocab_par:str, padding_par=20):
                """ This method is called whenever is necessary to create and save a trainign/ testing sets for model.It was used
                 to save end load the training and testing sets on google colab
                    Its parameters are:
                   - the path of corpus in where there are all need data
                   -the path where the trainign/testing data will be saved
                   -the type of task: input_x,babelnet,domains or lexnamesù
                   - the integer value for padding"""

                training_data_x = []
                sentences = Preprocessing.get_sentence_tokenize_from_path_corpus(self,path_corpus_par)
                index_type_vocab = Preprocessing.TrainingData.switch_type_task(self,type_task=type_vocab_par)
                if (index_type_vocab != 'Invalid Type Task'):
                        if(index_type_vocab == 1):
                            vocabolary = pickle.load(open(paths.INPUT_X_VOCABOLARY_PICKLE_PATH, "rb"))
                        elif(index_type_vocab== 2):
                            vocabolary = pickle.load(open(paths.BABELNET_VOCABOLARY_PICKLE_PATH, "rb"))
                        elif(index_type_vocab== 3):
                            vocabolary = pickle.load(open(paths.DOMAINS_VOCABOLARY_PICKLE_PATH, "rb"))
                        elif(index_type_vocab==4):
                            vocabolary = pickle.load(open(paths.LEXNAMES_VOCABOLARY_PICKLE_PATH, "rb"))
                        with open(path_save_training_data_par, "wb") as file_save:
                            for sentence in sentences:
                                words = word_tokenize(sentence)
                                training_data_sentence = []
                                for word in words:
                                    if word.lower() in vocabolary :
                                       training_data_sentence.append(vocabolary[word.lower()])
                                    else:
                                       training_data_sentence.append(vocabolary['UNK'])
                                training_data_x.append(np.array(training_data_sentence))
                            training_data_x = pad_sequences(training_data_x, truncating='pre', padding='post',maxlen=padding_par)
                            pickle.dump(training_data_x, file_save)
                        print('save data for %s in the file: %s ' % (type_vocab_par,path_save_training_data_par))
                else:
                    print('The Parameter of task%s type is not valid: please choose among: %s, %s, %s, %s' % ("'s", "input_x", "babelnet", "domains", "lexnames"))

        def save_embeddings_matrix(self, path_save_matrix_par: str, path_model_word_emebdding_par: str):
            """ This method is called whenever is necessary to create the embedding matrix .It was used
                to save and load the embedding matrix on google colab
                   Its parameters are:
                   -the path where the emebddings matrix will be saved
                   -the path in where tehre is the model for words embeddings"""

            model = KeyedVectors.load(path_model_word_emebdding_par)
            with open(paths.INPUT_X_VOCABOLARY_PICKLE_PATH, 'rb') as file:
                vocab = pickle.load(file)
                vocab_size = len(vocab) + 1
                embedding_matrix = np.zeros((vocab_size, 100))
                i = 0
                for word in model.wv.vocab:
                    embedding_vector = model[word]
                    if (embedding_vector is not None):
                        embedding_matrix[i] = embedding_vector
                    i += 1
            with open(path_save_matrix_par, 'wb') as save_file:
                 pickle.dump(embedding_matrix, save_file)
                 return print('embeddings matrix was saved to the file: %s' % path_save_matrix_par)

        def Summary(self):
            """ This method is called whenever is necessary to print all summary about vaculary and training/testing sets"""

            with open(paths.INPUT_X_VOCABOLARY_PICKLE_PATH, 'rb') as file:
                vocab_x = pickle.load(file)
            with open(paths.BABELNET_VOCABOLARY_PICKLE_PATH, 'rb') as file:
                vocab_bn = pickle.load(file)
            with open(paths.DOMAINS_VOCABOLARY_PICKLE_PATH, 'rb') as file:
                vocab_domains = pickle.load(file)
            with open(paths.LEXNAMES_VOCABOLARY_PICKLE_PATH, 'rb') as file:
                vocab_lexname = pickle.load(file)

                print('len vocabolary x : {0}' .format(len(vocab_x)))
                print('len vocabolary babelnet : {0}' .format(len(vocab_bn)))
                print('len vocabolary domains : {0}' .format(len(vocab_domains)))
                print('len vocabolary lexnames : {0}' .format(len(vocab_lexname)))

            with open(paths.TRAINING_X_PICKLE_PATH, 'rb') as file:
                 u = pickle.load(file)
                 print('SHAPE TRAINING X : {0}'  .format(np.shape(u)))
            with open(paths.TRAINING_Y_BABELNET_PICKLE_PATH, 'rb') as file:
                 u = pickle.load(file)
                 print('SHAPE TRAINING BABELNET Y : {0}'  .format(np.shape(u)))
            with open(paths.TRAINING_Y_LEXNAMES_PICKLE_PATH, 'rb') as file:
                 u = pickle.load(file)
                 print('SHAPE TRAINING LEXINAMES Y : {0}' .format(np.shape(u)))
            with open(paths.TRAINING_Y_DOMAINS_PICKLE_PATH, 'rb') as file:
                 u = pickle.load(file)
                 print('SHAPE TRAINING DOMAINS Y : {0}' .format(np.shape(u)))

            with open(paths.TRAINING_X_PICKLE_DEV_PATH, 'rb') as file:
                u = pickle.load(file)
                print('SHAPE TRAINING X DEV : {0}' .format(np.shape(u)))
            with open(paths.TRAINING_Y_BABELNET_PICKLE_DEV_PATH, 'rb') as file:
                u = pickle.load(file)
                print('SHAPE TRAINING BABELNET Y DEV : {0}' .format(np.shape(u)))
            with open(paths.TRAINING_Y_LEXNAMES_PICKLE_DEV_PATH, 'rb') as file:
                u = pickle.load(file)
                print('SHAPE TRAINING LEXINAMES Y DEV : {0}' .format(np.shape(u)))
            with open(paths.TRAINING_Y_DOMAINS_PICKLE_DEV_PATH, 'rb') as file:
                u = pickle.load(file)
                print('SHAPE TRAINING DOMAINS Y DEV : {0}' .format(np.shape(u)))

    class PredictionData(object):
        """ This is a sub-class for preprocessing of prediction. its methods are called whenever is necessary
            to does some preprocessings actions on predict function.
        """

        def get_input_x(self,type_task: str,path_mapping_for_vocab:str,corpus:str,padding_par=20):
            """ This method is called whenever is necessary returning the data evaluation for predictions. It aims to
            take data from xml file raganato and create evaluation data following the vocabulary in according to task.
             Its parameters are:
                -the path of mapping file with which the vacabulary can be created
                -the type of task: input_x,babelnet,domains or lexnames
                -the corpus with which is possible to create training/testing sets
                -the integer value for paddding"""

            input_data_x = []
            index_type_vocab = Preprocessing.TrainingData.switch_type_task(self,type_task=type_task) # compute index of task
            if (index_type_vocab != 'Invalid Type Task'):  # if the task is not valid
                vocabolary = Preprocessing.TrainingData.get_single_vocabolary(self,type_vocab_par = type_task,path_mapping_task=path_mapping_for_vocab)
                print('lenght of Corpus Input X: %s' % str(len(corpus)))
                for sentence in corpus:                    # for each sentence of corpus for prediction
                    input_data_sentence = []
                    for word in sentence:                  # for each word of sentence
                        if word[:word.index('%POS:')].lower() in vocabolary:    # if lemma + pos is in the vocabulary, then append it (lower case)
                            input_data_sentence.append(vocabolary[word[:word.index('%POS:')].lower().lower()])
                        else:
                            input_data_sentence.append(vocabolary['UNK'])    # if is not in the vocabulary, append UNK notation (value 1)
                    input_data_x.append(np.array(input_data_sentence))
                input_data_x = pad_sequences(input_data_x, truncating='pre', padding='post', maxlen=padding_par)  #padding of input
                Preprocessing.PredictionData.Summary(self=None,input_x = input_data_x, vocabolary= vocabolary)    #summary
                return input_data_x,vocabolary    # return data input and its vocabulary
            else:
                print('The Parameter of type%s type is not valid: please choose among: %s, %s, %s, %s' % ( "'s", "input_x", "babelnet", "domains", "lexnames"))

        def get_list_candidates(self, type_task:str,  lemma_par: str, pos_par: str, path_mapping_bn2wn:str,vocabolary_task:dict, vocabolary_lemma:dict,path_mapping_task=None):
            """ This method is called whenever is necessary returning the list of candidates  and their index from vocabulary in according to teh task
            Its parameters are:
                -the path of mapping file with which the vacabulary can be created
                -the type of task: babelnet,domains or lexnames
                -the lemma of words
                -the pos opf words
                -the path of babelnet-wordnet mapping
                -the vocabulary in according to the task
                -the path of mapping in according to teh task: (for babelnet is not necessary, for domains = mapping from bn->domains
                                                                  and for lexnames = mapping from bn-> lexnames)
                -the integer value for paddding"""

            index_type_vocab = Preprocessing.TrainingData.switch_type_task(self, type_task=type_task)
            if(index_type_vocab != 'Invalid Type Task' or index_type_vocab != 1):
                pos_tag_dict = {"ADJ": wordnet.ADJ,
                            "NOUN": wordnet.NOUN,
                            "VERB": wordnet.VERB,
                            "ADV": wordnet.ADV,
                            "NUM": wordnet.NOUN}
                pos_tag = pos_tag_dict[pos_par]       # take its corresponding wn-pos tag from POS
                synset_condidates_wn = []
                if lemma_par in  vocabolary_lemma:    # if lemma is in the vocabulary, compute its sysnet keys synset from lemma and pos
                    candidates = wordnet.synsets(lemma_par, pos_tag)
                    for syns in candidates:           # for each sysnet key candidates, compute its wordnet synset
                        synset_condidates_wn.append("wn:" + str(syns.offset()).zfill(8) + syns.pos())
                else:                                 # if lemma is not in the vocabulary, then take its most frequences synset (MFS)
                    candidates = wordnet.synsets(lemma_par)[0]   #MFS
                    synset_condidates_wn.append("wn:" + str(candidates.offset()).zfill(8) + candidates.pos())
                mapping_bn_wn = Preprocessing.Mapping.getDictAllBnWn(self=None,mapping_bn2wn_path=path_mapping_bn2wn)
                babelnet_candidates = []
                for wn in synset_condidates_wn:        # for each wordnet, map them in babelnet synsets
                    if wn in mapping_bn_wn:
                        babelnet_candidates.append(mapping_bn_wn[wn])   # take all babelnet synsets
                list_candidates = []
                if(index_type_vocab == 2):             # if the task is babelnet
                    list_candidates = babelnet_candidates     #the list of candidates is for babelnet
                elif(index_type_vocab == 3):           # if the task is domains
                    mapping_bn_domains = Preprocessing.Mapping.getDictAllBnDomain(self=None, mapping_bn2dom_path=path_mapping_task) #computes dictionary domains-babelnet from mapping
                    for bn in babelnet_candidates:     # for each babelnet candidates
                        try:
                            list_candidates.append(mapping_bn_domains[bn])  # if babelnet synset is in the dict, then take it
                        except KeyError:
                            list_candidates.append('factotum')              # else take the factotum notation
                elif(index_type_vocab == 4):            # if the task is lexnames
                    mapping_bn_lexnames = Preprocessing.Mapping.getDictAllBnLex(self=None,mapping_bn2lex_path=path_mapping_task) #compute dictionary for babelnet-lexnames
                    for bn in babelnet_candidates:      # for each babelnet synsets
                        try:
                          list_candidates.append(mapping_bn_lexnames[bn])  # takes lexnames corresponding to the babelnet synset
                        except KeyError:                # else continue
                          continue;
                list_index = []
                for candidate in list_candidates:       # for each candidates
                    for value, index in vocabolary_task.items():   # for each value index from vocabulary about task
                        if (value == candidate):        # if the value is in the vocabulary
                            list_index.append(index)    # takes its index from vccabulary
                return list_index, list_candidates      # return the list of candidates and its list of index from vocabulary
            else:
                print('The Parameter of type%s type is not valid: please choose among: %s, %s, %s' % ("'s", "babelnet", "domains", "lexnames"))

        def Summary(self, input_x: str, vocabolary: dict):
            return print('SHAPE INPUT X : {0}' .format(np.shape(input_x)) + '\nSHAPE VOCABOLARY X : {0}'.format(len(vocabolary)))

    class EvaluationData(object):
        """ This is a sub-class for evaluation. It aims to expose methods ( in this case only one) to create gold data for evaluation"""

        def save_gold_key_true(self, path_gold_key_par:str, type_task_par: str, path_save_evaluation:str,path_mapping_task=None, path_mapping_bn2wn_par=paths.MAPPING_BN2WN_PATH):
            """ this method is used whenever is necessary create the gold evaluation file to compute the score F1 with prediction's file:
             it parameters are:
             -the path of gold key from evaluation datasets
             -the type of task: babelnet,domains and lexnames
             -the path of file where evaluations will be saved
             - the path of mapping in according to the task  (for babelnet is not necessary, for domains = mapping from bn->domains
                                                              and for lexnames = mapping from bn-> lexnames)"""

            type_task_par = Preprocessing.TrainingData.switch_type_task(self=None, type_task=type_task_par)
            dict_mapping_key = Preprocessing.Mapping.getDictAllIdKey(self=None,data_gold_key_path=path_gold_key_par)
            with open(path_save_evaluation, 'a',encoding='utf-8') as file_save:
                if(type_task_par != 'Invalid Type Task' or type_task_par != 1):
                    dict_mapping_bn = Preprocessing.Mapping.getDictAllBnWn(self=None, mapping_bn2wn_path=path_mapping_bn2wn_par)
                    if(type_task_par == 2):   #task babelnet
                        for key, value in dict_mapping_key.items():
                            try:
                                if ' ' in value:  # two synset
                                    list_wn_id = value.split()
                                    synset = Preprocessing.synset_from_sense_key(self, list_wn_id[0])
                                    synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
                                    synset_bn = dict_mapping_bn[synset_id]
                                else:
                                    synset = Preprocessing.synset_from_sense_key(self, value)
                                    synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
                                    synset_bn = dict_mapping_bn[synset_id]
                                file_save.write("%s %s\n" % (key,synset_bn))
                            except:
                                file_save.write("%s %s\n" % (key, synset_bn))
                        file_save.close()
                    if (type_task_par == 3):  #task domains
                        dict_mapping_domains = Preprocessing.Mapping.getDictAllBnDomain(self=None,mapping_bn2dom_path=path_mapping_task)
                        for key, value in dict_mapping_key.items():
                            try:
                                if ' ' in value:  # two synset key
                                    list_wn_id = value.split()
                                    synset = Preprocessing.synset_from_sense_key(self, list_wn_id[0])
                                    synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
                                    synset_bn = dict_mapping_bn[synset_id]
                                else:
                                    synset = Preprocessing.synset_from_sense_key(self, value)
                                    synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
                                    synset_bn = dict_mapping_bn[synset_id]
                                try:
                                    domain = dict_mapping_domains[synset_bn]
                                    file_save.write("%s %s\n" % (key, "\t".join(domain.split('/'))))
                                except KeyError:
                                    file_save.write("%s %s\n" % (key, "factotum"))
                            except WordNetError:
                                try:
                                    domain = dict_mapping_domains[synset_bn]
                                    file_save.write("%s %s\n" % (key, "\t".join(domain.split('/'))))
                                except KeyError:
                                    file_save.write("%s %s\n" % (key, "factotum"))
                        file_save.close()
                    if (type_task_par == 4): #task lexnames
                        dict_mapping_lexnames = Preprocessing.Mapping.getDictAllBnLex(self=None,mapping_bn2lex_path=path_mapping_task)
                        for key, value in dict_mapping_key.items():
                            try:
                                if ' ' in value:  # two synset key
                                    list_wn_id = value.split()
                                    synset = Preprocessing.synset_from_sense_key(self, list_wn_id[0])
                                    synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
                                    synset_bn = dict_mapping_bn[synset_id]
                                else:
                                    synset = Preprocessing.synset_from_sense_key(self, value)
                                    synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
                                    synset_bn = dict_mapping_bn[synset_id]
                                lexnames = dict_mapping_lexnames[synset_bn]
                                file_save.write("%s %s\n" % (key, lexnames))
                            except WordNetError:
                                file_save.write("%s %s\n" % (key, lexnames))
                        file_save.close()
                else:
                    print('The Parameter of type%s type is not valid: please choose among: %s, %s, %s' % ("'s", "babelnet", "domains", "lexnames"))

