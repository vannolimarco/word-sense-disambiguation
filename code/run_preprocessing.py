import preprocessing
import pathconfig

######################################################################
###################### RUN FOR PREPROCESSING  ########################
######################################################################

""" in this file py, is possible run all preprocessing in order to elaborate data for the training of model.
    It use the class Preprocessing and its relative sub-class. Many files were deleted , like Semcor, so is not
    possible run this part of preprocessing"""

paths = pathconfig.paths()   #create instance for paths from config_path file

corpus = preprocessing.Preprocessing.Corpus()  # instance class corpus

input_x = "input_x"
babelnet_task = 'babelnet'
domains_taks = 'domains'
lexnames_task = 'lexnames'


""" Creation of all corpus needs for training sets"""

print('START corpus TRAINING...')
corpus.create_and_save_corpus_x(path_xml_par=paths.SEMCOR_PATH,path_save_corpus_par=paths.CORPUS_X_PATH) #creates the corpus for input data
corpus.create_and_save_corpus_babelnet_y(path_xml_par=paths.SEMCOR_PATH,path_gold_key_par=paths.SEMCOR_GOLD_KEY_PATH,path_save_corpus_par=paths.CORPUS_BABELNET_LABELS_PATH) #creates the corpus fine grained
corpus.create_and_save_corpus_domains_y(path_corpus_babelnet_par=paths.CORPUS_BABELNET_LABELS_PATH,path_save_corpus_par=paths.CORPUS_DOMAINS_LABELS_PATH) #creates the corpus coarse grained (domains)
corpus.create_and_save_corpus_lexnames_y(path_xml_par=paths.SEMCOR_PATH,path_gold_key_par=paths.SEMCOR_GOLD_KEY_PATH,path_save_corpus_par=paths.CORPUS_LEX_LABELS_PATH) #creates the corpus coarse grained (lexnames)

""" Creation of all corpus needs for testing sets """
print('START corpus TESTING...')
corpus.create_and_save_corpus_x(path_xml_par=paths.SEMEVAL2007_PATH,path_save_corpus_par=paths.CORPUS_X_DEV_PATH) #create the corpus for input testing stes
corpus.create_and_save_corpus_babelnet_y(path_xml_par=paths.SEMEVAL2007_PATH,path_gold_key_par=paths.SEMEVAL2007_GOLD_KEY_PATH,path_save_corpus_par=paths.CORPUS_BABELNET_LABELS_DEV_PATH)   #creates the corpus fine grained (testing sets)
corpus.create_and_save_corpus_domains_y(path_corpus_babelnet_par=paths.CORPUS_BABELNET_LABELS_DEV_PATH,path_save_corpus_par=paths.CORPUS_DOMAINS_LABELS_DEV_PATH)  #creates the corpus coarse grained domains (testing sets)
corpus.create_and_save_corpus_lexnames_y(path_xml_par=paths.SEMEVAL2007_PATH,path_gold_key_par=paths.SEMEVAL2007_GOLD_KEY_PATH,path_save_corpus_par=paths.CORPUS_LEX_LABELS_DEV_PATH) #creates the corpus coarse grained (lexnames) testing sets


#PREPROCESSING CREATE TRAINING DATA X AND Y

preprocessing = preprocessing.Preprocessing.TrainingData()  #create object from sub class training data

#VOCABOLARY
""" Creation of all vocabulary """
preprocessing.create_and_save_single_vocabolary(path_save_file_par=paths.INPUT_X_VOCABOLARY_PICKLE_PATH,type_vocab_par=input_x) #create and save vocanulary for input data
preprocessing.create_and_save_single_vocabolary(path_save_file_par=paths.BABELNET_VOCABOLARY_PICKLE_PATH,type_vocab_par=babelnet_task) #create and save vocanulary for babelnet task
preprocessing.create_and_save_single_vocabolary(path_save_file_par=paths.DOMAINS_VOCABOLARY_PICKLE_PATH,type_vocab_par=domains_taks) #create and save vocanulary for domains task
preprocessing.create_and_save_single_vocabolary(path_save_file_par=paths.LEXNAMES_VOCABOLARY_PICKLE_PATH,type_vocab_par=lexnames_task) #create and save vocanulary for lexnames task

#TRAINING X Y
""" Creation of all training sets """
preprocessing.save_data_from_corpus_file(path_corpus_par=paths.CORPUS_X_PATH,path_save_training_data_par=paths.TRAINING_X_PICKLE_PATH,type_vocab_par=input_x)
preprocessing.save_data_from_corpus_file(path_corpus_par=paths.CORPUS_BABELNET_LABELS_PATH,path_save_training_data_par=paths.TRAINING_Y_BABELNET_PICKLE_PATH,type_vocab_par=babelnet_task)
preprocessing.save_data_from_corpus_file(path_corpus_par=paths.CORPUS_DOMAINS_LABELS_PATH,path_save_training_data_par=paths.TRAINING_Y_DOMAINS_PICKLE_PATH,type_vocab_par=domains_taks)
preprocessing.save_data_from_corpus_file(path_corpus_par=paths.CORPUS_LEX_LABELS_PATH,path_save_training_data_par=paths.TRAINING_Y_LEXNAMES_PICKLE_PATH,type_vocab_par=lexnames_task)

#TESTING X Y
""" Creation of all testing sets """
preprocessing.save_data_from_corpus_file(path_corpus_par=paths.CORPUS_X_DEV_PATH,path_save_training_data_par=paths.TRAINING_X_PICKLE_DEV_PATH,type_vocab_par=input_x)
preprocessing.save_data_from_corpus_file(path_corpus_par=paths.CORPUS_BABELNET_LABELS_DEV_PATH,path_save_training_data_par=paths.TRAINING_Y_BABELNET_PICKLE_DEV_PATH,type_vocab_par=babelnet_task)
preprocessing.save_data_from_corpus_file(path_corpus_par=paths.CORPUS_DOMAINS_LABELS_DEV_PATH,path_save_training_data_par=paths.TRAINING_Y_DOMAINS_PICKLE_DEV_PATH,type_vocab_par=domains_taks)
preprocessing.save_data_from_corpus_file(path_corpus_par=paths.CORPUS_LEX_LABELS_DEV_PATH,path_save_training_data_par=paths.TRAINING_Y_LEXNAMES_PICKLE_DEV_PATH,type_vocab_par=lexnames_task)


#MATRIX WORD EMBEDDINGS
preprocessing.save_embeddings_matrix(path_save_matrix_par=paths.EMBEDDING_MATRIX,path_model_word_emebdding_par=paths.MODEL_WORD_EMBEDDINGS)

############################################################
###################    SHAPE DATA     ######################
############################################################
preprocessing.Summary()


