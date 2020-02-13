import os

class paths(object):
    """ This is a class with goals to call all data paths from it. It  simplifies and streamlines the code from long paths.
    It is used following this rules:
    - in the file needs to include the file : import pathconfig,
    - create object from class : paths = pathconfig.paths()
    - call path from property of class: for example path_semcor = paths.SEMCOR_PATH
    All paths were set in order to respect the path of Raganato Framework.
    Change all path in order to set own path and used them in the code.
    I remember that for path mappings the path are the same. So use this class to call them.
    Many files were deleted so
    """

    def __init__(self):
        self.DATA_DIR = '..\\resources\\'    #the folder resources
        self.SEMEVAL2007 = 'semeval2007\\'
        self.SEMEVAL2013 = 'semeval2013\\'
        self.SEMEVAL2015 = 'semeval2015\\'
        self.SENSEVAL2 = 'senseval2\\'
        self.SENSEVAL3 = 'senseval3\\'

        # Resources path base
        self.BASE_RESOURCES = _BASE_RES_PATH = '..\\resources\\'
        self.BASE_RESOURCES_PREDICTIONS = _BASE_RES_PRED_PATH = '..\\resources\\predictions\\'
        self.BASE_RESOURCES_EVALUATIONS = _BASE_RES_EVAL_PATH = '..\\resources\\evaluations\\'

        # Corpus Training
        self.CORPUS_X_PATH = _BASE_RES_PATH + 'corpus_x.txt'
        self.CORPUS_WORDNET_LABELS_PATH = _BASE_RES_PATH + 'corpus_wordnet_labels.txt' #path of file fine grained
        self.CORPUS_DOMAINS_LABELS_PATH = _BASE_RES_PATH + 'corpus_domains_labels.txt'
        self.CORPUS_LEX_LABELS_PATH = _BASE_RES_PATH + 'corpus_lexname_labels.txt'
        self.CORPUS_BABELNET_LABELS_PATH = _BASE_RES_PATH + 'corpus_babelnet_labels.txt'

        # Corpus Test
        self.CORPUS_X_DEV_PATH = _BASE_RES_PATH + 'corpus_x_dev.txt'
        self.CORPUS_WORDNET_LABELS_DEV_PATH = _BASE_RES_PATH + 'corpus_wordnet_labels_dev.txt'  # path of file fine grained
        self.CORPUS_DOMAINS_LABELS_DEV_PATH = _BASE_RES_PATH + 'corpus_domains_labels_dev.txt'
        self.CORPUS_LEX_LABELS_DEV_PATH = _BASE_RES_PATH + 'corpus_lexname_labels_dev.txt'
        self.CORPUS_BABELNET_LABELS_DEV_PATH = _BASE_RES_PATH + 'corpus_babelnet_labels_dev.txt'


        # Corpus Evaluation
        self.CORPUS_EVALUATION_PATH = _BASE_RES_PATH + 'corpus_evaluation.txt'

        # Path base WSD framework
        self.BASE_DATA_PATH = _BASE_DATA_PATH = '..\\resources\\WSD_Evaluation_Framework\\WSD_Evaluation_Framework\\'

        # path for all-words trainining/test/evaluations
        self.SEMCOR_PATH = _BASE_DATA_PATH + 'Training_Corpora\\SemCor\\semcor.data.xml'
        self.SEMCOR_GOLD_KEY_PATH = _BASE_DATA_PATH + 'Training_Corpora\\SemCor\\semcor.gold.key.txt'
        self.SEMCOR_OMSTI_PATH = _BASE_DATA_PATH + 'Training_Corpora\\SemCor+OMSTI\\semcor+omsti.data.xml'
        self.SEMCOR_OMSTI_GOLD_KEY_PATH = _BASE_DATA_PATH + 'Training_Corpora\\SemCor+OMSTI\\semcor+omsti.gold.key.txt'
        self.SEMEVAL2007_PATH = _BASE_DATA_PATH + 'Evaluation_Datasets\\semeval2007\\semeval2007.data.xml'
        self.SEMEVAL2007_GOLD_KEY_PATH = _BASE_DATA_PATH + 'Evaluation_Datasets\\semeval2007\\semeval2007.gold.key.txt'
        self.SEMEVAL2013_PATH = _BASE_DATA_PATH + 'Evaluation_Datasets\\semeval2013\\semeval2013.data.xml'
        self.SEMEVAL2013_GOLD_KEY_PATH = _BASE_DATA_PATH + 'Evaluation_Datasets\\semeval2013\\semeval2013.gold.key.txt'
        self.SEMEVAL2015_PATH = _BASE_DATA_PATH + 'Evaluation_Datasets\\semeval2015\\semeval2015.data.xml'
        self.SEMEVAL2015_GOLD_KEY_PATH = _BASE_DATA_PATH + 'Evaluation_Datasets\\semeval2015\\semeval2015.gold.key.txt'
        self.SENSEVAL2_PATH = _BASE_DATA_PATH + 'Evaluation_Datasets\\senseval2\\senseval2.data.xml'
        self.SENSEVAL2_GOLD_KEY_PATH = _BASE_DATA_PATH + 'Evaluation_Datasets\\senseval2\\senseval2.gold.key.txt'
        self.SENSEVAL3_PATH = _BASE_DATA_PATH + 'Evaluation_Datasets\\senseval3\\senseval3.data.xml'
        self.SENSEVAL3_GOLD_KEY_PATH = _BASE_DATA_PATH + 'Evaluation_Datasets\\senseval3\\senseval3.gold.key.txt'
        self.ALL_PATH = _BASE_DATA_PATH + 'Evaluation_Datasets\\ALL\\ALL.data.xml'
        self.ALL_GOLD_KEY_PATH = _BASE_DATA_PATH + 'Evaluation_Datasets\\ALL\\ALL.gold.key.txt'

        # path mapping
        self.MAPPING_BN2WN_PATH = _BASE_RES_PATH + 'babelnet2wordnet.tsv'
        self.MAPPING_BN2WNDOM_PATH = _BASE_RES_PATH + 'babelnet2wndomains.tsv'
        self.MAPPING_BN2LEX_PATH =  _BASE_RES_PATH + 'babelnet2lexnames.tsv'

        # path file pickle

        self.ALL_VOCABOLARY_PICKLE_PATH = _BASE_RES_PATH + 'all_vocabolary.dat'
        self.INPUT_X_VOCABOLARY_PICKLE_PATH = _BASE_RES_PATH + 'input_x_vocabolary.dat'
        self.BABELNET_VOCABOLARY_PICKLE_PATH = _BASE_RES_PATH + 'babelnet_vocabolary.dat'
        self.DOMAINS_VOCABOLARY_PICKLE_PATH = _BASE_RES_PATH + 'domains_vocabolary.dat'
        self.LEXNAMES_VOCABOLARY_PICKLE_PATH = _BASE_RES_PATH + 'lexnames_vocabolary.dat'

        #TRAINGING
        self.TRAINING_X_PICKLE_PATH = _BASE_RES_PATH + 'training_x.dat'
        self.TRAINING_Y_BABELNET_PICKLE_PATH = _BASE_RES_PATH + 'training_y_babelnet.dat'
        self.TRAINING_Y_DOMAINS_PICKLE_PATH = _BASE_RES_PATH + 'training_y_domains.dat'
        self.TRAINING_Y_LEXNAMES_PICKLE_PATH = _BASE_RES_PATH + 'training_y_lexnames.dat'

        #TESTING
        self.TRAINING_X_PICKLE_DEV_PATH = _BASE_RES_PATH + 'training_x_dev.dat'
        self.TRAINING_Y_BABELNET_PICKLE_DEV_PATH = _BASE_RES_PATH + 'training_y_babelnet_dev.dat'
        self.TRAINING_Y_DOMAINS_PICKLE_DEV_PATH = _BASE_RES_PATH + 'training_y_domains_dev.dat'
        self.TRAINING_Y_LEXNAMES_PICKLE_DEV_PATH = _BASE_RES_PATH + 'training_y_lexnames_dev.dat'

        #WORD EMBEDDINGS
        self.MODEL_WORD_EMBEDDINGS = _BASE_RES_PATH + 'model_word_embeddings.json'
        self.EMBEDDINGS_VEC = _BASE_RES_PATH + 'embeddings.vec'
        self.EMBEDDING_MATRIX = _BASE_RES_PATH + 'embedding_matrix.vec.matrix'

        #MODEL AND WEIGHTS
        self.MODEL = _BASE_RES_PATH + 'model.json'
        self.WEIGHTS = _BASE_RES_PATH + 'weights.h5'

        #PATH FILE EVALUATION INPUT X
        self.INPUT_X_EVALUATION_PATH = _BASE_RES_PATH + 'evaluation_input_x.dat'

        #PATH FILE PREDICTIONS OUTPUT
        self.PREDICT_2007_BABELNET_PATH = _BASE_RES_PRED_PATH + '{}babelnet.key'.format(self.SEMEVAL2007)
        self.PREDICT_2007_DOMAINS_PATH = _BASE_RES_PRED_PATH + '{}domains.key'.format(self.SEMEVAL2007)
        self.PREDICT_2007_LEXNAMES_PATH = _BASE_RES_PRED_PATH + '{}lexnames.key'.format(self.SEMEVAL2007)

        # PATH FILE PREDICTION 2013
        self.PREDICT_2013_BABELNET_PATH = _BASE_RES_PRED_PATH + '{}babelnet.key'.format(self.SEMEVAL2013)
        self.PREDICT_2013_DOMAINS_PATH = _BASE_RES_PRED_PATH + '{}domains.key'.format(self.SEMEVAL2013)
        self.PREDICT_2013_LEXNAMES_PATH = _BASE_RES_PRED_PATH + '{}lexnames.key'.format(self.SEMEVAL2013)

        # PATH FILE PREDICTION 2013
        self.PREDICT_2015_BABELNET_PATH = _BASE_RES_PRED_PATH + '{}babelnet.key'.format(self.SEMEVAL2015)
        self.PREDICT_2015_DOMAINS_PATH = _BASE_RES_PRED_PATH + '{}domains.key'.format(self.SEMEVAL2015)
        self.PREDICT_2015_LEXNAMES_PATH = _BASE_RES_PRED_PATH + '{}lexnames.key'.format(self.SEMEVAL2015)

        # PATH FILE PREDICTION 2013
        self.PREDICT_2_BABELNET_PATH = _BASE_RES_PRED_PATH + '{}babelnet.key'.format(self.SENSEVAL2)
        self.PREDICT_2_DOMAINS_PATH = _BASE_RES_PRED_PATH + '{}domains.key'.format(self.SENSEVAL2)
        self.PREDICT_2_LEXNAMES_PATH = _BASE_RES_PRED_PATH + '{}lexnames.key'.format(self.SENSEVAL2)

        # PATH FILE PREDICTION 2013
        self.PREDICT_3_BABELNET_PATH = _BASE_RES_PRED_PATH + '{}babelnet.key'.format(self.SENSEVAL3)
        self.PREDICT_3_DOMAINS_PATH = _BASE_RES_PRED_PATH + '{}domains.key'.format(self.SENSEVAL3)
        self.PREDICT_3_LEXNAMES_PATH = _BASE_RES_PRED_PATH + '{}lexnames.key'.format(self.SENSEVAL3)

        # PATH FILE EVALUATION GOLD 2007
        self.EVALUATION_2007_GOLD_BABELNET_PATH = _BASE_RES_EVAL_PATH + '{}babelnet.gold.key.txt'.format(self.SEMEVAL2007)
        self.EVALUATION_2007_GOLD_DOMAINS_PATH = _BASE_RES_EVAL_PATH + '{}domains.gold.key.txt'.format(self.SEMEVAL2007)
        self.EVALUATION_2007_GOLD_LEXNAMES_PATH = _BASE_RES_EVAL_PATH + '{}lexnames.gold.key.txt'.format(self.SEMEVAL2007)

        # PATH FILE EVALUATION GOLD 2013
        self.EVALUATION_2013_GOLD_BABELNET_PATH = _BASE_RES_EVAL_PATH + '{}babelnet.gold.key.txt'.format(self.SEMEVAL2013)
        self.EVALUATION_2013_GOLD_DOMAINS_PATH = _BASE_RES_EVAL_PATH + '{}domains.gold.key.txt'.format(self.SEMEVAL2013)
        self.EVALUATION_2013_GOLD_LEXNAMES_PATH = _BASE_RES_EVAL_PATH + '{}lexnames.gold.key.txt'.format(self.SEMEVAL2013)

        # PATH FILE EVALUATION GOLD 2015
        self.EVALUATION_2015_GOLD_BABELNET_PATH = _BASE_RES_EVAL_PATH + '{}babelnet.gold.key.txt'.format(self.SEMEVAL2015)
        self.EVALUATION_2015_GOLD_DOMAINS_PATH = _BASE_RES_EVAL_PATH + '{}domains.gold.key.txt'.format(self.SEMEVAL2015)
        self.EVALUATION_2015_GOLD_LEXNAMES_PATH = _BASE_RES_EVAL_PATH + '{}lexnames.gold.key.txt'.format(self.SEMEVAL2015)

        # PATH FILE EVALUATION GOLD 2
        self.EVALUATION_2_GOLD_BABELNET_PATH = _BASE_RES_EVAL_PATH + '{}babelnet.gold.key.txt'.format(self.SENSEVAL2)
        self.EVALUATION_2_GOLD_DOMAINS_PATH = _BASE_RES_EVAL_PATH + '{}domains.gold.key.txt'.format(self.SENSEVAL2)
        self.EVALUATION_2_GOLD_LEXNAMES_PATH = _BASE_RES_EVAL_PATH + '{}lexnames.gold.key.txt'.format(self.SENSEVAL2)

        # PATH FILE EVALUATION GOLD 3
        self.EVALUATION_3_GOLD_BABELNET_PATH = _BASE_RES_EVAL_PATH + '{}babelnet.gold.key.txt'.format(self.SENSEVAL3)
        self.EVALUATION_3_GOLD_DOMAINS_PATH = _BASE_RES_EVAL_PATH + '{}domains.gold.key.txt'.format(self.SENSEVAL3)
        self.EVALUATION_3_GOLD_LEXNAMES_PATH = _BASE_RES_EVAL_PATH + '{}lexnames.gold.key.txt'.format(self.SENSEVAL3)

