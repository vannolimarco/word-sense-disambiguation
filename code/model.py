import numpy as np
import nltk
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from sklearn.model_selection import train_test_split
import pickle
from keras import optimizers
import matplotlib.pyplot as plt
from google.colab import drive
from google.colab import files
from gensim.models import KeyedVectors
import math
import io
import os
from gensim.models import Word2Vec

import warnings
import os
from tensorflow.keras.backend import set_session


""" ONLY READ
in this file is possible see the model used to prediction. It follows a multitask model.
It was runned in Google Colab and many row of code were written to read file with google colab (drive)"""

#######################         MODEL MULTITASK        ###############################

######################################################################################
#######################GLOBAL VARIABLE: PATH OF FILE FOR TRAINING#####################
######################################################################################


nltk.download('punkt')
drive.mount('/Colab_Notebooks')


'/Colab_Notebooks'

corpus_x = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources", 'corpus_x.txt')
path_vocabolary_input_x = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                       'input_x_vocabolary.dat')
path_vocabolary_babelnet = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                        'babelnet_vocabolary.dat')
path_vocabolary_domains = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                       'domains_vocabolary.dat')
path_vocabolary_lexnames = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                        'lexnames_vocabolary.dat')

# training data
path_training_x = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                               'training_x.dat')
path_training_babelnet_y = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                        'training_y_babelnet.dat')
path_training_domains_y = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                       'training_y_domains.dat')
path_training_lexnames_y = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                        'training_y_lexnames.dat')

# testing data
path_training_dev_x = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                   'training_x_dev.dat')
path_training_babelnet_dev_y = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                            'training_y_babelnet_dev.dat')
path_training_domains_dev_y = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                           'training_y_domains_dev.dat')
path_training_lexnames_dev_y = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                            'training_y_lexnames_dev.dat')

######################################################################################
###############################GLOBAL VARIABLE: DATA##################################
######################################################################################

# MODEL AND WORD EMEBEDDINGS
path_resources_model_we = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                       'model_word_embeddings.json')
path_embeddings_vec = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                   'embeddings.vec')
path_embeddings_matrix = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                      'embedding_matrix.vec.matrix')

# VOCABA ND MATRIX EMBEDDINGS
model = KeyedVectors.load(path_resources_model_we)
# model = Word2Vec.load(resources_path_model_we)      #loads the embeddings sense
# embedding_vocab_size = model.wv.vectors.shape[0]
embedding_matrix = pickle.load(open(path_embeddings_matrix, "rb"))

vocab_input_x = pickle.load(open(path_vocabolary_input_x, "rb"))
vocab_babelnet = pickle.load(open(path_vocabolary_babelnet, "rb"))
vocab_domains = pickle.load(open(path_vocabolary_domains, "rb"))
vocab_lexnames = pickle.load(open(path_vocabolary_lexnames, "rb"))

## DATA TRAINING
x_train = pickle.load(open(path_training_x, "rb"))
y_train_babelnet = pickle.load(open(path_training_babelnet_y, "rb"))
y_train_domains = pickle.load(open(path_training_domains_y, "rb"))
y_train_lexnames = pickle.load(open(path_training_lexnames_y, "rb"))

# DATA TESTING
x_dev = pickle.load(open(path_training_dev_x, "rb"))
y_dev_babelnet = pickle.load(open(path_training_babelnet_dev_y, "rb"))
y_dev_domains = pickle.load(open(path_training_domains_dev_y, "rb"))
y_dev_lexnames = pickle.load(open(path_training_lexnames_dev_y, "rb"))

# SAVE MODEL
resources_path_model = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                    'model.json')
resources_path_weights = os.path.join("/Colab_Notebooks/My Drive/public_homework_3/public_homework_3/resources",
                                      'weights.h5')

#######################################################################################
###############################GLOBAL VARIABLE OF MODEL################################
#######################################################################################

# DEFINE SOME COSTANTS
MAX_LENGTH = 120
VOCAB_SIZE = len(vocab_input_x) + 1
EMBEDDING_SIZE = model.wv.vectors.shape[1]
HIDDEN_SIZE = 300
EPOCHS = 35

#######################################################################################
###############################METHODS OF MODEL########################################
#######################################################################################


LOSS_LIST = ["sparse_categorical_crossentropy", 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy']

METRICS = {'babelnet': 'accuracy', 'domains': 'accuracy', 'lexnames': 'accuracy'}


# method to train model.


def train_model():

    #to use GPU on colab
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    # to log device placement (on which device the operation ran)
    config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    # set this TensorFlow session as the default session for Keras
    set_session(sess)

    batch_size = 32
    model = create_multi_task_model_keras(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, LOSS_LIST, METRICS)

    model.summary()

    tensorboard = K.callbacks.TensorBoard("logging/keras_model", histogram_freq=50)

    print("\nStarting training...")

    checkpoint = K.callbacks.ModelCheckpoint(resources_path_model, monitor='val_loss', verbose=1, save_best_only=True,
                                             mode='min')
    cbk = [checkpoint, tensorboard]

    history = model.fit(x_train, [np.expand_dims(y_train_babelnet, axis=2), np.expand_dims(y_train_domains, axis=2),
                                  np.expand_dims(y_train_lexnames, axis=2)],
                        epochs=EPOCHS, shuffle=True, batch_size=batch_size, validation_data=(x_dev, [
            np.expand_dims(y_dev_babelnet, axis=2), np.expand_dims(y_dev_domains, axis=2),
            np.expand_dims(y_dev_lexnames, axis=2)]), callbacks=cbk)

    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy babelnet
    plt.plot(history.history['babelnet_acc'])
    plt.plot(history.history['val_babelnet_acc'])
    plt.title('model accuracy babelnet')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss babelnet
    plt.plot(history.history['babelnet_loss'])
    plt.plot(history.history['val_babelnet_loss'])
    plt.title('model loss babelnet')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for accuracy domains
    plt.plot(history.history['domains_acc'])
    plt.plot(history.history['val_domains_acc'])
    plt.title('model accuracy domains')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss domains
    plt.plot(history.history['domains_loss'])
    plt.plot(history.history['val_domains_loss'])
    plt.title('model loss domains')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for accuracy lexnames
    plt.plot(history.history['lexnames_acc'])
    plt.plot(history.history['val_lexnames_acc'])
    plt.title('model accuracy lexnames')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss lexnames
    plt.plot(history.history['lexnames_loss'])
    plt.plot(history.history['val_lexnames_loss'])
    plt.title('model loss lexnames')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # save model and wheights of model (json)
    model_json = model.to_json()
    with open(resources_path_model, "w") as json_file:
        json_file.write(model_json)

        print("Saved model to file (format json): " + resources_path_model)

    # serialize weights to HDF5
    model.save_weights(resources_path_weights)
    print("Saved weights of model to file : " + resources_path_weights)


# method to create structure of model.

def create_multi_task_model_keras(vocab_size_input, embedding_size, hidden_size, loss_list, metrics):
    # create sequential model
    print('creation of multi-task model...')

    # add layer for inputs
    x = K.layers.Input(shape=(20,), name='input_x')

    embedding_x = K.layers.Embedding(vocab_size_input, embedding_size, weights=[embedding_matrix], mask_zero=True,
                                     embeddings_initializer='uniform', name='embedding_layer')(x)

    # put the input from embedding layer to bi-LSTM
    lstm_result = (K.layers.Bidirectional(
        K.layers.LSTM(hidden_size, return_sequences=True, dropout=0.8, recurrent_dropout=0.8, name='lstm_layer'),
        name='bidirectional_layer'))(embedding_x)

    #softmax layers for each outputs
    y1 = K.layers.TimeDistributed(
        K.layers.Dense((len(vocab_babelnet) + 1), activation='softmax', name='babelnet_dense'), name='babelnet')(
        lstm_result)
    y2 = K.layers.TimeDistributed(K.layers.Dense((len(vocab_domains) + 1), activation='softmax', name='domains_dense'),
                                  name='domains')(lstm_result)
    y3 = K.layers.TimeDistributed(
        K.layers.Dense((len(vocab_lexnames) + 1), activation='softmax', name='lexnames_dense'), name='lexnames')(
        lstm_result)

    # create model with 1 input and 3 output
    model = K.models.Model(inputs=[x], outputs=[y1, y2, y3])

    #Adam optimizer
    Adam = K.optimizers.Adam(lr=0.001, epsilon=None, decay=0.0, amsgrad=False)

    # compile model and return it!
    model.compile(loss=loss_list, optimizer=Adam, metrics=metrics)

    print('End creation of model...')
    return model


#######################################################################################
###############################TRAIN OF MODEL##########################################
#######################################################################################
print('size vocabolary: ')
print(len(vocab_input_x))
print(EMBEDDING_SIZE)
print('-----------------')
train_model()