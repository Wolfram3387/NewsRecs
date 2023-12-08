import pandas as pd
import numpy as np
import random
import warnings
import time
import datetime
import re
import string
import itertools
import pickle
import joblib
import csv

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from collections import Counter, defaultdict

import tensorflow as tf
from keras.utils import custom_object_scope
from keras.models import load_model
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, Embedding, LSTM
from keras.models import Model
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.regularizers import l2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D, Concatenate
from keras.layers import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Multiply, Dropout, Subtract, Add, Conv2D

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import warnings
from utils.neural_networks.loss import contrastive_loss, euclidean_distance,model
warnings.filterwarnings('ignore')



t = Tokenizer()

def get_distance(user_keywords, article_keywords):
    prediction_data = article_keywords
    prediction_vector = t.texts_to_sequences([prediction_data])
    prediction_vector = pad_sequences(prediction_vector, maxlen=200)

    assistant_data = user_keywords
    assistant_vector = t.texts_to_sequences([assistant_data])
    assistant_vector = pad_sequences(assistant_vector, maxlen=200)

    result = model.predict([prediction_vector, assistant_vector])[0][0]
    return result
