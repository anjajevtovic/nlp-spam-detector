'''
    Author: Anja Jevtovic
    Date: 16.09.2023.
    About:
        This script is creating and saving neural network model for email
        spam detection using pretrained NLP model Bert additionally fitted
        with ENRON spam data over one epoch.
'''

import string
from collections import Counter

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords

import keras_nlp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

import scipy.sparse as sparse

from xgboost import XGBClassifier

BATCH_SIZE = 16

def load_data(dataset_size='standard'):
    data = pd.read_csv("./data/enron/preprocessed/enron_spam_data.csv")

    # print(data.head())
    # print(data.columns)

    # bar = data['Spam/Ham'].value_counts().plot(kind='bar',
    #                                         figsize=(10,10),
    #                                         title='Spam/Ham Ratio')
    # plt.axes(bar)
    # # plt.show()

    df = data[["Message", "Spam/Ham"]]
    df.rename(columns={"Message": "message", "Spam/Ham": "label"}, inplace=True)
    df.label = df.label.apply(lambda x: 1 if x == "ham" else 0).copy()
    df.dropna(inplace=True)
    
    if dataset_size=='standard':
        return df
    else:

        # print(df.head())
        # print(df.columns)

        ham = df[df.label == 1].copy()
        spam = df[df.label == 0].copy()

        small_df = ham.head(2000)
        small_df = pd.concat([small_df, spam.head(2000)], axis=0)
        small_df = small_df.reset_index()

        return small_df


def untrained_bert_classifier(df):
    msg_train, msg_test, label_train, label_test = train_test_split(
        df.message, df.label, test_size=0.2
    )
    classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased", num_classes=2)
    classifier.evaluate(msg_test, label.test)


def pretrained_bert_classifier(df):
    msg_train, msg_test, label_train, label_test = train_test_split(
        df.message, df.label, test_size=0.2
    )

    classifier = keras_nlp.models.BertClassifier.from_preset(
        "bert_tiny_en_uncased", num_classes=2, activation="softmax"
    )

    classifier.fit(
        msg_train,
        label_train,
        batch_size=BATCH_SIZE,
        validation_data=msg_test,
        epochs=1,
    )

    classifier.evaluate(msg_test, label_test)
    classifier.save('./models/pretrained-bert-classifier.keras')


def text_preprocess(message):
    # Remove punctuations
    nopunc = [char for char in message if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    nopunc = nopunc.lower()

    nostop = [
        word
        for word in nopunc.split()
        if word.lower() not in stopwords.words("english") and word.isalpha()
    ]

    return nostop


def preprocessed_xgboost(df):
    # top 10 occuring words in spam emails
    # nltk.download('stopwords')

    # ham_corpus = Counter()

    # for msg in ham_small.message:
    #     ham_corpus.update(text_preprocess(msg))

    # print(f'Top 10 occuring words in ham emails are: {ham_corpus.most_common(10)}')

    # spam_corpus = Counter()

    # for msg in spam_small.message:
    #     spam_corpus.update(text_preprocess(msg))

    # print(f'Top 10 occuring words in spam emails are: {spam_corpus.most_common(10)}')

    df.message = df.message.apply(text_preprocess).copy()
    print(df.head())

    # Convert messages (as lists of string tokens) to strings
    df.message = df.message.agg(lambda x: " ".join(map(str, x)))
    print(df.head())

    # Initialize count vectorizer
    vectorizer = CountVectorizer()
    bow_transformer = vectorizer.fit(df.message)

    # Fetch the vocabulary set
    print(f"Total number of vocab words: {len(vectorizer.vocabulary_)}")
    # Convert strings to vectors using BoW
    messages_bow = bow_transformer.transform(df.message)

    # Print the shape of the sparse matrix and count the number of non-zero occurrences
    print(f"Shape of sparse matrix: {messages_bow.shape}")
    print(f"Amount of non-zero occurrences: {messages_bow.nnz}")

    tfidf_transformer = TfidfTransformer().fit(messages_bow)

    # Transform entire BoW into tf-idf corpus
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    print(messages_tfidf.shape)


    # Convert spam and ham labels to 0 and 1 (or, vice-versa)
    FactorResult = pd.factorize(df.label)
    df.label = FactorResult[0]
    df.head()
    print(messages_tfidf)

    sparse.save_npz('./models/preprocessed_enron_sparse_matrix.npz', messages_tfidf)

    # Split the dataset to train and test sets
    msg_train, msg_test, label_train, label_test = train_test_split(
        messages_tfidf, df.label, test_size=0.2
    )

    clf = XGBClassifier()

    clf.fit(msg_train, label_train)
    predict_train = clf.predict(msg_train)

    print(
        f"Accuracy of Train dataset: {metrics.accuracy_score(label_train, predict_train):0.3f}"
    )

    print(
        "predicted:",
        clf.predict(
            tfidf_transformer.transform(bow_transformer.transform([df.message[9]]))
        )[0],
    )
    print("expected:", df["label"][9])
    # print the overall accuracy of the model
    label_predictions = clf.predict(msg_test)
    print(f"Accuracy of the model: {metrics.accuracy_score(label_test, label_predictions):0.3f}")


def compile_model(model):
    '''
    simply compile the model with adam optimzer
    '''
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy']) 


def fit_model(model, epochs, X_train, y_train, X_test, y_test):
    '''
    fit the model with given epochs, train and test data
    '''
    history = model.fit(X_train,
              y_train,
             epochs=epochs,
             validation_data=(X_test, y_test),
             validation_steps=int(0.2*len(X_test)))
    return history


def bi_ltsm_network_from_scratch(df):

    text_words_lengths = [len(df.loc[i]['message'].split()) for i in range(0, len(df))]
    total_length = np.sum(text_words_lengths)
    text_words_mean = int(np.mean(text_words_lengths))

    print(f'total len: {total_length}, text words mean: {text_words_mean}')

    X, y = np.asanyarray(df.message), np.asanyarray(df.label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

    MAXTOKENS = total_length
    OUTPUTLEN = text_words_mean

    text_vec = layers.TextVectorization(
        max_tokens=MAXTOKENS,
        standardize='lower_and_strip_punctuation',
        output_mode='int',
        output_sequence_length=OUTPUTLEN
    )

    text_vec.adapt(X_train)

    embedding_layer = layers.Embedding(
        input_dim=MAXTOKENS,
        output_dim=128,
        embeddings_initializer='uniform',
        input_length=OUTPUTLEN
    )

    input_layer = layers.Input(shape=(1,), dtype=tf.string) # Input layer, string type(text)
    vec_layer = text_vec(input_layer) # text vectorization layer(built previous lines)
    embedding_layer_model = embedding_layer(vec_layer) # word embedding layer
    bi_lstm = layers.Bidirectional(layers.LSTM(64, activation='tanh', return_sequences=True))(embedding_layer_model) # Bidirectional-LSTM, 64 units
    lstm = layers.Bidirectional(layers.LSTM(64))(bi_lstm)
    flatten = layers.Flatten()(lstm) # Flatten layer for enering in dense layers
    dropout = layers.Dropout(.1)(flatten) # drop out layer
    x = layers.Dense(32, activation='relu')(dropout) # Dense layer
    output_layer = layers.Dense(1, activation='sigmoid')(x) # output layer
    model_2 = keras.Model(input_layer, output_layer) # final model

    compile_model(model_2) # compile the model

    fit_model(model_2, epochs=5, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test) # fit the model
    model_2.evaluate(X_test, y_test)


# bi_ltsm_network_from_scratch(load_data()) # loss: 0.1096 - accuracy: 0.9588
pretrained_bert_classifier(load_data()) # loss: 0.5374 - accuracy: 0.9663
# preprocessed_xgboost(load_data()) #Accuracy of Train dataset: 0.995 Accuracy of the model: 0.968; Accuracy of Train dataset: 0.987 