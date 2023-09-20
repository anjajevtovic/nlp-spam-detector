
import string
import json

import pandas as pd
import numpy as np

from nltk import download
from nltk.corpus import stopwords

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from tensorflow import argmax
from tensorflow import string as tstr
import tensorflow.keras as keras
from tensorflow.keras import layers
from keras_nlp.models import BertClassifier

from xgboost import XGBClassifier


BATCH_SIZE = 32
EPOCHS     = 5

results = {}


def evaluate_model(model, x, y, bert=0):
    if bert==0:
        y_preds = np.round(model.predict(x))
    else:
        y_preds = argmax(model.predict(x), axis=-1).numpy()

    accuracy = accuracy_score(y, y_preds)
    precision = precision_score(y, y_preds)
    recall = recall_score(y, y_preds)
    f1 = f1_score(y, y_preds)
    
    model_results_dict = {'accuracy':accuracy,
                         'precision':precision,
                         'recall':recall,
                         'f1-score':f1}
    
    return model_results_dict


def text_preprocess(message):
    nopunc = [char for char in message if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    nopunc = nopunc.lower()

    nostop = [
        word
        for word in nopunc.split()
        if word.lower() not in stopwords.words("english") and word.isalpha()
    ]

    return nostop


def load_data(dataset_size='small'):
    df = pd.read_csv("./data/enron_spam_data.csv")
    # nltk.download('stopwords')

    # print(data.head())
    # print(data.columns)

    # bar = data['Spam/Ham'].value_counts().plot(kind='bar',
    #                                         figsize=(10,10),
    #                                         title='Spam/Ham Ratio')
    # plt.axes(bar)
    # # plt.show()

    data = df[["Message", "Spam/Ham"]]
    data.rename(columns={"Message": "message", "Spam/Ham": "label"}, inplace=True)
    data.label = data.label.apply(lambda x: 1 if x == "ham" else 0)
    data.dropna(inplace=True)
    
    if dataset_size=='standard':
        data.message = data.message.apply(text_preprocess)
        data.message = data.message.agg(lambda x: " ".join(map(str, x)))

        return data
    elif dataset_size=='small':

        # print(df.head())
        # print(df.columns)

        ham = data[data.label == 1].copy()
        spam = data[data.label == 0].copy()

        small_df = ham.head(3000)
        small_df = pd.concat([small_df, spam.head(3000)], axis=0)
        small_df = small_df.reset_index()

        small_df.message = small_df.message.apply(text_preprocess)
        small_df.message = small_df.message.agg(lambda x: " ".join(map(str, x)))

        return small_df
    else:
        return None


def untrained_bert(x_test, y_test):
    classifier = BertClassifier.from_preset("bert_tiny_en_uncased", num_classes=2)
    loss, accuracy = classifier.evaluate(x_test, y_test)
    results['untrained_bert'] = [loss, accuracy]


def xgboost(x_train, y_train, x_test, y_test):
    # BoW
    count_vectorizer = CountVectorizer().fit(x_train)
    count_matrix_train, count_matrix_test = count_vectorizer.transform(x_train), count_vectorizer.transform(x_test)

    # Tf-idf
    tfidf_transformer = TfidfTransformer().fit(count_matrix_train)
    x_train_vec, x_test_vec = tfidf_transformer.transform(count_matrix_train), tfidf_transformer.transform(count_matrix_test)

    classifier = XGBClassifier()

    classifier.fit(x_train_vec, y_train)
    # predict_train = classifier.predict(x_train_vec)

    # print(f"Accuracy of Train dataset: {accuracy_score(y_train, predict_train):0.3f}")
    # label_predictions = classifier.predict(x_test_vec)
    # print(f"Accuracy of the model: {accuracy_score(y_test, label_predictions):0.3f}")
    # print(classification_report(y_test, label_predictions))

    results['xgboost'] = evaluate_model(classifier, x_test_vec, y_test)


def bi_lstm(x_train, y_train, x_test, y_test):
    text_words_lengths = [len(data.loc[i]['message'].split()) for i in range(0, len(data))]
    total_length = np.sum(text_words_lengths)
    text_words_mean = int(np.mean(text_words_lengths))
    #print(text_words_lengths[:5], total_length, text_words_mean)  #[195, 5, 105, 95, 52] 434615 108

    MAXTOKENS  = total_length
    OUTPUTLEN  = text_words_mean

    text_vec = layers.TextVectorization(
        max_tokens = MAXTOKENS,
        output_mode ='int',
        output_sequence_length = OUTPUTLEN
    )

    text_vec.adapt(x_train)

    embedding_layer = layers.Embedding(
        input_dim = MAXTOKENS,
        output_dim = 128,
        embeddings_initializer = 'uniform',
        input_length = OUTPUTLEN
    )
    
    input_layer = layers.Input(shape=(1,), dtype=tstr)
    vec_layer = text_vec(input_layer)
    embedding_layer_model = embedding_layer(vec_layer)
    bi_lstm = layers.Bidirectional(layers.LSTM(64, activation='tanh', return_sequences=True))(embedding_layer_model)
    lstm = layers.Bidirectional(layers.LSTM(64))(bi_lstm)
    flatten = layers.Flatten()(lstm)
    dropout = layers.Dropout(.1)(flatten)
    x = layers.Dense(32, activation='relu')(dropout)
    output_layer = layers.Dense(1, activation='sigmoid')(x)
    bi_lstm_model = keras.Model(input_layer, output_layer)

    bi_lstm_model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    
    history = bi_lstm_model.fit(x_train,
             y_train,
             epochs=EPOCHS,
             validation_data=(x_test, y_test),
             validation_steps=int(0.2*len(x_test)))

    loss, accuracy = bi_lstm_model.evaluate(x_test, y_test)

    results['bi-lstm'] = evaluate_model(bi_lstm_model, x_test, y_test)


def pretrained_bert(x_train, y_train, x_test, y_test):

    classifier = BertClassifier.from_preset(
        "bert_tiny_en_uncased", num_classes=2, activation="softmax"
    )

    history = classifier.fit(x_train,
             y_train,
             batch_size=BATCH_SIZE,
             epochs=EPOCHS,
             validation_data=(x_test, y_test),
             validation_steps=int(0.2*len(x_test)))

    loss, accuracy = classifier.evaluate(x_test, y_test)

    classifier.save('./models/bert-classifier.keras')

    results['bert'] = evaluate_model(classifier, x_test, y_test, bert=1)


data = load_data()

x, y = np.asanyarray(data['message']), np.asanyarray(data['label'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)

untrained_bert(x_test, y_test)
xgboost(x_train, y_train, x_test, y_test)
bi_lstm(x_train, y_train, x_test, y_test)
pretrained_bert(x_train, y_train, x_test, y_test)
print(results)

with open('final_results.json', 'w') as opt:
    json.dump(results, opt)
