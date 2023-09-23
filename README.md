# Spam Email Detection using ML and NN

**Keywords:** *Machine Learning, Neural Networks, XGBoost, Bi-LSTM, BERT Classifier, Enron*

## How to run the code:</br>
```
git clone https://github.com/anjajevtovic/nlp-spam-detector .
cd nlp-spam-detector
pip install -r requirements.txt
python3 spam_detection_models.py -t                 # to train the models
python3 spam_detection_models.py -p [message_text]  # to predict if the message is spam/ham
```

## Introduction
Email service as we know it today was accessible to the public since the 1990s. It took only a couple of
years (1994. The ‘Green Card Spam’ incident is one of the examples) for spam emails to get recognized
as a significant problem. Ever since, email threats evolved and become one of the leading vectors for
cyberattacks. According to Statista, the global spam volume of total email traffic in 2022 was 48.63%,
with the average across the last ten years being around 57%. The term spam in this context translates to
any unsolicited, potentially harmful email.

To see this statistic from another perspective, the number of emails sent and received per day in 2023 is
347.3 billion. And even though it was predicted email would become obsolete by 2020, the statistics show
only growth over the past years.

Electronic mail is a common means of communication due to its convenience and capacity to exchange
and store all kinds of information formats - messages, documents, images, videos, and links. The email
can have multiple recipients, it gets stored in everybody's mailbox waiting to be opened at any time,
making it very time-efficient and cost-effective. It is largely utilized in the business world where spam
emails prove to be the most harmful due to the financial losses they induce. Due to these attributes, email
is a convenient mean to execute scams, unwanted marketing, and other malicious activities.

Spam filters are being utilized as spam detection and prevention systems. The filtering is happening in
several places - on the ISP (internet service provider) side, ESP (email service provider) side, client side,
company side, etc…

Spam filters often combine multiple techniques to improve accuracy and effectiveness. Such techniques
are Content-based filtering, Real-time blackhole lists, Whitelisting, Blacklisting, Machine Learning
Techniques, SPF and DKIM verification and integrity checks, and so on… Shortly put, Knowledge
Engineering and Machine Learning are two popular ways of filtering unsolicited emails.

The downside of Knowledge engineering is that it requires a lot of rules, constant maintenance, and
updates. Hence, Machine Learning proved to be efficient for this purpose as a set of rules is not required,
especially deep learning techniques within Natural Language Processing.

Although both deep learning and NLP have been present and utilized since the 2000s, the computing
power and knowledge we harvest today enabled the development of new techniques, architectures, and
models - giving the opportunity to develop faster, more optimized, and easily customizable solutions.
The aim of this project is to showcase the current state of the Deep Learning field by creating and
evaluating three models (XGBoost, Bi-LSTM, and pre-trained BERT) for detecting spam messages.

## Related Work
Spam detection solutions are a part of standard packages of every corporation and even smaller
companies, therefore, the current state-of-art approach is not accessible via the Internet. Sophisticated
solutions already exist in commercial products using a combination of here mentioned approaches and the
biggest downside of public research is the quality of data used.

However, given the importance of this problem, there are a number of researches covering this topic and
showcasing what that ‘state-of-art’ solution most likely would be using in the backend.

In this regard, [1] did an experiment using an Enron subset of 5171 emails. The idea was to compare
multiple ML and Deep Learning models by recording detailed metrics for each. Accuracy, precision,
recall, and f1 are considered. From ML algorithms, Random Forest has achieved the highest accuracy of
96.8%. For Deep Learning, Keras API was utilized and Bi-LSTM achieved validation accuracy of 98.5%.
It is stated that Bi-LSTM can be considered the best model for email spam detection because it does
sequencing in both directions.

In another research [2] it was concluded that Natural Language Processing enhanced the model's accuracy
and the word embedding concept has been introduced. Pre-trained fine-tuned BERT transformer model is
compared to Deep Neural Network (essentially Bi-LSTM with two Dense layers on top), and also
standard ML algorithms - k-NN and Naive Bayes. The Enron subset is being used with around 5,000
emails. As per [2], accuracy and f1 score were the main metrics for deciding what the top-performing
model is.

Finally, [3] offered a Keggle Notebook showcasing Random Forest, a model containing random text
vectorization, embedding, and Dense layers, a Bi-LSTM model, and a model combining the USE
embedding layer (transfer learning) and Sequential API. It is worth noting that all four models were not
drastically far from each other in accuracy and f1 metrics, however, the USE Transfer Learning model
achieved the best results with Bi-LSTM following.

## Methodology
By observing methodologies from related research it was concluded that this work should go in the
following direction - three models that showed as top performers will be constructed and compared.
Decision trees proved to be more successful in text classification compared to other ML algorithms.
Random Forests are mostly applied, however, XGBoost proved to be slightly more efficient, therefore,
here XGBoost will be constructed as a baseline ML algorithm with Bag-of-Words and TF-IDF
preprocessing. The second model will be Bi-LSTM with vectorization and word embedding built with
Keras. And, finally, as the current state-of-art model when it comes to Natural Language Processing, the
BERT model will be pretrained (with minimal fine-tuning) to see its performance.

### Data Collection
All models are being trained and evaluated on the Enron dataset, where the whole compiled CSV contains
33 716 emails, however, the script has the option to run model training either on the whole dataset or a
smaller version which takes 6000 emails into consideration. Results presented in this work were compiled
based on a smaller dataset. The dataset was divided in such a manner to be balanced - 3000 ham and 3000
spam email messages, and it is worth noting that the initial dataset is balanced as well.

### Architecture
![architecture](https://github.com/anjajevtovic/nlp-spam-detector/blob/main/architecture.jpg)

### Data Pre-processing
Upon data loading, there is initial preprocessing where every message is stripped out of any punctuation,
converted to all lowercase, and stripped out of all stopwords. Afterward, each model has its own
preprocessing methods.

XGBoost is preprocessed with CountVectorizer (Bag-of-Words) and TF-IDF from sci-kit-learn, and
Bi-LSTM on the other hand is preprocessed with TectVectorization and Embedding Keras layers. Finally,
there is no explicit preprocessing for the BERT model since the idea was how it would behave with
minimal fine-tuning. From the preset, BERT is implementing Tokenization, Special Tokens, Padding and
Truncation, Segment IDs, Masking, Vocabulary, Normalization, and Batching.

The second preprocessing is happening after splitting the data into Train (80%) and Test (20%).

#### Stop Word Removal
Stop words are commonly occurring words such as articles, pronouns, and prepositions. The removal
process excludes the unnecessary words of little value and helps us focus more on the text that requires
our attention.

#### Bag-of-Words
Bag of words counts the occurrences of words in the text, disregarding the order of words and the
structure of the document.

#### TF-IDF
The bag of words approach could pose a problem wherein the stop words are assigned a greater frequency
than the informational words. Term Frequency-Inverse Document Frequency (TF-IDF) helps in rescaling
the frequency of words by how often they appear in the texts so that stop words could get penalized. The
TF-IDF technique rewards the frequently occurring words but punishes the too commonly occurring
words in several texts.

#### Word embedding
Word embeddings help in representing individual words as real-valued vectors in the lower dimensional
space. Put simply, it’s the conversion of text to numerical data (vectors) which can facilitate the analysis
by an NLP model.
Unlike BoW, word embeddings use a predefined vector space to map words irrespective of corpus size.
Word embeddings can determine the semantic relationship between words in the text, whereas BoW
cannot.

### Models

#### XGBoost Model
XGBoost stands for ‘Extreme Gradient Boosting’ and it represents an efficient and scalable approach for
training machine learning models. The idea is to combine the predictions from multiple weak models to
produce a stronger one. It has become one of the most popular ML algorithms due to its ability to handle
large datasets and achieve very good results in classification and regression tasks.

#### Bi-LSTM Model
A Bidirectional LSTM is a sequence processing model that consists of two LSTMs: one taking the input
in a forward direction, and the other in a backwards direction. Bi-LSTMs effectively increase the amount
of information available to the network, improving the context available to the algorithm (e.g. knowing
what words immediately follow and precede a word in a sentence).

#### BERT Model
BERT classifier is a pre-trained model published by Google. There are quite a few models one can choose
from based on a task that should be solved, the size of the corpus, and other factors, and here the BERT is
the most comprehensive one. Bert model was trained on English Wikipedia with 2500 million words and
BookCorpus of 800 million words. It uses attention models to learn the contextual relation between the
words in a sentence. It essentially consisted of two parts - an encoder and a decoder.

## Results and Analysis
Data was split into training/tests with a ratio of 4/1 (80/20%). For evaluation, cross-validation was used
since there is no other suitable dataset publicly available. Accuracy, Precision, Recall, and F1-Score metrics
were recorded for each model (Table 1). But Accuracy and F1-Score are the ones taken into
consideration. Accuracy is measuring intuitive interpretation, how classes are balanced and simple,
whereas F1-Score is showing how balanced are precision and recall, and how class imbalance is being
handled.

|Model|Accuracy|Precision|Recall|F1-Score|
|------------|------------|------------|------------|------------|
|XGBoost|0.97|0.986|0.954|0.97|
|Bi-LSTM|0.98|0.977|0.983|0.98|
|BERT|0.985|0.988|0.981|0.985|

The first thing observed is that Accuracy and F1-Score are almost the same (longer decimals are
different), this typically suggests that the model is performing well in terms of overall classification.
Based on previous research, it was expected that all three models should yield close metrics. However, it
is somewhat unexpected that the BERT model, only fitted with spam email dataset in 5 epochs, without
any further fine-tuning yielded the best results out of three. Another thing to note is that the results of this
work do not extremely deviate in metrics values that similar models produced in other research.

## Conclusion and Future Work
This paper tried to observe the latest ML/Deep Learning standards when it comes to text processing and
classification by building spam detection models based on the top three performing algorithms. Although
the BERT model performed the best (Accuracy: 0.985, F1-Score: 0.985), it is notable that all models had
similarly processed data and that picking one model as the best cannot be done. Best results
seem to be generated by combining multiple approaches and devoting time to understanding the
problem, dataset, and preprocessing it requires.
This work can be further extended by additinal fine-tuning Bi-LSTM and BERT models (batch size, number
of epochs, loss function) and also by sampling up-to-date datasets.

## References
[1] P. Malhotra and S. Malik, “Spam Email Detection using Machine Learning and Deep Learning
Techniques” in Social Science Research Network, 2022 </br>
[2] I. AbdulNabi and Q. Yaseen, “Spam email detection using deep learning techniques”, in Procedia
Computer Science, 2021</br>
[3] K. Kouhyar, “SMS Spam Detection(~99% Acc) with Bi-LSTM/USE/RFC”, [Online], 2021,
Available:
https://www.kaggle.com/code/cyruskouhyar/sms-spam-detection-99-acc-with-bi-lstm-use-rfc#Conclusion-%E2%9A%A1</br>
[4] S. Alla, “Building Your First NLP Application to Detect SPAM”, [Online], 2021, Available:
https://blog.paperspace.com/nlp-spam-detection-application-with-scikitlearn-XGBoost/</br>
[5] Dataset Reference: https://github.com/MWiechmann/enron_spam_data