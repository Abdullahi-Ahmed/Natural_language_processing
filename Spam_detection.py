import sys
import nltk
import sklearn
import pandas
import numpy

print('Python:{}'.format(sys.version))
print('NLTK:{}'.format(nltk.__version__))
print('Scikit-learn:{}'.format(sklearn.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Numpy:{}'.format(numpy.__version__))

import pandas as pd
import numpy as np
#load the dataset of the sms messages
df = pd.read_table('SMSSpamCollection', header=None, encoding ='utf-8')

#check class distribution
classes =df[0]
classes.value_counts()

from sklearn.preprocessing import LabelEncoder
#convert class labels to binary values, 0 =ham and 1=spam
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)
Y[:10]

#store the SMS message data
#this a series data object
text_messages =df[1]
print(text_messages[:10])

#use regular expressions to replace email addresses, URLs, phone numbers, other numbers

#Replace email addresses with 'email'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                                                                          'emailaddress')

#Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                                                                          'webaddress')

#Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = processed.str.replace(r'£|\$','monrysymb')

#Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                                                                           'phonenumber')

#Replace number with "number"
processed = processed.str.replace(r'\d+(\.\.d+)?', 'number')

#Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')

#Remove whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')



#change words to lower case -Hello,Helloi.e
processed = processed.str.lower()

processed

from nltk.corpus import stopwords

#remove stop words from text messages

stop_words = set(stopwords.words('english'))

processed = processed.apply(lambda x: ' '.join(
     term for term in x.split() if term not in stop_words))

#Remove word stems using a porter stemmer
ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(
      ps.stem(term) for term in x.split()))

import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize

# create bag-of-words
all_words = []
for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)

#print the total number of words and the 15 most common words
print('Number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(15)))

#use the 1500 most common as features
word_features =list(all_words.keys())[:1500]


# the find_features function will determine which of the 1500 word features are contained in the review
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


# lets see an example!
features = find_features(processed[0])
for key, value in features.items():
    if value == True:
        print(key)

#now lets do it for all the messages
messages = list(zip(processed, Y))

#define a seed for reproducibility
seed =1
np.random.seed = seed
np.random.shuffle(messages)

#call find_features function for each SMS message
featuresets =[(find_features(text), label) for (text, label) in messages]

#we can split the featuresets into training and testing datasets using sklearn
from sklearn import model_selection

#split the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)

#we cn use sklearn algorithms in NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model =SklearnClassifier(SVC(kernel = 'linear'))

#train the model on the training data
model.train(training)

#and test on the testing dataset!
accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy: {}".format(accuracy))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#Define models to train
names =["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier","Naive Bayes","SVM Linear"]

Classifier = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter =100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]
models =zip(names, Classifier)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("{} Accuracy: {}".format(name,accuracy))

#Ensemble methods -Voting classifier
from sklearn.ensemble import VotingClassifier

names = ["K Nearest Neighbors" , "Decision Tress", "random Forest", "Logistic Regression", "SGD Classifier",
        "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter =100),
    MultinomialNB(),
    SVC(kernel= 'liner')

]

models = list(zip(names, classifiers))

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting ='hard', n_jobs = -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier: Accuracy:{}".format(accuracy))

#make class label prediction for testing set
txt_features, labels =zip(*testing)

prediction =nltk_ensemble.classify_many(txt_features)
# print a confusion matrix and a classification report
print(classification_report(labels, prediction))

pd.DataFrame(
confusion_matrix(labels, prediction),
index = [['actual','actual'],['ham', 'spam']],
columns=[['predicted','predicted'],['ham','spam']])