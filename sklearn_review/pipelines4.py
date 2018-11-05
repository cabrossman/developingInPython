#https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines
#http://queirozf.com/entries/scikit-learn-pipeline-examples
import os
dirr = 'C:\\Users\\chrisb\\OneDrive - Leesa\\jobs\\developingInPython\\sklearn_review\\'
os.chdir(dirr)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier


class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]

class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


class PreProcessing(BaseEstimator, TransformerMixin):
    """
       does text processing and creates new columns
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dfcopy = X.copy()
        dfcopy.dropna(axis=0)
        dfcopy.set_index('id', inplace = True)
        #lowering and removing punctuation
        dfcopy['processed'] = dfcopy['text'].apply(lambda x: re.sub(r'[^\w\s]','', x.lower()))
        #numerical feature engineering
        #total length of sentence
        dfcopy['length'] = dfcopy['processed'].apply(lambda x: len(x))
        #get number of words
        dfcopy['words'] = dfcopy['processed'].apply(lambda x: len(x.split(' ')))
        dfcopy['words_not_stopword'] = dfcopy['processed'].apply(lambda x: len([t for t in x.split(' ') if t not in stopWords]))
        #get the average word length
        dfcopy['avg_word_length'] = dfcopy['processed'].apply(lambda x: np.mean([len(t) for t in x.split(' ') if t not in stopWords]) if len([len(t) for t in x.split(' ') if t not in stopWords]) > 0 else 0)
        #get the average word length
        dfcopy['commas'] = dfcopy['text'].apply(lambda x: x.count(','))
        dfcopy = dfcopy.drop('text', axis =1)
        return(dfcopy)

class PostProcessing(BaseEstimator, TransformerMixin):
    """
       converts to dense representation for keras
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return(X.todense())

def create_model(optimizer='adagrad',kernel_initializer='glorot_uniform', dropout=0.2):
    model = Sequential()
    model.add(Dense(64,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(3,activation='sigmoid',kernel_initializer=kernel_initializer))
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model

#get data
df = pd.read_csv('pipelines2data/train.csv')
#split data
X_train, X_test, y_train, y_test = train_test_split(df.drop('author',axis=1), df['author'], test_size=0.33, random_state=42)

#m.fit(x=new_x.todense(), y = pd.get_dummies(y_train).values)
clf = KerasClassifier(build_fn=create_model, epochs = 8)

"""
Pipelines -- each will return a matrix/df with equal number of observations
"""

text = Pipeline([
                ('selector', TextSelector(key='processed')),
                ('tfidf', TfidfVectorizer( stop_words='english', max_df = .95))
            ])
length =  Pipeline([
                ('selector', NumberSelector(key='length')),
                ('standard', StandardScaler())
            ])
words =  Pipeline([
                ('selector', NumberSelector(key='words')),
                ('standard', StandardScaler())
            ])
words_not_stopword =  Pipeline([
                ('selector', NumberSelector(key='words_not_stopword')),
                ('standard', StandardScaler())
            ])
avg_word_length =  Pipeline([
                ('selector', NumberSelector(key='avg_word_length')),
                ('standard', StandardScaler())
            ])
commas =  Pipeline([
                ('selector', NumberSelector(key='commas')),
                ('standard', StandardScaler()),
            ])

"""
Now cBind them together with Feature Uninion and push that too to a pipeline
"""
feats = FeatureUnion([('text', text), 
                      ('length', length),
                      ('words', words),
                      ('words_not_stopword', words_not_stopword),
                      ('avg_word_length', avg_word_length),
                      ('commas', commas)])

"""
add classifier at end
"""
pipeline = Pipeline([
    ('preprocessing', PreProcessing()), #first prep
    ('features',feats), #all pull from pre, then union
    ('PostProcessing', PostProcessing()), # convert X matrix to dense representation
    ('clf',clf) #keras regressor
])

pipeline.fit(X_train, pd.get_dummies(y_train).values)
preds_mat = pipeline.predict(X_test)
#convert to catgories
preds = np.choose(preds_mat, np.unique(y_train))
np.mean(preds == y_test)


"""
lets tune the pipeline hyper params
"""
#how many possbile tuning features
pipeline.get_params().keys()

hyperparameters = { 'features__text__tfidf__ngram_range': [(1,1), (1,2)],
                    'clf__epochs':[8],
                  }

clf = GridSearchCV(pipeline, hyperparameters, cv=3)

# Fit and tune model
clf.fit(X_train, pd.get_dummies(y_train).values)

clf.refit
preds = clf.predict(X_test)
preds = np.choose(preds_mat, np.unique(y_train))
probs = clf.predict_proba(X_test)
np.mean(preds == y_test)