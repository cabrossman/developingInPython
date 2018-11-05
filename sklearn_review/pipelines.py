#https://www.kaggle.com/sermakarevich/sklearn-pipelines-tutorial/notebook

import os
dirr = 'C:\\Users\\chrisb\\OneDrive - Leesa\\jobs\\developingInPython\\sklearn_review\\'
os.chdir(dirr)

import pandas as pd
import numpy as np
from scipy import sparse

from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import roc_auc_score

# get data
df = pd.read_csv('all/train.csv')
x = df['comment_text'].values[:5000]
y = df['toxic'].values[:5000]

# default params
#SCORING =['roc_auc','f1'] use cross_validate
SCORING ='roc_auc'
CV =3
N_JOBS =-1
MAX_FEATURES = 2500

tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
lr = LogisticRegression()
p = Pipeline([
    ('tfidf', tfidf),
    ('lr', lr)
])

cross_val_score(estimator=p, X=x, y=y, scoring=SCORING, cv=CV, n_jobs=N_JOBS)
#cross_validate(estimator=p, X=x, y=y, scoring=SCORING, cv=CV, n_jobs=N_JOBS) #uses multiple

"""
Lets create or own Estimator to reproduce Jeremy`s notebook in pipelines. 
This estimator is created with sklearn BaseEstimator class and needs to have fit and transform methods. 
First Pipeline callss fit methods to learn your dataset and then calls transform to apply knowledge and 
does some transformations.
"""

class NBFeaturer(BaseEstimator):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def preprocess_x(self, x, r):
        return x.multiply(r)
    
    def pr(self, x, y_i, y):
        p = x[y==y_i].sum(0)
        return (p+self.alpha) / ((y==y_i).sum()+self.alpha)

    def fit(self, x, y=None):
        self._r = sparse.csr_matrix(np.log(self.pr(x,1,y) / self.pr(x,0,y)))
        return self
    
    def transform(self, x):
        x_nb = self.preprocess_x(x, self._r)
        return x_nb

tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
lr = LogisticRegression()
nb = NBFeaturer(1)
p = Pipeline([
    ('tfidf', tfidf),
    ('nb', nb),
    ('lr', lr)
])

cross_val_score(estimator=p, X=x, y=y, scoring=SCORING, cv=CV, n_jobs=N_JOBS)

"""Lets add one more custom Estimator to our pipeline, called Lemmatizer"""

class Lemmatizer(BaseEstimator):
    def __init__(self):
        self.l = WordNetLemmatizer()
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x = map(lambda r:  ' '.join([self.l.lemmatize(i.lower()) for i in r.split()]), x)
        x = np.array(list(x))
        return x

lm = Lemmatizer()
tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
lr = LogisticRegression()
nb = NBFeaturer(1)
p = Pipeline([
    ('lm', lm),
    ('tfidf', tfidf),
    ('nb', nb),
    ('lr', lr)
])

cross_val_score(estimator=p, X=x, y=y, scoring=SCORING, cv=CV, n_jobs=N_JOBS)

"""
Pipelines also allow you to process different features in a different way and then concat the result. 
FeatureUnion halps us with this. Lets create additional tfidf vectorizer for chars and join its results with 
words vectorizer.
"""

max_features = 2500
lm = Lemmatizer()
tfidf_w = TfidfVectorizer(max_features=MAX_FEATURES, analyzer='word')
tfidf_c = TfidfVectorizer(max_features=MAX_FEATURES, analyzer='char')
lr = LogisticRegression()
nb = NBFeaturer(1)
p = Pipeline([
    ('lm', lm),
    ('wc_tfidfs', 
         FeatureUnion([
            ('tfidf_w', tfidf_w), 
            ('tfidf_c', tfidf_c), 
         ])
    ),
    ('nb', nb),
    ('lr', lr)
])

cross_val_score(estimator=p, X=x, y=y, scoring=SCORING, cv=CV, n_jobs=N_JOBS)

"""
Who does not like finetuning? Lets make it simple with pipelines and GridSearchCV/RandomizedSearchCV.
"""
PARAM_GRID = [{
    'wc_tfidfs__tfidf_w__max_features': [2500], 
    'wc_tfidfs__tfidf_c__stop_words': [2500, 5000],
    'lr__C': [3.],
}]

grid = GridSearchCV(estimator=p, cv=CV, n_jobs=N_JOBS, param_grid=PARAM_GRID, scoring=SCORING, 
                            return_train_score=False, verbose=1)
grid.fit(x, y)
grid.cv_results_