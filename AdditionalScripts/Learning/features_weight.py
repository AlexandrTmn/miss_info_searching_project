# -*- coding: utf-8 -*-
"""
Features weighting

Getting weights for features for different models
Tree - decision tree
SVC - Support Vector Classificator
NN - Neural Network (MLP Classifier)
CNB - Naive Bayes Classifier
KNB - K-Neighbors classifier
"""

# imports
from AdditionalScripts.DBScripts.DBConnection import db_connection
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from joblib import load

# ignoring warnings from libraries
warnings.filterwarnings("ignore")

# connecting to database and getting data for experimental metrics
conn = db_connection()

# setup for pandas
pd.set_option("display.max.columns", None)
pd.set_option('display.max_colwidth', None)


# Getting data from database
def get_data():
    for tx in conn.transaction():
        with tx:
            data_json = tx.query_json("""
            select tweet {
            Hand_mark, 
            TMData: {TM_Emotes,TM_Mean,TM_Vulgar,TM_Word_Count,TM_Sentiment}, 
            TMMData: {TM_Low_To_High_Diffustion, TM_Biggest_Connected_Component, TM_Tweets_With_URLs_Part,
            TM_Isolated_Tweets_Part}, 
            UMData: {UM_Originality, UM_Influence, UM_Role, UM_Engagement, UM_Trust},
            EMData: {EM_Hashtags, EM_Links, EM_Repeated_Symbols,EM_Repeated_Words, EM_Spelling,EM_Exclamation_Mark,
            EM_Pronoun, EM_Conditional_Words, EM_Sensory_Verbs,EM_Ordinal_Adjectives, EM_Relative_Time, EM_Numbers_Count
            ,EM_Quantity_Adjective, EM_Certainty_Words, EM_Days_Between}
            };
            """)

            y = json.loads(data_json)
            data_pd = pd.json_normalize(y, max_level=1)
    return data_pd


data = get_data()

# drop nan columns and rows and setting up
data = data.drop(columns=['UMData', 'TMMData', 'TMData'])
data1 = data.dropna()
data1 = data1[0:10000]
# split and scaling
X = data1.drop('Hand_mark', axis=1)
y = data1['Hand_mark']
standardScale = StandardScaler()
label_Encoder = LabelEncoder()

# String to numerical type
X['TMData.TM_Sentiment'] = label_Encoder.fit_transform(X['TMData.TM_Sentiment'])
standardScale.fit_transform(X)

# connecting SMOTE to the data
over_sampler = SMOTE(k_neighbors=3)
X, y = over_sampler.fit_resample(X, y)

# Classifications
k_neighbors = KNeighborsClassifier()
tree = DecisionTreeClassifier()
SVR = SVC()
nn = MLPClassifier()
cnb = ComplementNB()

# Number of features n_features_to_select=n, where n is number 1-...n
selector_neighbors = SequentialFeatureSelector(k_neighbors, n_features_to_select=7).fit(X, y)
selector_tree = SequentialFeatureSelector(tree, n_features_to_select=7).fit(X, y)
selector_svr = SequentialFeatureSelector(SVR, n_features_to_select=7).fit(X, y)
selector_nn = SequentialFeatureSelector(nn, n_features_to_select=10).fit(X, y)
selector_cnb = SequentialFeatureSelector(cnb, n_features_to_select=7).fit(X, y)

selector_nn = load('Logs/old/selector_nn(26-16).joblib')

# Printing names of features
print('Метрики: SVR: {}\nKN: {}\nRidge: {}\nLR: {}\nTree: {}'.format(selector_svr.get_support(),
                                                                     selector_neighbors.get_support(),
                                                                     selector_cnb.get_support(),
                                                                     selector_nn.get_support(),
                                                                     selector_tree.get_support()))

# applying selected features to data
selector_neighbors_X = selector_neighbors.transform(X)
selector_svr_X = selector_svr.transform(X)
selector_cnb_X = selector_cnb.transform(X)
selector_nn_X = selector_nn.transform(X)
selector_tree_X = selector_tree.transform(X)
# remembering seed of process
random.seed(1)

# Grids with params
param_grid_k_neighbors = {
    "n_neighbors": range(5, 10, 1),
    "leaf_size": range(30, 50, 1),
    "p": [1, 2]}

param_grid_svc = {"kernel": ["rbf"],
                  "C": np.logspace(-9, 9, num=25, base=10),
                  "gamma": np.logspace(-9, 9, num=25, base=10)}

param_grid_nb = {
    "alpha": np.arange(0.0, 1.0, 0.01),
    "norm": [True, False],
}

param_grid_dt = {
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ['random', "best"],
    "max_depth": [None, 1, 2, 3, 4, 5],
    "min_samples_split": range(2, 5),
    "min_samples_leaf": range(1, 3)
}

param_grid_nn = {
    "activation": ["identity", "logistic", "tanh", "relu"],
    "hidden_layer_sizes": range(10, 150, 10),
    "solver": ["lbfgs", "sgd", "adam"],
    "alpha": np.arange(0.0, 0.001, 0.0001),
    "learning_rate": ["constant", "invscaling", "adaptive"]
}
random.seed(1)

# selecting params for classifiers
cv_kn = EvolutionaryAlgorithmSearchCV(estimator=KNeighborsClassifier(),
                                      params=param_grid_k_neighbors,
                                      scoring="accuracy",
                                      cv=StratifiedKFold(n_splits=4),
                                      verbose=1,
                                      population_size=50,
                                      gene_mutation_prob=0.10,
                                      gene_crossover_prob=0.5,
                                      tournament_size=3,
                                      generations_number=5,
                                      n_jobs=4)

cv_nb = EvolutionaryAlgorithmSearchCV(estimator=ComplementNB(),
                                      params=param_grid_nb,
                                      scoring="accuracy",
                                      cv=StratifiedKFold(n_splits=4),
                                      verbose=1,
                                      population_size=50,
                                      gene_mutation_prob=0.10,
                                      gene_crossover_prob=0.5,
                                      tournament_size=3,
                                      generations_number=5,
                                      n_jobs=4)

cv_svc = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                       params=param_grid_svc,
                                       scoring="accuracy",
                                       cv=StratifiedKFold(n_splits=4),
                                       verbose=1,
                                       population_size=50,
                                       gene_mutation_prob=0.10,
                                       gene_crossover_prob=0.5,
                                       tournament_size=3,
                                       generations_number=5,
                                       n_jobs=4)

cv_dt = EvolutionaryAlgorithmSearchCV(estimator=DecisionTreeClassifier(),
                                      params=param_grid_dt,
                                      scoring="accuracy",
                                      cv=StratifiedKFold(n_splits=4),
                                      verbose=1,
                                      population_size=50,
                                      gene_mutation_prob=0.10,
                                      gene_crossover_prob=0.5,
                                      tournament_size=3,
                                      generations_number=5,
                                      n_jobs=4)

cv_nn = EvolutionaryAlgorithmSearchCV(estimator=MLPClassifier(),
                                      params=param_grid_nn,
                                      scoring="accuracy",
                                      cv=StratifiedKFold(n_splits=4),
                                      verbose=1,
                                      population_size=50,
                                      gene_mutation_prob=0.10,
                                      gene_crossover_prob=0.5,
                                      tournament_size=3,
                                      generations_number=5,
                                      n_jobs=4)

# start process of getting params
if __name__ == "__main__":
    print(1)
    cv_kn.fit(selector_neighbors_X, y)
    cv_nb.fit(selector_cnb_X, y)
    cv_svc.fit(selector_svr_X, y)
    cv_dt.fit(selector_tree_X, y)
    cv_nn.fit(selector_nn_X, y)
