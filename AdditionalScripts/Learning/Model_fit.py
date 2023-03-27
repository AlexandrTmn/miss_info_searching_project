import datetime
import json
import warnings

from joblib import load
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import dump
from joblib import parallel_backend
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from AdditionalScripts.DBScripts.DBConnection import db_connection

# from sklearn.metrics import classification_report


warnings.filterwarnings("ignore")

conn = db_connection()
pd.set_option("display.max.columns", None)
pd.set_option('display.max_colwidth', None)


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
# drop nan columns and rows
data = data.drop(columns=['UMData', 'TMMData', 'TMData'])
data1 = data.dropna()
# data1 = data1[0:10000]

X = data1.drop('Hand_mark', axis=1)
y = data1['Hand_mark']
standardScale = StandardScaler()
label_Encoder = LabelEncoder()
print(len(X), len(y))
X['TMData.TM_Sentiment'] = label_Encoder.fit_transform(X['TMData.TM_Sentiment'])
standardScale.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=128)
over_sampler = SMOTE(k_neighbors=3)
X, y = over_sampler.fit_resample(X_train, y_train)
print(len(X), len(y))
# X, y = X_train,y_train
print('Data ready')
k_neighbors = KNeighborsClassifier(n_neighbors=7, leaf_size=37, p=1)
tree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=3,
                              min_samples_leaf=2)
SVR = SVC(kernel='rbf', C=0.001, gamma=1.7782794100389227e-07)
nn = MLPClassifier(activation='relu', hidden_layer_sizes=130, solver='adam', alpha=0.0004, learning_rate='invscaling')
cnb = ComplementNB(alpha=0.17, norm=True)
print('Selector start')
selnei = SequentialFeatureSelector(k_neighbors, n_features_to_select=2).fit(X, y)
seltree = SequentialFeatureSelector(tree, n_features_to_select=7).fit(X, y)
selsvr = SequentialFeatureSelector(SVR, n_features_to_select=7).fit(X, y)
selnn = SequentialFeatureSelector(nn, n_features_to_select=7).fit(X, y)
selcnb = SequentialFeatureSelector(cnb, n_features_to_select=7).fit(X, y)

date = datetime.datetime.today()

s1 = dump(selnei, 'selector_nei({}-{}).joblib'.format(str(date.day), str(date.hour)))
s2 = dump(seltree, 'selector_tree({}-{}).joblib'.format(str(date.day), str(date.hour)))
s3 = dump(selsvr, 'selector_svr({}-{}).joblib'.format(str(date.day), str(date.hour)))
s4 = dump(selnn, 'selector_nn({}-{}).joblib'.format(str(date.day), str(date.hour)))
s5 = dump(selcnb, 'selector_cnb({}-{}).joblib'.format(str(date.day), str(date.hour)))

print('Selector done\nTransform start')

import os, glob

features = 9
folder_path = 'F:\Аспирант\Проект\AdditionalScripts\Learning\Logs\\''' + str(features) + '_features'
sel1, sel2, sel3, sel4, sel5 = glob.glob(os.path.join(folder_path, '*.joblib'))
selcnb, selnei, selnn, selsvr, seltree = load(sel1), load(sel2), load(sel3), load(sel4), load(sel5)
selcnb_x, selnei_x, selnn_x, selsvr_x, seltree_x = selcnb.transform(X), selnei.transform(X), \
    selnn.transform(X), selsvr.transform(X), seltree.transform(X)

print('Transform ended\nFit start')
with parallel_backend('threading', n_jobs=4):
    # knf = k_neighbors.fit(selnei_x, y)
    # svrf = SVR.fit(selsvr_x, y)
    # cnbf = cnb.fit(selcnb_x, y)
    # nnf = nn.fit(selnn_x, y)
    # treef = tree.fit(seltree_x, y)

    knf = k_neighbors.fit(X, y)
    svrf = SVR.fit(X, y)
    cnbf = cnb.fit(X, y)
    nnf = nn.fit(X, y)
    treef = tree.fit(X, y)
print('Fit ended\nSaving')
s1 = dump(knf, filename='knf({}-{}).joblib'.format(str(date.day), str(date.hour)))
s2 = dump(svrf, filename='svrf({}-{}).joblib'.format(str(date.day), str(date.hour)))
s3 = dump(cnbf, filename='cnbf({}-{}).joblib'.format(str(date.day), str(date.hour)))
s4 = dump(nnf, filename='nnf({}-{}).joblib'.format(str(date.day), str(date.hour)))
s5 = dump(treef, filename='treef({}-{}).joblib'.format(str(date.day), str(date.hour)))
