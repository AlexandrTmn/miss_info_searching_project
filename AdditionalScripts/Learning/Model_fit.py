import datetime
import json
import warnings
import os, glob
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
from sklearn.metrics import classification_report

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


# Loading data from DB
data = get_data()

# drop nan columns and rows
data = data.drop(columns=['UMData', 'TMMData', 'TMData'])
data1 = data.dropna()

# Choosing X and y
X = data1.drop('Hand_mark', axis=1)
y = data1['Hand_mark']

# Scaling and normalising
label_Encoder = LabelEncoder()
standardScale = StandardScaler()
X['TMData.TM_Sentiment'] = label_Encoder.fit_transform(X['TMData.TM_Sentiment'])
standardScale.fit_transform(X)
# Splitting on test and training datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=128)

# Applying SMOTE
over_sampler = SMOTE(k_neighbors=3)
X, y = over_sampler.fit_resample(X_train, y_train)
# X, y = X_train,y_train

# Defining models
print('Data ready')
k_neighbors = KNeighborsClassifier(n_neighbors=7, leaf_size=37, p=1)
tree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=3,
                              min_samples_leaf=2)
SVR = SVC(kernel='rbf', C=0.001, gamma=1.7782794100389227e-07)
nn = MLPClassifier(activation='relu', hidden_layer_sizes=130, solver='adam', alpha=0.0004, learning_rate='invscaling')
cnb = ComplementNB(alpha=0.17, norm=True)

# Feature selection
print('Selector start')
sel_nei = SequentialFeatureSelector(k_neighbors, n_features_to_select=2).fit(X, y)
sel_tree = SequentialFeatureSelector(tree, n_features_to_select=7).fit(X, y)
sel_svr = SequentialFeatureSelector(SVR, n_features_to_select=7).fit(X, y)
sel_nn = SequentialFeatureSelector(nn, n_features_to_select=7).fit(X, y)
sel_cnb = SequentialFeatureSelector(cnb, n_features_to_select=7).fit(X, y)

date = datetime.datetime.today()

# Saving features
dump(sel_nei, 'selector_nei({}-{}).joblib'.format(str(date.day), str(date.hour)))
dump(sel_tree, 'selector_tree({}-{}).joblib'.format(str(date.day), str(date.hour)))
dump(sel_svr, 'selector_svr({}-{}).joblib'.format(str(date.day), str(date.hour)))
dump(sel_nn, 'selector_nn({}-{}).joblib'.format(str(date.day), str(date.hour)))
dump(sel_cnb, 'selector_cnb({}-{}).joblib'.format(str(date.day), str(date.hour)))

# Applying features selections
print('Selector done\nTransform start')
features = 9
folder_path = 'F:\Аспирант\Проект\AdditionalScripts\Learning\Logs\\''' + str(features) + '_features'
sel1, sel2, sel3, sel4, sel5 = glob.glob(os.path.join(folder_path, '*.joblib'))
sel_cnb, sel_nei, sel_nn, sel_svr, sel_tree = load(sel1), load(sel2), load(sel3), load(sel4), load(sel5)
sel_cnb_x, sel_nei_x, sel_nn_x, sel_svr_x, sel_tree_x = sel_cnb.transform(X), sel_nei.transform(X), \
    sel_nn.transform(X), sel_svr.transform(X), sel_tree.transform(X)

print('Transform ended\nFit start')

# Fitting models
with parallel_backend('threading', n_jobs=4):
    # kn = k_neighbors.fit(sel_nei_x, y)
    # svr = SVR.fit(sel_svr_x, y)
    # cnb = cnb.fit(sel_cnb_x, y)
    # nn = nn.fit(sel_nn_x, y)
    # tree = tree.fit(sel_tree_x, y)

    kn = k_neighbors.fit(X, y)
    svr = SVR.fit(X, y)
    cnb = cnb.fit(X, y)
    nnf = nn.fit(X, y)
    tree = tree.fit(X, y)

# Saving models
print('Fit ended\nSaving')
dump(kn, filename='kn({}-{}).joblib'.format(str(date.day), str(date.hour)))
dump(svr, filename='svr({}-{}).joblib'.format(str(date.day), str(date.hour)))
dump(cnb, filename='cnb({}-{}).joblib'.format(str(date.day), str(date.hour)))
dump(nnf, filename='nn({}-{}).joblib'.format(str(date.day), str(date.hour)))
dump(tree, filename='tree({}-{}).joblib'.format(str(date.day), str(date.hour)))
