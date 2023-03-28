from AdditionalScripts.DBScripts.DBConnection import db_connection
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from numpy import argmax
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

conn = db_connection()
pd.set_option("display.max.columns", None)
pd.set_option('display.max_colwidth', None)


class DataFormat:

    @classmethod
    def get_data(cls):
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


data = DataFormat.get_data()
# drop nan columns and rows
data = data.drop(columns=['UMData', 'TMMData', 'TMData'])
data1 = data.dropna()

# split and scaling
X = data1.drop('Hand_mark', axis=1)
y = data1['Hand_mark']
standardScale = StandardScaler()
label_Encoder = LabelEncoder()

# String to numerical type
X['TMData.TM_Sentiment'] = label_Encoder.fit_transform(X['TMData.TM_Sentiment'])
standardScale.fit_transform(X)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Fixing over sampling
def build_and_test(X_tr, X_te, y_tr, y_te, class_weight=None):
    # Build and Plot PCA
    pca = PCA(n_components=14)
    pca.fit(X_tr)
    x_pca = pca.transform(X_tr)

    # Build and fit the model
    if class_weight:
        model = SVC(gamma=0.001, C=1.7782794100389227e-07, kernel='rbf', class_weight=class_weight)
    else:
        model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=3,
                                       min_samples_leaf=2)
    model.fit(X_tr, y_tr)

    # Test the model
    y_pred = model.predict(X_te)
    print('Precision score %s' % precision_score(y_te, y_pred))
    print('Recall score %s' % recall_score(y_te, y_pred))
    print('F1-score score %s' % f1_score(y_te, y_pred))
    print('Accuracy score %s' % accuracy_score(y_te, y_pred))

    # Print a classification report
    print(classification_report(y_te, y_pred))
    return 0


# choice of model for imbalance data
print('imb')
imb = build_and_test(X_train, X_test, y_train, y_test)

print('ROS')
random_over_sample = RandomOverSampler(random_state=3)
X_res, y_res = random_over_sample.fit_resample(X_train, y_train)
ROS = build_and_test(X_res, X_test, y_res, y_test)

print('NM')
under_sampler = NearMiss()
X_res, y_res = under_sampler.fit_resample(X_train, y_train)
NM = build_and_test(X_res, X_test, y_res, y_test)

print('RUS')
random_under_sampler = RandomUnderSampler(random_state=3)
X_res, y_res = random_under_sampler.fit_resample(X_train, y_train)
RUS = build_and_test(X_res, X_test, y_res, y_test)

print('SMOTE')
over_sampler = SMOTE(k_neighbors=3)
X, y = over_sampler.fit_resample(X_train, y_train)
smote = build_and_test(X, X_test, y, y_test)
