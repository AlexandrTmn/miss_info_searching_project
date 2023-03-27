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

# over_sampler = RandomOverSampler(random_state=42)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train, y_train = over_sampler.fit_resample(X_train, y_train)
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Fixing over sampling
def build_and_test(X_tr, X_te, y_tr, y_te, class_weight=None, threshold=False):
    # Build and Plot PCA
    pca = PCA(n_components=14)
    pca.fit(X_tr)
    x_pca = pca.transform(X_tr)

    # Build and fit the model
    if class_weight:
        model = SVC(gamma=0.001, C=1.7782794100389227e-07, kernel='rbf', class_weight=class_weight)
    else:
        model = DecisionTreeClassifier()
    model.fit(X_tr, y_tr)

    # Test the model
    y_pred = model.predict(X_te)
    print('Precision score %s' % precision_score(y_te, y_pred))
    print('Recall score %s' % recall_score(y_te, y_pred))
    print('F1-score score %s' % f1_score(y_te, y_pred))
    print('Accuracy score %s' % accuracy_score(y_te, y_pred))

    y_score = model.predict_proba(X_te)
    fpr0, tpr0, thresholds = roc_curve(y_te, y_score[:, 1])
    roc_auc0 = auc(fpr0, tpr0)

    best_threshold = None
    if threshold:
        J = tpr0 - fpr0
        ix = argmax(J)
        best_threshold = thresholds[ix]

    # Print a classification report
    print(classification_report(y_te, y_pred))
    return roc_auc0, fpr0, tpr0, best_threshold


# choice of model for imbalance data
print('imb')
roc_auc_imb, fpr_imb, tpr_imb, _ = build_and_test(X_train, X_test, y_train, y_test)
random_over_sample = RandomOverSampler()
print('ROS')
X_res, y_res = random_over_sample.fit_resample(X_train, y_train)
roc_auc_ros, fpr_ros, tpr_ros, _ = build_and_test(X_res, X_test, y_res, y_test)
print('NM')
under_sampler = NearMiss()
X_res, y_res = under_sampler.fit_resample(X_train, y_train)
roc_auc_nm, fpr_nm, tpr_nm, _ = build_and_test(X_res, X_test, y_res, y_test)
print('RUS')
random_under_sampler = RandomUnderSampler(random_state=42)
X_res, y_res = random_under_sampler.fit_resample(X_train, y_train)
roc_auc_rus, fpr_rus, tpr_rus, _ = build_and_test(X_res, X_test, y_res, y_test)
print('SMOTE')
over_sampler = SMOTE(k_neighbors=3)
X_res, y_res = over_sampler.fit_resample(X_train, y_train)
roc_auc_smote, fpr_smote, tpr_smote, _ = build_and_test(X_res, X_test, y_res, y_test)
print(_)
print('Thr')
roc_auc_thr, fpr_thr, tpr_thr, threshold = build_and_test(X_train, X_test, y_train, y_test, threshold=True)

plt.plot(fpr_imb, tpr_imb, lw=3, label='Imbalanced $AUC_0$ = %.3f' % (roc_auc_imb))
plt.plot(fpr_ros, tpr_ros, lw=3, label='ROS $AUC_0$ = %.3f' % (roc_auc_ros))
plt.plot(fpr_smote, tpr_smote, lw=3, label='SMOTE $AUC_0$ = %.3f' % (roc_auc_smote))
plt.plot(fpr_rus, tpr_rus, lw=3, label='RUS $AUC_0$ = %.3f' % (roc_auc_rus))
plt.plot(fpr_nm, tpr_nm, lw=3, label='NM $AUC_0$ = %.3f' % (roc_auc_nm))
plt.plot(fpr_thr, tpr_thr, lw=3, label='NM $AUC_0$ = %.3f' % (roc_auc_thr))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc="lower right", fontsize=14, frameon=False)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()
