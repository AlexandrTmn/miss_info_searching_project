import json
import os, glob, warnings
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from AdditionalScripts.DBScripts.DBConnection import db_connection
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
import tensorflow as tf
import plotly as pl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE

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
data = data.drop(columns=['UMData', 'TMMData', 'TMData'])
data1 = data.dropna()

X = data1.drop('Hand_mark', axis=1)
y = data1['Hand_mark']

standardScale = StandardScaler()
label_Encoder = LabelEncoder()
X['TMData.TM_Sentiment'] = label_Encoder.fit_transform(X['TMData.TM_Sentiment'])
standardScale.fit_transform(X)
stats = {}

over_sampler = SMOTE(k_neighbors=3)
X, y = over_sampler.fit_resample(X, y)


def report(features: int, smote=True):
    if features == 6 or features == 9:
        folder_path = 'F:\Аспирант\Проект\AdditionalScripts\Learning\Logs\\''' + str(features) + '_features'
        sel1, sel2, sel3, sel4, sel5 = glob.glob(os.path.join(folder_path, '*.joblib'))
        selcnb, selnei, selnn, selsvr, seltree = load(sel1), load(sel2), load(sel3), load(sel4), load(sel5)
        selcnb_x, selnei_x, selnn_x, selsvr_x, seltree_x = selcnb.transform(X), selnei.transform(X), \
            selnn.transform(X), selsvr.transform(X), seltree.transform(X)

        if smote is True:
            model_path = 'F:\Аспирант\Проект\AdditionalScripts\Learning\Logs\\''' + str(features) + '_features\models'
        else:
            model_path = 'F:\Аспирант\Проект\AdditionalScripts\Learning\Logs\\''' + str(features) + \
                         '_features_nosmote\models'

        cnbf, knf, nnf, svrf, treef = glob.glob(os.path.join(model_path, '*.joblib'))
        cnbf, knf, nnf, svrf, treef = load(cnbf), load(knf), load(nnf), load(svrf), load(treef)

        X_train_cnb, X_test_cnb, y_train_cnb, y_test_cnb = train_test_split(selcnb_x, y, test_size=0.30,
                                                                            random_state=128)
        X_train_nei, X_test_nei, y_train_nei, y_test_nei = train_test_split(selnei_x, y, test_size=0.30,
                                                                            random_state=128)
        X_train_nn, X_test_nn, y_trai_nn, y_test_nn = train_test_split(selnn_x, y, test_size=0.30, random_state=128)
        X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(selsvr_x, y, test_size=0.30,
                                                                            random_state=128)
        X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(seltree_x, y, test_size=0.30,
                                                                                random_state=128)

        y_pred_cnbf, y_pred_knf, y_pred_nnf, y_pred_svrf, y_pred_treef = cnbf.predict(X_test_cnb), \
            knf.predict(X_test_nei), nnf.predict(X_test_nn), svrf.predict(X_test_svr), treef.predict(X_test_tree)

        cnbf_rep, knf_rep, nnf_rep, svrf_rep, treef_rep = \
            classification_report(y_test_nei, y_pred_knf, output_dict=True), \
                classification_report(y_test_svr, y_pred_svrf, output_dict=True), \
                classification_report(y_test_cnb, y_pred_cnbf, output_dict=True), \
                classification_report(y_test_nn, y_pred_nnf, output_dict=True), \
                classification_report(y_test_tree, y_pred_treef, output_dict=True),
        print('Features number:', features, '\n', 'SMOTE is', smote, '\n', )
        print('cnbf\n', cnbf_rep, '\n knf\n', knf_rep, '\n nnf\n', nnf_rep, '\n svrf\n', svrf_rep, '\n treef\n',
              treef_rep)

        cnbf_conf, knf_conf, nnf_conf, svrf_conf, treef_conf = \
            confusion_matrix(y_test_nei, y_pred_cnbf), \
                confusion_matrix(y_test_svr, y_pred_knf), \
                confusion_matrix(y_test_cnb, y_pred_nnf), \
                confusion_matrix(y_test_nn, y_pred_svrf), \
                confusion_matrix(y_test_tree, y_pred_treef),
        print('cnbf\n', cnbf_conf, '\n knf\n', knf_conf, '\n nnf\n', nnf_conf, '\n svrf\n', svrf_conf, '\n treef\n',
              treef_conf)

        features_am = str(features) + '_features_and_smote_is_' + str(smote)
        stats[features_am] = {}
        stats[features_am]['cnb'] = (round(cnbf_rep['False']['f1-score'], 8), round(cnbf_rep['False']['precision'], 8),
                                     round(cnbf_rep['False']['recall'], 8))
        stats[features_am]['knf'] = (round(knf_rep['False']['f1-score'], 8), round(knf_rep['False']['precision'], 8),
                                     round(knf_rep['False']['recall'], 8))
        stats[features_am]['nnf'] = (round(nnf_rep['False']['f1-score'], 8), round(nnf_rep['False']['precision'], 8),
                                     round(nnf_rep['False']['recall'], 8))
        stats[features_am]['svrf'] = (round(svrf_rep['False']['f1-score'], 8), round(svrf_rep['False']['precision'], 8),
                                      round(svrf_rep['False']['recall'], 8))
        stats[features_am]['treef'] = (
            round(treef_rep['False']['f1-score'], 8), round(treef_rep['False']['precision'], 8),
            (treef_rep['False']['recall'], 8))

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=120)
        if smote is True:
            models29_path = 'F:\Аспирант\Проект\AdditionalScripts\Learning\Logs\\''' + str(29) + \
                            '_features\models'
        else:
            models29_path = 'F:\Аспирант\Проект\AdditionalScripts\Learning\Logs\\''' + str(29) + \
                            '_features_nosmote\models'

        features_am = str(features) + '_features_and_smote_is_' + str(smote)
        stats[features_am] = {}
        cnbf, knf, nnf, svrf, treef = glob.glob(os.path.join(models29_path, '*.joblib'))
        cnbf, knf, nnf, svrf, treef = load(cnbf), load(knf), load(nnf), load(svrf), load(treef)

        y_pred_cnbf, y_pred_knf, y_pred_nnf, y_pred_svrf, y_pred_treef = cnbf.predict(X_test), \
            knf.predict(X_test), nnf.predict(X_test), svrf.predict(X_test), treef.predict(X_test)

        cnbf_rep, knf_rep, nnf_rep, svrf_rep, treef_rep = \
            classification_report(y_test, y_pred_cnbf, output_dict=True), \
                classification_report(y_test, y_pred_knf, output_dict=True), \
                classification_report(y_test, y_pred_nnf, output_dict=True), \
                classification_report(y_test, y_pred_svrf, output_dict=True), \
                classification_report(y_test, y_pred_treef, output_dict=True),
        print('Features number:', features, '\n', 'SMOTE is', smote, '\n', )
        print('cnbf\n', cnbf_rep, '\n knf\n', knf_rep, '\n nnf\n', nnf_rep, '\n svrf\n', svrf_rep, '\n treef\n',
              treef_rep)

        cnbf_conf, knf_conf, nnf_conf, svrf_conf, treef_conf = \
            confusion_matrix(y_test, y_pred_cnbf), \
                confusion_matrix(y_test, y_pred_knf), \
                confusion_matrix(y_test, y_pred_nnf), \
                confusion_matrix(y_test, y_pred_svrf), \
                confusion_matrix(y_test, y_pred_treef),
        print('cnbf\n', cnbf_conf, '\n knf\n', knf_conf, '\n nnf\n', nnf_conf, '\n svrf\n', svrf_conf, '\n treef\n',
              treef_conf)

        stats[features_am]['cnb'] = (round(cnbf_rep['False']['f1-score'], 8), round(cnbf_rep['False']['precision'], 8),
                                     round(cnbf_rep['False']['recall'], 8))
        stats[features_am]['knf'] = (round(knf_rep['False']['f1-score'], 8), round(knf_rep['False']['precision'], 8),
                                     round(knf_rep['False']['recall'], 8))
        stats[features_am]['nnf'] = (round(nnf_rep['False']['f1-score'], 8), round(nnf_rep['False']['precision'], 8),
                                     round(nnf_rep['False']['recall'], 8))
        stats[features_am]['svrf'] = (round(svrf_rep['False']['f1-score'], 8), round(svrf_rep['False']['precision'], 8),
                                      round(svrf_rep['False']['recall'], 8))
        stats[features_am]['treef'] = (round(treef_rep['False']['f1-score'], 8),
                                       round(treef_rep['False']['precision'], 8),
                                       round(treef_rep['False']['recall'], 8))

    return stats


def features_list(features: int):
    model_path = 'F:\Аспирант\Проект\AdditionalScripts\Learning\Logs\\''' + str(features) + '_features'
    for filename in glob.glob(os.path.join(model_path, '*.joblib')):
        print('\nВыбранные фичи для', filename.title())
        sel = load(filename)
        for i, j in enumerate(sel.support_):
            if j == True:
                print(data1.columns[i])


def plot(stats):
    columns = ['6_features_and_smote_is_True', '9_features_and_smote_is_True',
               '29_features_and_smote_is_True', '6_features_and_smote_is_False',
               '9_features_and_smote_is_False', '29_features_and_smote_is_False']
    df = pd.DataFrame.from_dict(stats, orient='columns').rename_axis('Model')
    df = df.reset_index(level=0)
    print(df)
    df.to_csv('test')
    datapx = px.data
    fig = px.bar(df, x="Model", y=df.columns.drop('Model'), barmode="group",
                 template="seaborn", title="Оценка Accuracy для моделей",
                 pattern_shape="Model", pattern_shape_sequence=[".", "x", "+"])
    fig.show()

    return None


features = [6, 9, 29]
smote = [True, False]
for j in smote:
    for i in features:
        # features_list(features=i)
        report(features=i, smote=j)

# plot(stats)

# print(stats)

# tf.keras.models.load_model("TF-weight.index")
