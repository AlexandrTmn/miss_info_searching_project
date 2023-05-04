# -*- coding: utf-8 -*-
"""
Models

get_data -
report -
"""

# imports
import glob
import numpy as np
import json
import os
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler, LabelEncoder
from AdditionalScripts.DBScripts.DBConnection import db_connection
import tensorflow as tf
# settings
pd.set_option("display.max.columns", None)
pd.set_option('display.max_colwidth', None)


# Import data from database
def get_data():
    """
    Importing data from edgedb
    """
    conn = db_connection()
    for tx in conn.transaction():
        with tx:
            data_json = tx.query_json("""
            select tweet {
            TweetId, Hand_mark,
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


def report(tweet_id, smote):
    # Preparing data
    data = get_data()
    if tweet_id in data['TweetId']:
        print('Yes')
    data = data.loc[data['TweetId'] == np.int(tweet_id)]
    data = data.drop(columns=['UMData', 'TMMData', 'TMData'])
    # Drop intermediate data
    x = data.drop('Hand_mark', axis=1)
    x = x.drop('TweetId', axis=1)
    # Scaling data and Encoding
    standard_scale = StandardScaler()
    label_encoder = LabelEncoder()
    x['TMData.TM_Sentiment'] = label_encoder.fit_transform(x['TMData.TM_Sentiment'])
    standard_scale.fit_transform(x)

    features = [6, 9, 29]
    result = pd.DataFrame(columns=['Number_of_features', 'Model', 'SMOTE', 'Result', 'Numbers'])
    index = 0
    # Loading models from save
    for ftr in features:
        # Loading data for 6 and 9 features
        if ftr == 6 or ftr == 9:
            # Loading selected features from save
            folder_path = 'F:\Аспирант\Проект\AdditionalScripts\Learning\Logs\\''' + str(ftr) + '_features'
            sel1, sel2, sel3, sel4, sel5 = glob.glob(os.path.join(folder_path, '*.joblib'))
            sel_cnb, sel_nei, sel_nn, sel_svr, sel_tree = load(sel1), load(sel2), load(sel3), load(sel4), load(sel5)
            sel_cnb_x, sel_nei_x, sel_nn_x, sel_svr_x, sel_tree_x = sel_cnb.transform(x), sel_nei.transform(x), \
                sel_nn.transform(x), sel_svr.transform(x), sel_tree.transform(x)

            # loading models from save for 6 and 9 features
            if smote is True:
                model_path = 'F:\Аспирант\Проект\AdditionalScripts\Learning\Logs\\''' + str(ftr) + '_features\models'
            else:
                model_path = 'F:\Аспирант\Проект\AdditionalScripts\Learning\Logs\\''' + str(ftr) + \
                             '_features_nosmote\models'

            cnb, kn, nn, svr, tree = glob.glob(os.path.join(model_path, '*.joblib'))
            cnb, kn, nn, svr, tree = load(cnb), load(kn), load(nn), load(svr), load(tree)

            # Predicting for models
            data_to_append = {
                'CNB': cnb.predict(sel_cnb_x),
                'KN': kn.predict(sel_nei_x),
                'MLP': nn.predict(sel_nn_x),
                'SVC': svr.predict(sel_svr_x),
                'DT': tree.predict(sel_tree_x)
            }

            numbers_to_append = {
                'CNB': cnb.predict_proba(sel_cnb_x),
                'KN': kn.predict_proba(sel_nei_x),
                'MLP': nn.predict_proba(sel_nn_x),
                'SVC': None,
                'DT': tree.predict_proba(sel_tree_x)
            }

            for items in data_to_append.items():
                result.loc[index] = [ftr, items[0], smote, items[1][0], numbers_to_append[items[0]]]
                index += 1
        else:
            # loading models from save for 29 features
            if smote is True:
                models29_path = 'F:\Аспирант\Проект\AdditionalScripts\Learning\Logs\\''' + str(29) + \
                                '_features\models'
            else:
                models29_path = 'F:\Аспирант\Проект\AdditionalScripts\Learning\Logs\\''' + str(29) + \
                                '_features_nosmote\models'

            cnb, kn, nn, svr, tree = glob.glob(os.path.join(models29_path, '*.joblib'))
            cnb, kn, nn, svr, tree = load(cnb), load(kn), load(nn), load(svr), load(tree)
            # new_model = tf.keras.models.load_model('AdditionalScripts/Learning/Logs/Keras models/Keras_model(24-11).h5')
            # Predicting for models
            data_to_append = {
                'CNB': cnb.predict(x),
                'KN': kn.predict(x),
                'MLP': nn.predict(x),
                'SVC': svr.predict(x),
                'DT': tree.predict(x),
                # 'TF': new_model.predict(x)
            }
            numbers_to_append = {
                'CNB': cnb.predict_proba(x),
                'KN': kn.predict_proba(x),
                'MLP': nn.predict_proba(x),
                'SVC': None,
                'DT': tree.predict_proba(x),
                # 'TF': new_model.predict_proba(x)
            }

            for items in data_to_append.items():
                result.loc[index] = [ftr, items[0], smote, items[1][0], numbers_to_append[items[0]]]
                index += 1

    return result
