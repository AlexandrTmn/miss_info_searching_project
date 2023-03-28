import json
import warnings
import pandas as pd
from joblib import load
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from AdditionalScripts.DBScripts.DBConnection import db_connection
from tensorflow import keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import datetime
from keras.utils import FeatureSpace

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

names = data1.columns
for i in names:
    if data1[i].dtype == 'bool':
        data1[i] = data1[i].astype('object')

x = np.asarray(X).astype('float32')
train_df = tf.convert_to_tensor(x)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(train_df)

# feature_space = FeatureSpace(
#     features={
#         "EM_Hashtags": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "EM_Links": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "EM_Repeated_Symbols": FeatureSpace.float_normalized(),
#         "EM_Repeated_Words": FeatureSpace.float_normalized(),
#         "EM_Spelling": FeatureSpace.float_normalized(),
#         "EM_Exclamation_Mark": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "EM_Pronoun": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "EM_Conditional_Words": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "EM_Sensory_Verbs": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "EM_Ordinal_Adjectives": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "EM_Relative_Time": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "EM_Numbers_Count": FeatureSpace.float_normalized(),
#         "EM_Quantity_Adjective": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "EM_Certainty_Words": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "EM_Days_Between": FeatureSpace.float_normalized(),
#         "UM_Originality": FeatureSpace.float_normalized(),
#         "UM_Influence": FeatureSpace.integer_categorical(),
#         "UM_Role": FeatureSpace.float_normalized(),
#         "UM_Engagement": FeatureSpace.float_normalized(),
#         "UM_Trust": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "TM_Low_To_High_Diffustion": FeatureSpace.float_normalized(),
#         "TM_Biggest_Connected_Component": FeatureSpace.float_normalized(),
#         "TM_Tweets_With_URLs_Part": FeatureSpace.float_normalized(),
#         "TM_Isolated_Tweets_Part": FeatureSpace.float_normalized(),
#         "TM_Emotes": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "TM_Mean": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "TM_Vulgar": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "TM_Word_Count": FeatureSpace.integer_categorical(num_oov_indices=0),
#         "TM_Sentiment": FeatureSpace.string_categorical(num_oov_indices=0),
#     },
#     output_mode="concat",
# )

metrics = [
    tf.keras.metrics.BinaryAccuracy(),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.Precision(name='Precision'),
    keras.metrics.Recall(name="Recall"),
]


# model 1
def get_basic_model():
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(128, activation='selu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adamax(1e-2), loss="mse", metrics=metrics
    )
    return model


date = datetime.datetime.today()
model = get_basic_model()
model.summary()
history = model.fit(train_df, y, epochs=500, batch_size=256, verbose=2, )
# print(model.get_metrics_result())
model.save('Logs/Keras models/Keras_model({}-{}).h5'.format(str(date.day), str(date.hour)))
