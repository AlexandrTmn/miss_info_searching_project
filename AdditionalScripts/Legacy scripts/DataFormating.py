from AdditionalScripts.DBScripts.DBConnection import db_connection
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

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

over_sampler = RandomOverSampler(random_state=42)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=128)
X_train, y_train = over_sampler.fit_resample(X_train, y_train)
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# Fixing over sampling


def allRegressors(X_train, X_test, y_train, y_test):
    """
    This function use multiple machine learning regressors and show us the results of them
    :param X_train: train input
    :param X_test: test input
    :param y_train: train output
    :param y_test: test output
    :return: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R Square for each regressors
    """

    # for Linear Regression
    print("*************************************************************************")
    lm_model = LinearRegression()
    lm_model.fit(X_train, y_train)
    lm_train_predictions = lm_model.predict(X_train)
    lm_test_predictions = lm_model.predict(X_test)
    print('Train MSE for Linear Regression:', mean_squared_error(y_train, lm_train_predictions))
    print('Test MSE for Linear Regression:', mean_squared_error(y_test, lm_test_predictions))
    print('Train RMSE for Linear Regression:', np.sqrt(mean_squared_error(y_train, lm_train_predictions)))
    print('Test RMSE for Linear Regression:', np.sqrt(mean_squared_error(y_test, lm_test_predictions)))
    print('Train MAE for Linear Regression:', mean_absolute_error(y_train, lm_train_predictions))
    print('Test MAE for Linear Regression:', mean_absolute_error(y_test, lm_test_predictions))
    print('Train R Square for Linear Regression:', r2_score(y_train, lm_train_predictions))
    print('Test R Square for Linear Regression:', r2_score(y_test, lm_test_predictions))

    # for Support Vector Machine Regressor
    print("*************************************************************************")
    svm_model = SVR()
    svm_model.fit(X_train, y_train)
    svm_train_predictions = svm_model.predict(X_train)
    svm_test_predictions = svm_model.predict(X_test)
    print('Train MSE for Support Vector Regression:', mean_squared_error(y_train, svm_train_predictions))
    print('Test MSE for Support Vector Regression:', mean_squared_error(y_test, svm_test_predictions))
    print('Train RMSE for Support Vector Regression:', np.sqrt(mean_squared_error(y_train, svm_train_predictions)))
    print('Test RMSE for Support Vector Regression:', np.sqrt(mean_squared_error(y_test, svm_test_predictions)))
    print('Train MAE for Support Vector Regression:', mean_absolute_error(y_train, svm_train_predictions))
    print('Test MAE for Support Vector Regression:', mean_absolute_error(y_test, svm_test_predictions))
    print('Train R Square for Support Vector Regression:', r2_score(y_train, svm_train_predictions))
    print('Test R Square for Support Vector Regression:', r2_score(y_test, svm_test_predictions))

    # for Random Forest Regression
    print("*************************************************************************")
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    rf_train_predictions = rf_model.predict(X_train)
    rf_test_predictions = rf_model.predict(X_test)
    print('Train MSE for Random Forest Regression:', mean_squared_error(y_train, rf_train_predictions))
    print('Test MSE for Random Forest Regression:', mean_squared_error(y_test, rf_test_predictions))
    print('Train RMSE for Random Forest Regression:', np.sqrt(mean_squared_error(y_train, rf_train_predictions)))
    print('Test RMSE for Random Forest Regression:', np.sqrt(mean_squared_error(y_test, rf_test_predictions)))
    print('Train MAE for Random Forest Regression:', mean_absolute_error(y_train, rf_train_predictions))
    print('Test MAE for Random Forest Regression:', mean_absolute_error(y_test, rf_test_predictions))
    print('Train R Square for Random Forest Regression:', r2_score(y_train, rf_train_predictions))
    print('Test R Square for Random Forest Regression:', r2_score(y_test, rf_test_predictions))

    # for Gradient Boosting Regression
    print("*************************************************************************")
    gb_model = GradientBoostingRegressor()
    gb_model.fit(X_train, y_train)
    gb_train_predictions = gb_model.predict(X_train)
    gb_test_predictions = gb_model.predict(X_test)
    print('Train MSE for Gradient Boosting Regression:', mean_squared_error(y_train, gb_train_predictions))
    print('Test MSE for Gradient Boosting Regression:', mean_squared_error(y_test, gb_test_predictions))
    print('Train RMSE for Gradient Boosting Regression:', np.sqrt(mean_squared_error(y_train, gb_train_predictions)))
    print('Test RMSE for Gradient Boosting Regression:', np.sqrt(mean_squared_error(y_test, gb_test_predictions)))
    print('Train MAE for Gradient Boosting Regression:', mean_absolute_error(y_train, gb_train_predictions))
    print('Test MAE for Gradient Boosting Regression:', mean_absolute_error(y_test, gb_test_predictions))
    print('Train R Square for Gradient Boosting Regression:', r2_score(y_train, gb_train_predictions))
    print('Test R Square for Gradient Boosting Regression:', r2_score(y_test, gb_test_predictions))

    # for KNeighbors Regression
    print("*************************************************************************")
    kn_model = KNeighborsRegressor()
    kn_model.fit(X_train, y_train)
    kn_train_predictions = kn_model.predict(X_train)
    kn_test_predictions = kn_model.predict(X_test)
    print('Train MSE for KNeighbors Regression:', mean_squared_error(y_train, kn_train_predictions))
    print('Test MSE for KNeighbors Regression:', mean_squared_error(y_test, kn_test_predictions))
    print('Train RMSE for KNeighbors Regression:', np.sqrt(mean_squared_error(y_train, kn_train_predictions)))
    print('Test RMSE for KNeighbors Regression:', np.sqrt(mean_squared_error(y_test, kn_test_predictions)))
    print('Train MAE for KNeighbors Regression:', mean_absolute_error(y_train, kn_train_predictions))
    print('Test MAE for KNeighbors Regression:', mean_absolute_error(y_test, kn_test_predictions))
    print('Train R Square for KNeighbors Regression:', r2_score(y_train, kn_train_predictions))
    print('Test R Square for KNeighbors Regression:', r2_score(y_test, kn_test_predictions))

    # for Decision Tree Regresion
    print("*************************************************************************")
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, y_train)
    dt_train_predictions = dt_model.predict(X_train)
    dt_test_predictions = dt_model.predict(X_test)
    print('Train MSE for Decision Tree Regresion:', mean_squared_error(y_train, dt_train_predictions))
    print('Test MSE for Decision Tree Regresion:', mean_squared_error(y_test, dt_test_predictions))
    print('Train RMSE for Decision Tree Regresion:', np.sqrt(mean_squared_error(y_train, dt_train_predictions)))
    print('Test RMSE for Decision Tree Regresion:', np.sqrt(mean_squared_error(y_test, dt_test_predictions)))
    print('Train MAE for Decision Tree Regresion:', mean_absolute_error(y_train, dt_train_predictions))
    print('Test MAE for Decision Tree Regresion:', mean_absolute_error(y_test, dt_test_predictions))
    print('Train R Square for Decision Tree Regresion:', r2_score(y_train, dt_train_predictions))
    print('Test R Square for Decision Tree Regresion:', r2_score(y_test, dt_test_predictions))

    # for Ridge Regression
    print("*************************************************************************")
    rid_model = Ridge(alpha=.5)
    rid_model.fit(X_train, y_train)
    rid_train_predictions = rid_model.predict(X_train)
    rid_test_predictions = rid_model.predict(X_test)
    print('Train MSE for Ridge Regression:', mean_squared_error(y_train, rid_train_predictions))
    print('Test MSE for Ridge Regression:', mean_squared_error(y_test, rid_test_predictions))
    print('Train RMSE for Ridge Regression:', np.sqrt(mean_squared_error(y_train, rid_train_predictions)))
    print('Test RMSE for Ridge Regression:', np.sqrt(mean_squared_error(y_test, rid_test_predictions)))
    print('Train MAE for Ridge Regression:', mean_absolute_error(y_train, rid_train_predictions))
    print('Test MAE for Ridge Regression:', mean_absolute_error(y_test, rid_test_predictions))
    print('Train R Square for Ridge Regression:', r2_score(y_train, rid_train_predictions))
    print('Test R Square for Ridge Regression:', r2_score(y_test, rid_test_predictions))

    # Summary
    print("---------------------------Summary---------------------------")
    print("*************************************************************************")
    print('Test MAE for Linear Regression:', mean_absolute_error(y_test, lm_test_predictions))
    print('Test RMSE for Linear Regression:', np.sqrt(mean_squared_error(y_test, lm_test_predictions)))
    print('Test R Square for Linear Regression:', r2_score(y_test, lm_test_predictions))
    print("*************************************************************************")
    print('Test MAE for Support Vector Regression:', mean_absolute_error(y_test, svm_test_predictions))
    print('Test RMSE for Support Vector Regression:', np.sqrt(mean_squared_error(y_test, svm_test_predictions)))
    print('Test R Square for Support Vector Regression:', r2_score(y_test, svm_test_predictions))
    print("*************************************************************************")
    print('Test MAE for Random Forest Regression:', mean_absolute_error(y_test, rf_test_predictions))
    print('Test RMSE for Random Forest Regression:', np.sqrt(mean_squared_error(y_test, rf_test_predictions)))
    print('Test R Square for Random Forest Regression:', r2_score(y_test, rf_test_predictions))
    print("*************************************************************************")
    print('Test MAE for Gradient Boosting Regression:', mean_absolute_error(y_test, gb_test_predictions))
    print('Test RMSE for Gradient Boosting Regression:', np.sqrt(mean_squared_error(y_test, gb_test_predictions)))
    print('Test R Square for Gradient Boosting Regression:', r2_score(y_test, gb_test_predictions))
    print("*************************************************************************")
    print('Test MAE for KNeighbors Regression:', mean_absolute_error(y_test, kn_test_predictions))
    print('Test RMSE for KNeighbors Regression:', np.sqrt(mean_squared_error(y_test, kn_test_predictions)))
    print('Test R Square for KNeighbors Regression:', r2_score(y_test, kn_test_predictions))
    print("*************************************************************************")
    print('Test MAE for Decision Tree Regresion:', mean_absolute_error(y_test, dt_test_predictions))
    print('Test RMSE for Decision Tree Regresion:', np.sqrt(mean_squared_error(y_test, dt_test_predictions)))
    print('Test R Square for Decision Tree Regresion:', r2_score(y_test, dt_test_predictions))
    print("*************************************************************************")
    print('Test MAE for Ridge Regression:', mean_absolute_error(y_test, rid_test_predictions))
    print('Test RMSE for Ridge Regression:', np.sqrt(mean_squared_error(y_test, rid_test_predictions)))
    print('Test R Square for Ridge Regression:', r2_score(y_test, rid_test_predictions))
    print("*************************************************************************")


# allRegressors(X_train, X_test, y_train, y_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from scikitplot.metrics import plot_roc
from scikitplot.metrics import plot_precision_recall
from scikitplot.metrics import plot_cumulative_gain
from scikitplot.metrics import plot_lift_curve
from numpy import argmax
import numpy as np


def build_and_test(X_tr, X_te, y_tr, y_te, class_weight=None, threshold=False):
    # Build and Plot PCA
    pca = PCA(n_components=2)
    pca.fit(X_tr)
    X_pca = pca.transform(X_tr)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_tr, cmap=plt.cm.prism, edgecolor='k', alpha=0.7)
    plt.show()

    # Build and fit the model
    if class_weight:
        model = DecisionTreeClassifier(class_weight=class_weight)
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

    # Calculate the best threshold
    best_threshold = None
    if threshold:
        J = tpr0 - fpr0
        ix = argmax(J)  # take the value which maximizes the J variable
        best_threshold = thresholds[ix]
        # adjust score according to threshold.
        y_score = np.array([[1, y[1]] if y[0] >= best_threshold else [0, y[1]] for y in y_score])

    # Plot metrics
    plot_roc(y_te, y_score)
    plt.show()

    plot_precision_recall(y_te, y_score)
    plt.show()

    plot_cumulative_gain(y_te, y_score)
    plt.show()

    plot_lift_curve(y_te, y_score)
    plt.show()

    # Print a classification report
    print(classification_report(y_te, y_pred))
    return roc_auc0, fpr0, tpr0, best_threshold


roc_auc_ros, fpr_ros, tpr_ros, _ = build_and_test(X_train, X_test, y_train, y_test)
