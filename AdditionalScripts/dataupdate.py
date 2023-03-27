# -*- coding: utf-8 -*-
"""
Data update and preprocessing

preprocess - preprocessing data for machine learning and metrics
time_update - update time from Twitter standard format
"""

# imports
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# preprocessing data for getting metrics
def preprocess(text):
    # Uncomment if not installed
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')

    # Lemmatize tweet text
    wnl = WordNetLemmatizer()
    stopword = set(stopwords.words('russian'))

    # remove new lines
    text = text.replace('\n', ' ')

    # remove links
    text = re.sub('https?://\S+|www\.\S+', ' ', text)

    # remove hashtags at the end of text
    text = re.sub('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', '', text)

    # remove handles
    text = re.sub('@[\w]+', '', text)

    # remove punctuations
    punc = '''.?!,:;_-[](){}'"`~|\/@#$%^&+=*'''
    for i in text:
        if i in punc:
            text = text.replace(i, ' ')

    # remove extra spaces
    re.sub("\s\s+", " ", text)

    # lower case
    text = text.strip().lower()

    # lemmatization
    text = [wnl.lemmatize(word) for word in text.split(' ')]
    text = " ".join(text)

    # stopword removal
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)

    # replace covid19 with covid
    text = text.replace('covid19', 'covid')
    return text


def time_update(data_row: str):
    data_fix = {'Day': data_row[0:10], 'Time': data_row[11:19]}
    return data_fix
