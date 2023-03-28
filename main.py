# -*- coding: utf-8 -*-
"""
main

ask_id - asking user for url or id
check_valid_url_or_id - checking if id or url is valid and not exist in DB already
tweepy_get_data - getting basic data from Twitter API
update_data - updating data for metrics
experimental_metrics - getting experimental metrics
legacy_metrics - getting legacy metrics
merging_data - merging multiple dicts
data_fix - fixing formats for models
"""

# imports
import json
import tweepy
import pandas as pd
import AdditionalScripts.Auth as Twp
import AdditionalScripts.dataupdate as dup
from AdditionalScripts.Metrics import Experimental, Metrics
from AdditionalScripts.DBScripts import DBInsert as Dbi
from AdditionalScripts.DBScripts.DBConnection import db_connection
from AdditionalScripts.Learning import Models
import datetime as dt
import edgedb.errors


def ask_id():
    """
    Getting Tweet ID or URL from user

    If getting something other than ID or URl return 0.
    Try again if there was an exception.
    """
    tweet = input('Please input tweet id of tweet url: \n')
    if tweet.isdigit():
        print('You entered id =', tweet)
        answer = input('Please enter Yes or No: \n')
        if answer == 'Yes':
            return tweet
        elif answer == 'No':
            return 0
        else:
            return 0

    elif 'http' in tweet:
        print('You entered url', tweet)
        answer = input('Please enter Yes or No: \n')
        if answer == 'Yes':
            return tweet
        elif answer == 'No':
            return 0
        else:
            return 0
    else:
        print('There is no id or url\nTry again')
        return 0


def check_valid_url_or_id(tweet_id_or_url):
    """
    Validating given url or id

    If ID or URL is valid then returning id for future data extraction
    """
    # connection to database for exist checking
    conn = db_connection()
    for tx in conn.transaction():
        with tx:
            data_json = tx.query_json("""
                select tweet {
                TweetId}
                """)

            y = json.loads(data_json)
            data_pd = pd.json_normalize(y, max_level=1)

    # Getting id from url
    if 'http' in tweet_id_or_url:
        tweet_id = str(tweet_id_or_url).rsplit('/', 1)[1]
    else:
        tweet_id = tweet_id_or_url

    # Checking if id already exist
    if tweet_id in data_pd['TweetId']:
        return False

    # Checking if id is valid
    try:
        Twp.api.lookup_statuses(tweet_id)
    except (tweepy.errors.NotFound, tweepy.errors.Forbidden):
        return False
    else:
        return tweet_id


def tweepy_get_data(tweet_id: int):
    """
    Getting basic data for tweet-object and user-object
    """
    b_data = Twp.apiP.get_status(tweet_id)
    b_data = b_data._json
    og_data = {
        'tweet_id': tweet_id,
        'space': None,
        'verified': b_data['user']['verified'],
        'followers': b_data['user']['followers_count'],
        'friends_count': b_data['user']['friends_count'],
        'likes_count': b_data['user']['favourites_count'],
        'statuses': b_data['user']['statuses_count'],
        'user_created_at': b_data['user']['created_at'],
        'text': b_data['text'],
        'created_at': b_data['created_at'],
        'in_reply_to_status_id': b_data['in_reply_to_status_id'],
        'in_reply_to_user_id': b_data['in_reply_to_user_id'],
        'Likers_Count': b_data['favorite_count'],
        'Retweeters_Count': b_data['retweet_count'],
        'Retweeters': []
    }
    return og_data


def update_data(bd: dict):
    """
    updating_data

    Updating data for metrics
    """
    text = dup.preprocess(bd['text'])
    time = dup.time_update(bd['created_at'])
    ud = {
        'text': text,
        'date': time['Day'],
        'time': time['Time']
    }
    return ud


def experimental_metrics(bdata: dict, udata: dict):
    """
    experimental_metrics

    Getting experimental metrics
    """
    ex_met = {
        'user_registration_year': Experimental.user_registration_year(bdata),
        'repeat_symbols_and_words': Experimental.repeat_symbols_and_words(udata['text']),
        'hashtags': Experimental.hashtags(bdata['text']),
        'links': Experimental.links(bdata['text']),
        'spell_check': Experimental.spell_check(bdata['text']),
        'exclamation_mark': Experimental.exclamation_mark(bdata['text']),
        'conditional_words': Experimental.conditional_words(bdata['text']),
        'sensory_verbs': Experimental.sensory_verbs(bdata['text']),
        'ordinal_adjectives': Experimental.ordinal_adjectives(bdata['text']),
        'relative_time': Experimental.relative_time(bdata['text']),
        'numbers_count': Experimental.numbers_count(bdata['text']),
        'quantity_adjective': Experimental.quantity_adjective(bdata['text']),
        'certainty_words': Experimental.certainty_words(bdata['text']),
        'pronoun': Experimental.pronoun(bdata['text']),
    }
    return ex_met


def legacy_metrics(bdata: dict, udata: dict):
    """
    legacy_metrics

    Getting legacy metrics
    """
    lmc_text = Metrics.Metrics.text_metrics(text=bdata['text'], up_text=udata['text'])
    lmc_user = Metrics.Metrics.user_metrics(user=bdata, tweet_id=valid_id)
    lmc_timeline = Metrics.Metrics.timeline(user=bdata, tweet_id=valid_id)

    if lmc_text == 0:
        lmc_text = 0, False, False, False, [{'Skip': 0}]
    if not lmc_user:
        lmc_user = [0, 0, 0, 0, False]

    if lmc_timeline is False or lmc_timeline == 0:
        lmc_timeline = [0, 0, 0, 0]
    else:
        lmc_timeline = [0, 0, 0, 0]

    leg_met = {
        'word_count': lmc_text[0],
        'vulgar': lmc_text[1],
        'mean': lmc_text[2],
        'emotes': lmc_text[3],
        'sentiment': lmc_text[4],
        'influence': lmc_user[0],
        'role': lmc_user[1],
        'verified': lmc_user[2],
        'originality': lmc_user[3],
        'engagement': lmc_user[4],
        'low_to_high_diffusion': lmc_timeline[0],
        'longest_component_diffusion': lmc_timeline[1],
        'nodes_with_urls_diffusion': lmc_timeline[2],
        'isolated_diffusion': lmc_timeline[3],
    }
    return leg_met


def merging_data(f_data: list):
    """
    Combining dicts into one
    """
    merge_data = {}
    for item in f_data:
        merge_data.update(item)
    return merge_data


def data_fix(m_data: dict):
    """
    Fixing incompatibility formats of data and DB
    """
    for k, v in m_data.items():
        if v is None:
            m_data[k] = 0

    m_data.update(user_created_at=dt.datetime.strptime(m_data['user_created_at'], '%a %b %d %H:%M:%S %z %Y'))
    m_data.update(created_at=dt.datetime.strptime(m_data['created_at'], '%a %b %d %H:%M:%S %z %Y'))
    return m_data


if __name__ == "__main__":
    tweet_start = ask_id()
    valid_id = check_valid_url_or_id(tweet_start)
    if valid_id is not False:
        print('Getting basic data...')
        basic_data = tweepy_get_data(tweet_id=valid_id)
        updated_data = update_data(bd=basic_data)
        print('Getting params calculated...')
        experimental_metrics = experimental_metrics(bdata=basic_data, udata=updated_data)
        legacy_metrics = legacy_metrics(bdata=basic_data, udata=updated_data)
        print('Combining data...')
        full_data = [basic_data, updated_data, experimental_metrics, legacy_metrics]
        merged_data = merging_data(f_data=full_data)
        fixed_data = data_fix(m_data=merged_data)
        print('Done,\nUploading to DB')
        print(fixed_data)
        try:
            Dbi.data_insert(merged_data)
        except edgedb.errors.QueryError:
            print('error uploading with error')

        smote = input('Smote = True or False\n')
        if smote:
            result = Models.report(valid_id, smote)
        elif not smote:
            result = Models.report(valid_id, smote)
        else:
            pass

        print(result)
    else:
        print('Something is wrong with url or id')
