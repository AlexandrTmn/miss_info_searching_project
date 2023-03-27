# -*- coding: utf-8 -*-
"""
Twitter API AUTH through Tweepy
"""
# imports
import tweepy as twp

"""
api - access through API 1.1
"""
auth = twp.AppAuthHandler(consumer_key='',
                          consumer_secret='')

api = twp.API(auth, wait_on_rate_limit=True)

"""
apiP - access through API 1.1 *Premium*
"""
authP = twp.AppAuthHandler(consumer_key='',
                           consumer_secret='')

apiP = twp.API(authP, wait_on_rate_limit=True)

"""
client - access through API 2.0
"""
client = twp.Client(
    bearer_token='',
    consumer_key='',
    consumer_secret='', wait_on_rate_limit=True)
