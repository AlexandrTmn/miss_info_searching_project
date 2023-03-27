# -*- coding: utf-8 -*-
"""
Twitter API AUTH through Tweepy
"""
# imports
import tweepy as twp

"""
api - access through API 1.1
"""
auth = twp.AppAuthHandler(consumer_key='X2V4I8HZixP14UbbrghHxa32v',
                          consumer_secret='b188hLoz78FgvQMKTnur43y8qKKa9kjkvBodHjcErThKYXUxId')

api = twp.API(auth, wait_on_rate_limit=True)

"""
apiP - access through API 1.1 *Premium*
"""
authP = twp.AppAuthHandler(consumer_key='X2V4I8HZixP14UbbrghHxa32v',
                           consumer_secret='b188hLoz78FgvQMKTnur43y8qKKa9kjkvBodHjcErThKYXUxId')

apiP = twp.API(authP, wait_on_rate_limit=True)

"""
client - access through API 2.0
"""
client = twp.Client(
    bearer_token='AAAAAAAAAAAAAAAAAAAAAHu56QAAAAAAHGPUMk22dqgUOcZh8vXGPWm9oQ4'
                 '%3DtMm5aGRcKBXMzai7D6FCsKLWITMvavIZNjwTv5fQLw8yRj2awd',
    consumer_key='X2V4I8HZixP14UbbrghHxa32v',
    consumer_secret='b188hLoz78FgvQMKTnur43y8qKKa9kjkvBodHjcErThKYXUxId', wait_on_rate_limit=True)
