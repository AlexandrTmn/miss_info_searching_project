import time

from AdditionalScripts.DBScripts import DBInsert
import pandas as pd
import datetime as dt
from AdditionalScripts import Auth
import re
import AdditionalScripts.DBScripts.DBConnection as DBC


def formating(dictrow: dict):
    row = dictrow
    if row['space'] == 0.0:
        row['space'] = False
    else:
        row['space'] = True

    row['created_at'] = dt.datetime.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S%z')
    row['user_created_at'] = dt.datetime.strptime(row['user_created_at'], '%Y-%m-%d %H:%M:%S%z')

    if row['Retweeters_Count'] != 0:
        try:
            retw = Auth.client.get_retweeters(id=row['tweet_id'], max_results=100)
        except ConnectionAbortedError:
            time.time(10)
            retw = Auth.client.get_retweeters(id=row['tweet_id'])
        row['Retweeters'] = [int(s) for s in re.findall(r'=(\d+)', str(retw.data))]
    else:
        row['Retweeters'] = []

    if row['in_reply_to_status_id'] == 'None':
        row['in_reply_to_status_id'] = 0

    if row['in_reply_to_user_id'] == 'None':
        row['in_reply_to_user_id'] = 0

    return row


conn = DBC.db_connection()
data_full = pd.read_csv('../../Data/Full.csv')
for i in range(0, len(data_full)):
    row = data_full.loc[i, :].to_dict()
    idcheck = conn.query(
        """
        SELECT tweet {TweetId} FILTER .TweetId = <int64>$tweet_id
        """, tweet_id=row['tweet_id'])
    if list(idcheck) == []:
        row_edit = formating(row)
        DBInsert.data_insert(row_edit)

