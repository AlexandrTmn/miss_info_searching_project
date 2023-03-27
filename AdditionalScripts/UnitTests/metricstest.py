import unittest as nt
import AdditionalScripts.Metrics.Metrics as MT
import AdditionalScripts.DBScripts.DBConnection as DBC
import json

conn = DBC.db_connection()

for tx in conn.transaction():
    with tx:
        metrix = tx.query_json(
            """
            select tweet {
            id, TMData, TMMData, UMData, TweetId,
            IMData: {
                text_updated,},
            TweetData: {Tweet_Text_Raw, Tweet_Created_at_Raw, Tweet_Retweets_Count_Raw,Tweet_Likes_Count_Raw},
            UserData: {
            User_Created_at_Raw, User_Followers_Count_Raw, User_Follows_Count_Raw,User_Tweets_count_Raw,
            User_Verified_Raw,User_likes_count_Raw},
            };
            """
        )
    y = json.loads(metrix)


class TestMetrics(nt.TestCase):
    def testspeed(self):
        # self.assertEqual(MT.Metrics.textmet(y[0]['IMData']['text_updated']), (6, True, False, False, [{'negative': 0.9890230894088745}]))
        self.assertEqual(MT.Metrics.user_metrics(y[0]), (3043, 1.0795267827801511, False, 0.06, 1272.6))

