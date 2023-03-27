import AdditionalScripts.DBScripts.DBConnection as DBC
import json
import AdditionalScripts.Metrics.Metrics as MT

conn = DBC.db_connection()
metrix = conn.query_json(
    """
    select tweet {
    id, TMData, TMMData, UMData, TweetId,
    IMData: {
        text_updated,},
    TweetData: {Tweet_Text_Raw, Tweet_Created_at_Raw, Tweet_Retweets_Count_Raw,Tweet_Likes_Count_Raw, 
    Tweet_Retweets_Ids_Raw},
    UserData: {
    User_Created_at_Raw, User_Followers_Count_Raw, User_Follows_Count_Raw,User_Tweets_count_Raw,
    User_Verified_Raw,User_likes_count_Raw},
    };
    """
)
y = json.loads(metrix)
for i in range(len(y)):
    for tx in conn.transaction():
        with tx:
            if (y[i]['TMData'] is None) and (y[i]['TMMData'] is None) and (y[i]['UMData'] is None) and \
                    y[i]['TweetData']['Tweet_Retweets_Count_Raw'] <= 10:

                tm = MT.Metrics.text_metrics(y[i]['IMData']['text_updated'])
                if tm == 0:
                    tm = 0, False, False, False, [{'Skip': 0}]
                um = MT.Metrics.user_metrics(y[i])
                if not um:
                    um = [0, 0, 0, 0, False]

                if y[i]['TweetData']['Tweet_Retweets_Count_Raw'] > 0:
                    tmm = MT.Metrics.timeline(user=y[i])
                    if tmm is False or tmm == 0:
                        tmm = [0, 0, 0, 0]
                else:
                    tmm = [0, 0, 0, 0]
                print(tmm)
                metrix = tx.query_single(
                    """
                    UPDATE tweet filter .id = <uuid>$id
                    SET {TMData:=(INSERT Text_Metrics {
                            TM_Emotes:=<bool>$emotes, TM_Mean:=<bool>$mean, TM_Vulgar:=<bool>$vulgar,
                            TM_Word_Count:=<int32>$wrd_count, TM_Sentiment:=<str>$sen})};
                    """, emotes=tm[3], mean=tm[2], vulgar=tm[1], wrd_count=tm[0], sen=list(tm[4][0].keys())[0],
                    id=y[i]['id'])

                metrix = tx.query_single(
                    """
                    UPDATE tweet filter .id = <uuid>$id
                    SET {TMMData:=(INSERT Time_Metrics {
                            TM_Low_To_High_Diffustion:=<float32>$lthd, TM_Biggest_Connected_Component:=<float32>$lcc,
                            TM_Tweets_With_URLs_Part:=<float32>$twup, TM_Isolated_Tweets_Part:=<float32>$itp})};
                    """, lthd=tmm[0], lcc=tmm[1], twup=tmm[2], itp=tmm[3], id=y[i]['id'])

                metrix = tx.query_single(
                    """
                    UPDATE tweet filter .id = <uuid>$id
                    SET {UMData:=(INSERT User_Metrics {
                            UM_Originality:=<float32>$og, UM_Influence:=<int32>$inf, UM_Role:=<float32>$role,
                            UM_Engagement:=<float32>$eng,UM_Trust:=<bool>$trust})};
                    """, og=um[3], inf=um[0], role=um[1], eng=um[4], trust=bool(um[2]), id=y[i]['id'])
