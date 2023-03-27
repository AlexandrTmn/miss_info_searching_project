import AdditionalScripts.DBScripts.DBConnection as DBc

conn = DBc.db_connection()


def data_insert(data: dict):
    """
    data_insert

    Inserting all data into database
    """
    conn.query(
        """
        INSERT tweet {
        Training_Sample:=<bool>$Training_Sample,
        TweetId:=<int64>$TweetId,
        Hand_mark:=<bool>$Hand_mark,
        UserData:=(INSERT User_Info_Raw {
                User_Verified_Raw:=<bool>$User_Verified_Raw, 
                User_Followers_Count_Raw:=<int32>$User_Followers_Count_Raw, 
                User_Follows_Count_Raw:=<int32>$User_Follows_Count_Raw, 
                User_likes_count_Raw:=<int32>$User_likes_count_Raw,  
                User_Tweets_count_Raw:=<int32>$User_Tweets_count_Raw, 
                User_Created_at_Raw:=<datetime>$User_Created_at_Raw }),
        TweetData:=(INSERT Tweet_Info_Raw {
                Tweet_Text_Raw:=<str>$Tweet_Text_Raw,
                Tweet_Created_at_Raw:=<datetime>$Tweet_Created_at_Raw,
                Tweet_Answer_to_Raw:=<int64>$Tweet_Answer_to_Raw,
                User_Answer_to_Raw:=<int64>$User_Answer_to_Raw,
                Tweet_Likes_Count_Raw:=<int32>$Tweet_Likes_Count_Raw,
                Tweet_Retweets_Count_Raw:=<int32>$Tweet_Retweets_Count_Raw,
                Tweet_Retweets_Ids_Raw:=<array<std::int64>>$Tweet_Retweets_Ids_Raw }),
        IMData:=(INSERT Intermediate_Data {
                Created_At_Date:=<str>$Created_At_Date,
                Created_At_Time:=<str>$Created_At_Time,
                text_updated:=<str>$text_updated}),
        UMData:=(INSERT User_Metrics {
                UM_Engagement:=<float32>$UM_Engagement,
                UM_Influence:=<int32>$UM_Influence,
                UM_Originality:=<float32>$UM_Originality,
                UM_Role:=<float32>$UM_Role,
                UM_Trust:=<bool>$UM_Trust}),
        TMData:=(INSERT Text_Metrics {
                TM_Emotes:=<bool>$TM_Emotes,
                TM_Mean:=<bool>$TM_Mean,
                TM_Sentiment:=<str>$TM_Sentiment,
                TM_Vulgar:=<bool>$TM_Vulgar,
                TM_Word_Count:=<int32>$TM_Word_Count}),
        TMMData:=(INSERT Time_Metrics {
                TM_Biggest_Connected_Component:=<float32>$TMM_Biggest_Connected_Component,
                TM_Isolated_Tweets_Part:=<float32>$TMM_Isolated_Tweets_Part,
                TM_Low_To_High_Diffustion:=<float32>$TMM_Low_To_High_Diffustion,
                TM_Tweets_With_URLs_Part:=<float32>$TMM_Tweet_With_URLs_Part}),        
        EMData:=(INSERT Experimental_Metrics {
                EM_Hashtags:=<bool>$EM_Hashtags,
                EM_Links:=<bool>$EM_Links,
                EM_Repeated_Symbols:=<float32>$EM_Repeated_Symbols,
                EM_Repeated_Words:=<float32>$EM_Repeated_Words,
                EM_Certainty_Words:=<bool>$EM_Certainty_Words,
                EM_Conditional_Words:=<bool>$EM_Conditional_Words,
                EM_Days_Between:=<float32>$EM_Days_Between,
                EM_Exclamation_Mark:=<bool>$EM_Exclamation_Mark,
                EM_Numbers_Count:=<float32>$EM_Numbers_Count,
                EM_Ordinal_Adjectives:=<bool>$EM_Ordinal_Adjectives,
                EM_Pronoun:=<bool>$EM_Pronoun,
                EM_Quantity_Adjective:=<bool>$EM_Quantity_Adjective,
                EM_Relative_Time:=<bool>$EM_Relative_Time,
                EM_Sensory_Verbs:=<bool>$EM_Sensory_Verbs,
                EM_Spelling:=<float32>$EM_Spelling}),
        }
        UNLESS conflict on .TweetId
        """,
        # Basic data
        Training_Sample=False, TweetId=int(data['tweet_id']), Hand_mark=False,
        User_Verified_Raw=data['verified'], User_Followers_Count_Raw=data['followers'],
        User_Follows_Count_Raw=data['friends_count'],
        User_likes_count_Raw=data['likes_count'], User_Tweets_count_Raw=data['statuses'],
        User_Created_at_Raw=data['user_created_at'],
        # Raw tweet data
        Tweet_Text_Raw=data['text'], Tweet_Created_at_Raw=data['created_at'],
        Tweet_Answer_to_Raw=int(data['in_reply_to_status_id']), User_Answer_to_Raw=int(data['in_reply_to_user_id']),
        Tweet_Likes_Count_Raw=data['Likers_Count'], Tweet_Retweets_Count_Raw=data['Retweeters_Count'],
        Tweet_Retweets_Ids_Raw=data['Retweeters'],
        # Updated tweet data
        Created_At_Date=data['date'],
        Created_At_Time=data['time'],
        text_updated=data['text'],
        # User metrics
        UM_Engagement=data['engagement'],
        UM_Influence=data['influence'],
        UM_Originality=data['originality'],
        UM_Role=data['role'],
        UM_Trust=data['verified'],
        # Text metrics
        TM_Emotes=data['emotes'],
        TM_Mean=data['mean'],
        TM_Sentiment=list((data['sentiment'][0]).keys())[0],
        TM_Vulgar=data['vulgar'],
        TM_Word_Count=data['word_count'],
        # Timeline metrics
        TMM_Biggest_Connected_Component=data['longest_component_diffusion'],
        TMM_Isolated_Tweets_Part=data['isolated_diffusion'],
        TMM_Low_To_High_Diffustion=data['low_to_high_diffusion'],
        TMM_Tweet_With_URLs_Part=data['nodes_with_urls_diffusion'],
        # Experimental metrics
        EM_Hashtags=data['hashtags'],
        EM_Links=data['links'],
        EM_Repeated_Symbols=data['repeat_symbols_and_words'][0],
        EM_Repeated_Words=data['repeat_symbols_and_words'][1],
        EM_Certainty_Words=data['certainty_words'],
        EM_Conditional_Words=data['conditional_words'],
        EM_Days_Between=data['user_registration_year'],
        EM_Exclamation_Mark=data['exclamation_mark'],
        EM_Numbers_Count=data['numbers_count'],
        EM_Ordinal_Adjectives=data['ordinal_adjectives'],
        EM_Pronoun=data['pronoun'],
        EM_Quantity_Adjective=data['quantity_adjective'],
        EM_Relative_Time=data['relative_time'],
        EM_Sensory_Verbs=data['sensory_verbs'],
        EM_Spelling=data['spell_check'], )

    return True
