module default {
    type tweet {
        required property TweetId -> int64 {
            constraint exclusive;
        }
        required property Training_Sample -> bool;
        property Hand_mark -> bool;
        link UserData -> default::User_Info_Raw {
            constraint exclusive;
        };
        link TweetData -> default::Tweet_Info_Raw {
            constraint exclusive;
        };
        link IMData -> default::Intermediate_Data {
            constraint exclusive;
        }
        link TMData -> default::Text_Metrics {
            constraint exclusive;
        }
        link UMData -> default::User_Metrics {
            constraint exclusive;
        }
        link TMMData -> default::Time_Metrics {
            constraint exclusive;
        }
        link PRData -> default::Prediction_Info {
            constraint exclusive;
        }
        link EMData -> default::Experimental_Metrics {
            constraint exclusive;
        }
    }

    type User_Info_Raw {
        property User_Verified_Raw -> bool;
        property User_Followers_Count_Raw -> int32;
        property User_Follows_Count_Raw -> int32;
        property User_likes_count_Raw -> int32;
        property User_Tweets_count_Raw -> int32;
        property User_Created_at_Raw -> datetime;
    }

    type Tweet_Info_Raw {
        property Tweet_Text_Raw -> str;
        property Tweet_Created_at_Raw -> datetime;
        property Tweet_Answer_to_Raw -> int64;
        property User_Answer_to_Raw -> int64;
        property Tweet_Likes_Count_Raw -> int32;
        property Tweet_Retweets_Count_Raw -> int32;
        property Tweet_Retweets_Ids_Raw -> array<int64>;
    }

    type Intermediate_Data {
        property Created_At_Date -> str;
        property Created_At_Time -> str;
        property text_updated -> str;
    }

    type Text_Metrics {
        property TM_Word_Count -> int32;
        property TM_Emotes -> bool;
        property TM_Vulgar -> bool;
        property TM_Mean -> bool; 
        property TM_Sentiment -> str;
    }

    type User_Metrics {
        property UM_Originality -> float32;
        property UM_Influence -> int32;
        property UM_Role -> float32;
        property UM_Engagement -> float32;
        property UM_Trust -> bool;
    }

    type Time_Metrics {
        property TM_Low_To_High_Diffustion -> float32;
        property TM_Biggest_Connected_Component -> float32;
        property TM_Tweets_With_URLs_Part -> float32;
        property TM_Isolated_Tweets_Part -> float32; 
    }

    Type Prediction_Info {
        property Prediction_Result -> bool;
        property Prediction_Result_data -> str;
        property Prediction_Scores -> array<str>;
    }

    Type Experimental_Metrics {
        property EM_Links -> bool;
        property EM_Hashtags -> bool;
        property EM_Repeated_Symbols -> float32;
        property EM_Repeated_Words -> float32;
        property EM_Spelling -> float32;
        property EM_Exclamation_Mark -> bool;
        property EM_Pronoun -> bool;
        property EM_Conditional_Words -> bool;
        property EM_Sensory_Verbs -> bool;
        property EM_Ordinal_Adjectives -> bool;
        property EM_Relative_Time -> bool;
        property EM_Numbers_Count -> float32;  
        property EM_Quantity_Adjective -> bool;  
        property EM_Certainty_Words -> bool;
        property EM_Days_Between -> float32;
    }

}