CREATE MIGRATION m1x3bhc2vhl7v2t2gv53bbigwjvh765r7tbdvnufrt4t77lpcb3eoq
    ONTO initial
{
  CREATE TYPE default::tweet {
      CREATE REQUIRED PROPERTY Hand_mark -> std::bool;
      CREATE REQUIRED PROPERTY Training_Sample -> std::bool;
      CREATE REQUIRED PROPERTY TweetId -> std::int64;
  };
  CREATE TYPE default::Intermediate_Data EXTENDING default::tweet {
      CREATE LINK IMData -> default::tweet;
      CREATE PROPERTY Created_At_Date -> std::datetime;
      CREATE PROPERTY Created_At_Time -> std::datetime;
      CREATE PROPERTY text_updated -> std::str;
  };
  CREATE TYPE default::Tweet_Info_Raw EXTENDING default::tweet {
      CREATE PROPERTY Tweet_Answer_to_Raw -> std::int64;
      CREATE PROPERTY Tweet_Created_at_Raw -> std::datetime;
      CREATE PROPERTY Tweet_Likes_Count_Raw -> std::int32;
      CREATE PROPERTY Tweet_Retweets_Count_Raw -> std::int32;
      CREATE PROPERTY Tweet_Retweets_Ids_Raw -> std::json;
      CREATE PROPERTY Tweet_Text_Raw -> std::str;
      CREATE PROPERTY User_Answer_to_Raw -> std::int64;
  };
  ALTER TYPE default::tweet {
      CREATE MULTI LINK TI -> default::Tweet_Info_Raw {
          CREATE CONSTRAINT std::exclusive;
      };
  };
  CREATE TYPE default::User_Info_Raw EXTENDING default::tweet {
      CREATE PROPERTY User_Created_at_Raw -> std::datetime;
      CREATE PROPERTY User_Followers_Count_Raw -> std::int32;
      CREATE PROPERTY User_Follows_Count_Raw -> std::int32;
      CREATE PROPERTY User_Tweets_count_Raw -> std::int32;
      CREATE PROPERTY User_Verified_Raw -> std::bool;
      CREATE PROPERTY User_likes_count_Raw -> std::int32;
  };
  ALTER TYPE default::tweet {
      CREATE MULTI LINK UI -> default::User_Info_Raw {
          CREATE CONSTRAINT std::exclusive;
      };
  };
  CREATE TYPE default::Prediction_Info EXTENDING default::tweet {
      CREATE LINK PI -> default::tweet;
      CREATE PROPERTY Prediction_Result -> std::bool;
      CREATE PROPERTY Prediction_Result_data -> std::str;
      CREATE PROPERTY Prediction_Scores -> array<std::str>;
  };
  CREATE TYPE default::Text_Metrics EXTENDING default::tweet {
      CREATE LINK TM -> default::tweet;
      CREATE PROPERTY TM_Emotes -> std::bool;
      CREATE PROPERTY TM_Mean -> std::bool;
      CREATE PROPERTY TM_Vulgar -> std::bool;
      CREATE PROPERTY TM_Word_Count -> std::int32;
  };
  CREATE TYPE default::Time_Metrics EXTENDING default::tweet {
      CREATE LINK TM -> default::tweet;
      CREATE PROPERTY TM_Biggest_Connected_Component -> std::int32;
      CREATE PROPERTY TM_Height_To_Width -> std::float32;
      CREATE PROPERTY TM_Isolated_Tweets_Part -> std::float32;
      CREATE PROPERTY TM_Low_To_High_Diffustion -> std::float32;
      CREATE PROPERTY TM_Original_Tweets_Part -> std::float32;
      CREATE PROPERTY TM_Tweets_With_URLs_Part -> std::float32;
  };
  CREATE TYPE default::User_Metrics EXTENDING default::tweet {
      CREATE LINK UM -> default::tweet;
      CREATE PROPERTY UM_Engagement -> std::float32;
      CREATE PROPERTY UM_Influence -> std::bool;
      CREATE PROPERTY UM_Originality -> std::float32;
      CREATE PROPERTY UM_Role -> std::float32;
  };
};
