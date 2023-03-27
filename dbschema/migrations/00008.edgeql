CREATE MIGRATION m1kn6awh3tvqzlzklzdi5zgxk6fox4pzvzamta2zeviaps2e2ejlfa
    ONTO m13matjxyxhm4yzrnddepxyz5ufef6b2xh5oavas7r4ek4suqefvqq
{
  ALTER TYPE default::Intermediate_Data DROP EXTENDING default::tweet;
  ALTER TYPE default::Prediction_Info DROP EXTENDING default::tweet;
  ALTER TYPE default::Text_Metrics DROP EXTENDING default::tweet;
  ALTER TYPE default::Time_Metrics DROP EXTENDING default::tweet;
  ALTER TYPE default::Tweet_Info_Raw DROP EXTENDING default::tweet;
  ALTER TYPE default::User_Info_Raw DROP EXTENDING default::tweet;
  ALTER TYPE default::User_Metrics DROP EXTENDING default::tweet;
};
