CREATE MIGRATION m1oxqyujd7er5zdkduszjuhojquhidesjt6sc4aml6ilwlp6lbizqq
    ONTO m17q7bsmu772rjz3tzqrsjkuvlb2nywr74pydctfbq5tzpigtt34fa
{
  ALTER TYPE default::Intermediate_Data {
      DROP LINK IMData;
  };
  ALTER TYPE default::Prediction_Info {
      DROP LINK PI;
  };
  ALTER TYPE default::Text_Metrics {
      DROP LINK TM;
  };
  ALTER TYPE default::Time_Metrics {
      DROP LINK TM;
  };
  ALTER TYPE default::User_Metrics {
      DROP LINK UM;
  };
};
