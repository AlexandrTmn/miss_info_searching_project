CREATE MIGRATION m1iopuwmct5inucvp6lvvi5y4u2yzesj7icouhle6sg3jc3dv3wyzq
    ONTO m1ba4icxun73evou36oo4vl764wzf35p3bbtakurqycaceaikzmgoa
{
  ALTER TYPE default::Text_Metrics {
      CREATE PROPERTY TM_Sentiment -> std::str;
  };
  ALTER TYPE default::Time_Metrics {
      DROP PROPERTY TM_Biggest_Connected_Component;
  };
  ALTER TYPE default::Time_Metrics {
      ALTER PROPERTY TM_Height_To_Width {
          RENAME TO TM_Biggest_Connected_Component;
      };
  };
  ALTER TYPE default::Time_Metrics {
      DROP PROPERTY TM_Original_Tweets_Part;
  };
  ALTER TYPE default::User_Metrics {
      CREATE PROPERTY UM_Trust -> std::bool;
  };
};
