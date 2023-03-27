CREATE MIGRATION m1ba4icxun73evou36oo4vl764wzf35p3bbtakurqycaceaikzmgoa
    ONTO m1g5x5f7bu52ei757fzgfqbaf6nkg547tyvyuc2o6vx2jskmxnkbhq
{
  ALTER TYPE default::Intermediate_Data {
      DROP PROPERTY Created_At_Date;
  };
  ALTER TYPE default::Intermediate_Data {
      CREATE PROPERTY Created_At_Date -> std::str;
  };
  ALTER TYPE default::Intermediate_Data {
      DROP PROPERTY Created_At_Time;
      CREATE PROPERTY Created_At_Time -> std::str;
  };
};
