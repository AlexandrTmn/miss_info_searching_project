CREATE MIGRATION m1g5x5f7bu52ei757fzgfqbaf6nkg547tyvyuc2o6vx2jskmxnkbhq
    ONTO m1j3tarbtwia2g4gegnuilmd6t6kidwj3ckzrvo7f455fk5ztsyasa
{
  ALTER TYPE default::tweet {
      CREATE LINK PRData -> default::Prediction_Info {
          CREATE CONSTRAINT std::exclusive;
      };
      CREATE LINK TMData -> default::Text_Metrics {
          CREATE CONSTRAINT std::exclusive;
      };
      CREATE LINK TMMData -> default::Time_Metrics {
          CREATE CONSTRAINT std::exclusive;
      };
      CREATE LINK UMData -> default::User_Metrics {
          CREATE CONSTRAINT std::exclusive;
      };
  };
};
