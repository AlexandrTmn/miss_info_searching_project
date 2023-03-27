CREATE MIGRATION m1qc32nnqaysxehx3jfqfpy7rqkw4uspi63jncmzkh3kakkejee6ia
    ONTO m1njspm3ckl5etuizxcsqwrkd65kaau4mq4g4iu7wp33kkl2pkzioq
{
  ALTER TYPE default::User_Metrics {
      DROP PROPERTY UM_Influence;
      CREATE PROPERTY UM_Influence -> std::int32;
  };
};
