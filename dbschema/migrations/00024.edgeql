CREATE MIGRATION m1njspm3ckl5etuizxcsqwrkd65kaau4mq4g4iu7wp33kkl2pkzioq
    ONTO m1q6473t5r5msz2kgpg5xbqer4zbewk6kdabmy265tmozaqpr2omca
{
  ALTER TYPE default::User_Metrics {
      CREATE PROPERTY UM_Trust -> std::bool;
  };
};
