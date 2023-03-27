CREATE MIGRATION m1phqekpbcceqdyrofwgzt4yymsjyaszggmrqbp3c7iftr5n2kk56a
    ONTO m1oxqyujd7er5zdkduszjuhojquhidesjt6sc4aml6ilwlp6lbizqq
{
  ALTER TYPE default::tweet {
      CREATE LINK TweetData -> default::Tweet_Info_Raw {
          CREATE CONSTRAINT std::exclusive;
      };
  };
};
