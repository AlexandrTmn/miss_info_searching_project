CREATE MIGRATION m1kifi7wkfqdqzxiiv6woslve2fdq3eleuyezfb4dkb52kwhonx7fq
    ONTO m1phqekpbcceqdyrofwgzt4yymsjyaszggmrqbp3c7iftr5n2kk56a
{
  ALTER TYPE default::Tweet_Info_Raw {
      DROP PROPERTY Tweet_Retweets_Ids_Raw;
      CREATE PROPERTY Tweet_Retweets_Ids_Raw -> array<std::int64>;
  };
};
