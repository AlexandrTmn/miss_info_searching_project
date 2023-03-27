CREATE MIGRATION m1jgp2jt4jxnajwmpljlg4gb2ackm73h43qqzptggurx5mxyjxvtoq
    ONTO m1qc32nnqaysxehx3jfqfpy7rqkw4uspi63jncmzkh3kakkejee6ia
{
  CREATE TYPE default::Experimental_Metrics {
      CREATE PROPERTY EM_Hashtags -> std::bool;
      CREATE PROPERTY EM_Links -> std::bool;
      CREATE PROPERTY EM_Repeated_Symbols -> std::float32;
      CREATE PROPERTY EM_Repeated_Words -> std::float32;
  };
  ALTER TYPE default::tweet {
      CREATE LINK EMData -> default::Experimental_Metrics {
          CREATE CONSTRAINT std::exclusive;
      };
  };
};
