CREATE MIGRATION m17q7bsmu772rjz3tzqrsjkuvlb2nywr74pydctfbq5tzpigtt34fa
    ONTO m1eyraignsyhsfln2sp3bj5ytgt5hyjny67cnecl3xk42pjq2ykqmq
{
  ALTER TYPE default::tweet {
      ALTER PROPERTY TweetId {
          CREATE CONSTRAINT std::exclusive;
      };
  };
};
