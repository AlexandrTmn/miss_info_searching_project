CREATE MIGRATION m1eyraignsyhsfln2sp3bj5ytgt5hyjny67cnecl3xk42pjq2ykqmq
    ONTO m15octp6mlr5bumlkutwsrxmm7ctg4ph7ww4bdzqi3adfqvko2vnca
{
  ALTER TYPE default::tweet {
      ALTER LINK UserData {
          CREATE CONSTRAINT std::exclusive;
      };
  };
};
