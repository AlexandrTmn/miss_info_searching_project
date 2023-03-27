CREATE MIGRATION m15octp6mlr5bumlkutwsrxmm7ctg4ph7ww4bdzqi3adfqvko2vnca
    ONTO m1hw3ew6oejxvmjdvm77pocp5z6j2scrw4carxqmknhtsjg4tekm5q
{
  ALTER TYPE default::tweet {
      DROP LINK TweetData;
  };
  ALTER TYPE default::tweet {
      DROP LINK UserData;
      CREATE LINK UserData -> default::User_Info_Raw;
  };
};
