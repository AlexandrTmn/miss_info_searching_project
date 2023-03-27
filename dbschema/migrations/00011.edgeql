CREATE MIGRATION m1zj3kpjer57ix272rlbucruddaxvwfkrkal5zr2nt2ukk4h666nfq
    ONTO m1comfpondc7cmicmylghnxe66ft6rpmiywwa4fg2gmjv3kkpjqwoq
{
  ALTER TYPE default::tweet {
      DROP LINK TweetData;
  };
  ALTER TYPE default::tweet {
      CREATE LINK TweetData -> default::Tweet_Info_Raw;
  };
  ALTER TYPE default::tweet {
      DROP LINK UserData;
      CREATE LINK UserData -> default::User_Info_Raw;
  };
};
