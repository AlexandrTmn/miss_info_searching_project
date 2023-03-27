CREATE MIGRATION m1hw3ew6oejxvmjdvm77pocp5z6j2scrw4carxqmknhtsjg4tekm5q
    ONTO m1zj3kpjer57ix272rlbucruddaxvwfkrkal5zr2nt2ukk4h666nfq
{
  ALTER TYPE default::tweet {
      ALTER LINK TweetData {
          SET MULTI;
      };
  };
  ALTER TYPE default::tweet {
      ALTER LINK UserData {
          SET MULTI;
      };
  };
};
