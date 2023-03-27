CREATE MIGRATION m1aefygtakcnbax5kwfqwctjjnybeiuk22n3dcfgdxv7eqswvguufa
    ONTO m1kn6awh3tvqzlzklzdi5zgxk6fox4pzvzamta2zeviaps2e2ejlfa
{
  ALTER TYPE default::Tweet_Info_Raw {
      ALTER LINK Tweet {
          RENAME TO TweetData;
      };
  };
  ALTER TYPE default::User_Info_Raw {
      ALTER LINK Tweet {
          RENAME TO UserData;
      };
  };
};
