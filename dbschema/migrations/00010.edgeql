CREATE MIGRATION m1comfpondc7cmicmylghnxe66ft6rpmiywwa4fg2gmjv3kkpjqwoq
    ONTO m1aefygtakcnbax5kwfqwctjjnybeiuk22n3dcfgdxv7eqswvguufa
{
  ALTER TYPE default::Tweet_Info_Raw {
      DROP LINK TweetData;
  };
  ALTER TYPE default::User_Info_Raw {
      DROP LINK UserData;
  };
  ALTER TYPE default::tweet {
      CREATE MULTI LINK TweetData -> default::Tweet_Info_Raw;
      CREATE MULTI LINK UserData -> default::User_Info_Raw;
  };
};
