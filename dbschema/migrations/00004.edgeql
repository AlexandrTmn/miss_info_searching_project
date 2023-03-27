CREATE MIGRATION m1g6lbilkeiwnk5ti2y7bdggb3va54vx5onhcgkp3epuxxv22ikrca
    ONTO m1fuhvorhjjf2dcrvfjfij46kuoqfcwetkyptyodicnuvl5op5od4a
{
  ALTER TYPE default::Tweet_Info_Raw {
      DROP LINK TI;
  };
  ALTER TYPE default::Tweet_Info_Raw {
      CREATE MULTI LINK TweetData -> default::tweet;
  };
  ALTER TYPE default::User_Info_Raw {
      DROP LINK UI;
  };
  ALTER TYPE default::User_Info_Raw {
      CREATE MULTI LINK UserData -> default::tweet;
  };
};
