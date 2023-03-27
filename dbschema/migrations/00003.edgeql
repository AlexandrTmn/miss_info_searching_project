CREATE MIGRATION m1fuhvorhjjf2dcrvfjfij46kuoqfcwetkyptyodicnuvl5op5od4a
    ONTO m1nkliojcp2zctzr5ujjbv6m2wrkc3vva22fx7anusz7ubf5evomta
{
  ALTER TYPE default::Tweet_Info_Raw {
      CREATE LINK TI -> default::tweet;
  };
  ALTER TYPE default::User_Info_Raw {
      CREATE LINK UI -> default::tweet;
  };
};
