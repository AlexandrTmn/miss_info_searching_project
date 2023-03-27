CREATE MIGRATION m1nkliojcp2zctzr5ujjbv6m2wrkc3vva22fx7anusz7ubf5evomta
    ONTO m1x3bhc2vhl7v2t2gv53bbigwjvh765r7tbdvnufrt4t77lpcb3eoq
{
  ALTER TYPE default::tweet {
      DROP LINK TI;
      DROP LINK UI;
  };
};
