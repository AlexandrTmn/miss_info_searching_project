CREATE MIGRATION m1lisorxeflkfixxfignz2bdeim46fpkthe3i4hkfvo42asfabh4ua
    ONTO m1zjxcinz2uupzaigchju5uepmwanbuwytoiwffgc6zsv6y7elnjxa
{
  DROP FUTURE nonrecursive_access_policies;
  ALTER TYPE default::tweet {
      ALTER PROPERTY Hand_mark {
          RESET OPTIONALITY;
      };
  };
};
