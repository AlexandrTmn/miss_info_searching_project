CREATE MIGRATION m1q6473t5r5msz2kgpg5xbqer4zbewk6kdabmy265tmozaqpr2omca
    ONTO m1iopuwmct5inucvp6lvvi5y4u2yzesj7icouhle6sg3jc3dv3wyzq
{
  ALTER TYPE default::User_Metrics {
      DROP PROPERTY UM_Trust;
  };
};
