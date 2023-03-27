CREATE MIGRATION m1z53lcnal54dxip75uubwr3qihf4xpqdb7cusvjfkuojchy7j2iqq
    ONTO m1c4teqhfst5onl5kfyppey34mctrfe4ze3nbgiqjo7iqtrbxrlfuq
{
  ALTER TYPE default::Tweet_Info_Raw {
      ALTER LINK TweetData {
          RENAME TO Tweet;
      };
  };
  ALTER TYPE default::User_Info_Raw {
      ALTER LINK UserData {
          RENAME TO Tweet;
      };
  };
};
