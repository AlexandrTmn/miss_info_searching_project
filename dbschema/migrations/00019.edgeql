CREATE MIGRATION m1j3tarbtwia2g4gegnuilmd6t6kidwj3ckzrvo7f455fk5ztsyasa
    ONTO m1kifi7wkfqdqzxiiv6woslve2fdq3eleuyezfb4dkb52kwhonx7fq
{
  ALTER TYPE default::tweet {
      CREATE LINK IMData -> default::Intermediate_Data {
          CREATE CONSTRAINT std::exclusive;
      };
  };
};
