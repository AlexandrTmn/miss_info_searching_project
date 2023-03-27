CREATE MIGRATION m1cfnsurwttuimmn66gicgqn57kli7io3u4dhk3yv57dnxcdfhvmfq
    ONTO m1jgp2jt4jxnajwmpljlg4gb2ackm73h43qqzptggurx5mxyjxvtoq
{
  ALTER TYPE default::Experimental_Metrics {
      CREATE PROPERTY EM_Certainty_Words -> std::bool;
      CREATE PROPERTY EM_Conditional_Words -> std::bool;
      CREATE PROPERTY EM_Days_Between -> std::float32;
      CREATE PROPERTY EM_Exclamation_Mark -> std::bool;
      CREATE PROPERTY EM_Numbers_Count -> std::float32;
      CREATE PROPERTY EM_Ordinal_Adjectives -> std::bool;
      CREATE PROPERTY EM_Pronoun -> std::bool;
      CREATE PROPERTY EM_Quantity_Adjective -> std::bool;
      CREATE PROPERTY EM_Relative_Time -> std::bool;
      CREATE PROPERTY EM_Sensory_Verbs -> std::bool;
      CREATE PROPERTY EM_Spelling -> std::float32;
  };
};
