from AdditionalScripts.DBScripts.DBConnection import db_connection
import pandas as pd
import json
import warnings
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
conn = db_connection()
pd.set_option("display.max.columns", None)
pd.set_option('display.max_colwidth', None)

for tx in conn.transaction():
    with tx:
        data_json = tx.query_json("""
        select tweet {
        Hand_mark, 
        TweetData:{},
        IMData:{text_updated},
        TMData: {TM_Emotes,TM_Mean,TM_Vulgar,TM_Word_Count,TM_Sentiment}, 
        TMMData: {TM_Low_To_High_Diffustion, TM_Biggest_Connected_Component, TM_Tweets_With_URLs_Part,
        TM_Isolated_Tweets_Part}, 
        UMData: {UM_Originality, UM_Influence, UM_Role, UM_Engagement, UM_Trust},
        EMData: {EM_Hashtags, EM_Links, EM_Repeated_Symbols,EM_Repeated_Words, EM_Spelling,EM_Exclamation_Mark,
        EM_Pronoun, EM_Conditional_Words, EM_Sensory_Verbs,EM_Ordinal_Adjectives, EM_Relative_Time, EM_Numbers_Count
        ,EM_Quantity_Adjective, EM_Certainty_Words, EM_Days_Between}
        };
        """)

y = json.loads(data_json)
df = pd.json_normalize(y, max_level=1)
# df = df.drop(columns=['UMData', 'TMMData', 'TMData'])
# df = df.dropna()


def null_check():
    cols = df.columns
    is_null = df[cols].isnull()
    print(is_null)


def data_describe():
    print("Выборка содержит {} записей и {} столбцов. \n".format(df.shape[0], df.shape[1]))
    print(df.dtypes)
    hand_mark = df.groupby("Hand_mark")
    print(hand_mark.describe().head())

    plt.figure(figsize=(15, 10))
    hand_mark.size().sort_values(ascending=False).plot.bar()
    plt.xticks(rotation=50)
    plt.xlabel("Ложь или Правда")
    plt.ylabel("Количество вхождений")
    # plt.show()


def word_cloud():
    text = df['IMData.text_updated']
    stopwords = set(STOPWORDS)
    stopwords.update(["это"])
    text = " ".join(review for review in text)
    print("There are {} words in the combination of all review.".format(len(text)))

    # Display the generated image:
    wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=50,
                          background_color="rgba(255, 255, 255, 0)", mode="RGBA").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    # wordcloud.to_file("AdditionalScripts//Learning//Logs//first_review.png")


data_describe()
word_cloud()
