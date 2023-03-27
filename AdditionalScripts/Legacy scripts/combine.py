import pandas as pd

def comb():
    data_new = pd.DataFrame(
        columns=['number', 'space', 'tweet_id', 'text', 'created_at',
                 'in_reply_to_status_id', 'in_reply_to_user_id', 'Retweeters_Count',
                 'Likers_Count', 'Retweeters', 'user_id', 'verified', 'followers',
                 'friends', 'listed', 'favourites', 'statuses', 'user_created_at'])

    paths = []
    for i in range(3, 21):
        paths.append('../FinalData/Test-{}.csv'.format(i))

    for path in paths:
        try:
            iter_csv = pd.read_csv(path, iterator=True, error_bad_lines=False, engine='python')
        except pd.errors.ParserError:
            pass
        df = pd.concat(
            [chunk[
                 (chunk['space'] == 1.0) | (chunk['space'] == 0.0) | (chunk['space'] == 1) | (chunk['space'] == 0)]
             for
             chunk in iter_csv])
        data_new = data_new.append(df)

    data_new = data_new.drop(columns=['number'])
    data_new.to_csv('F:\Аспирант\Проект\Data\Full.csv')
    return data_new

comb()
