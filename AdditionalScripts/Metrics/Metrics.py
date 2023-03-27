# -*- coding: utf-8 -*-
"""
Legacy metrics.

Getting legacy metrics from tweet data in database.
emotes (смайлики)
vulgar (вульгарные выражения)
sentiment (настроение)
rt_check (проверка ретвита)
word_count (количество слов)
mean (слова мнения)
reply_check (проверка ответа)
originality (оригинальность пользователя)
verified (проверка на доверие)
user_registration_year (год регистрации пользователя)
influence (влияние)
likes (отметки "нравится")
engagement (вовлечение пользователя)
role (роль)
check_relations (проверка связи пользователей)
nodes_with_urls_diffusion (доля узлов с ссылками)
low_to_high_diffusion (низкая - высокая диффузия)
isolated_diffusion (изолированные узлы)
longest_component_diffusion (наиболее связный компонент)
check_relations_with_follows (проверка связи с подписками)
"""

# imports
import emojis
import re
import treelib.exceptions
import tweepy.errors
from AdditionalScripts import Auth
import datetime as dt
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
import fasttext
from treelib import Tree

# checking if sentiment model is exist
FastTextSocialNetworkModel.MODEL_PATH = 'D:/fasttext-social-network-model.bin'
try:
    fasttext.FastText.eprint = lambda *args, **kwargs: None
except:
    pass

# constants
user_tweet_count = 0
user_id = 0
user_reply_count = 0


# class for getting legacy metrics
class Metrics:
    # getting text metrics
    @classmethod
    def text_metrics(cls, text: str, u_text: str):
        # getting text sentiment
        def sentiment():
            list_of_text = [text]
            tokenizer = RegexTokenizer()
            model = FastTextSocialNetworkModel(tokenizer=tokenizer)
            results = model.predict(list_of_text, k=1)
            return results

        # getting count of words in tweet text
        def word_count():
            return len(text.split(sep=' '))

        # getting emotes in tweet text
        def emotes():
            emojis.get(text)
            if emojis.count(text) != 0:
                return True
            else:
                return False

        # checking if vulgar words are in tweet text
        def vulgar():
            vlg_file_path = r'AdditionalScripts/Metrics/Data/Vulgar.txt'
            with open(vlg_file_path, encoding='utf8') as file:
                raw_lst = [line.rstrip() for line in file]
                regex = re.compile("(?=(" + "|".join(map(re.escape, raw_lst)) + "))")
                mn = re.findall(regex, u_text)
                if len(mn) != 0:
                    return True
                else:
                    return False

        # checking if mean words are in tweet text
        def mean():
            mean_file = 'AdditionalScripts/Metrics/Data/Mean.txt'
            with open(mean_file) as file:
                mean_list = [line.rstrip() for line in file]
            regex = re.compile("(?=(" + "|".join(map(re.escape, mean_list)) + "))")
            mn = re.findall(regex, u_text)
            if len(mn) != 0:
                return True
            else:
                return False

        return word_count(), vulgar(), mean(), emotes(), sentiment()

    # user metrics
    @classmethod
    def user_metrics(cls, user: dict, tweet_id):
        # getting additional data for user metrics
        def additional_data(api, tweet_id):
            global user_reply_count
            global user_id
            global user_tweet_count

            # checking if tweet is retweet
            def rt_check(text):
                if 'RT' in str(text):
                    return True
                else:
                    return False

            # checking if tweet is reply
            def reply_check(text):
                if '@' in str(text):
                    return True
                else:
                    return False

            try:
                tweet_id = api.get_tweet(tweet_id, tweet_fields=['author_id'], expansions=['author_id'])
            except tweepy.errors.Forbidden:
                return 0

            if tweet_id[1]:
                user_id_s = re.findall(r'=(\d+)', str(tweet_id[1]['users']))
            else:
                return False

            rt_count = 0
            timeline = api.get_users_tweets(user_id_s[0], max_results=100)
            if timeline.meta['result_count'] == 0:
                rt_count = 0
                user_reply_count = 0
            else:
                user_tweet_count = (len(timeline.data))
                for i in range(user_tweet_count):
                    if rt_check(timeline.data[i]):
                        rt_count += 1
                    if reply_check(timeline.data[i]):
                        user_reply_count += 1
                return rt_count, user_reply_count, user_tweet_count

        # getting originality of user
        def originality(ad_data: list):
            if ad_data is not None:
                return ad_data[0] / ad_data[2]
            else:
                return 0

        # getting if user was verified by Twitter
        def verified():
            return user['verified']

        # getting user influence
        def influence():
            return user['followers']

        # getting user role
        def role():
            if user['followers'] != 0:
                return user['friends_count'] / user['followers']
            else:
                return 1

        # getting user registration year
        def user_registration_year(basic_data):
            d0 = dt.datetime.strptime(basic_data['user_created_at'], '%a %b %d %H:%M:%S %z %Y')
            d1 = dt.datetime.strptime(basic_data['created_at'], '%a %b %d %H:%M:%S %z %Y')
            ury = d1.year - d0.year
            return ury

        # getting user engagement into Twitter system
        def engagement(ury: float, repl: int, lks: int):
            if ury == 0:
                eng = ((user['followers'] + user['friends_count'] + lks + repl) / 0.01)
            else:
                eng = ((user['followers'] + user['friends_count'] + lks + repl) / ury)
            return eng

        # getting links from Twitter API
        def likes(api):
            return len(api.get_liked_tweets(id=tweet_id))

        additional_data = additional_data(api=Auth.client, tweet_id=tweet_id)
        if not additional_data:
            return additional_data
        else:
            return influence(), role(), verified(), originality(ad_data=additional_data), engagement(
                user_registration_year(user), repl=user_reply_count, lks=likes(api=Auth.client))

    # building tweet diffusion tree
    @classmethod
    def timeline(cls, user, tweet_id):
        og_user_followers_count = user['followers']
        isolated = low_to_high = high_to_low = is_quote_status = url_count = 0
        checked = []
        dict_of_retweets = {}
        api = Auth.apiP
        tree = Tree()
        try:
            rt = api.get_retweets(id=tweet_id, count=100)
        except tweepy.errors.Forbidden:
            return 0
        except tweepy.errors.NotFound:
            return 0

        if not rt:
            return False
        # Getting original tweet id
        if rt[0].user:
            og_id = rt[0]._json['retweeted_status']['user']['id']
        else:
            return 0
        tree.create_node(str(og_id), str(og_id))
        width = len(rt)

        for tweet in rt:
            tweet = tweet._json
            time = dt.datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S %z %Y')
            dict_of_retweets[tweet['user']['id']] = [time, tweet['is_quote_status'],
                                                     tweet['favorite_count'], tweet['user']['followers_count']]
            if tweet['entities']['urls'] != 0:
                url_count += 1
            if tweet['is_quote_status']:
                is_quote_status += 1
            if tweet['favorited'] is False and tweet['retweeted'] is False:
                isolated += 1

        # checking relation between original user and follower
        def check_relations(sid, tid):
            return api.get_friendship(source_id=sid, target_id=tid)

        dict_of_retweets = dict(sorted(dict_of_retweets.items(), key=lambda p: p[1]))
        dict_of_retweets_copy = dict_of_retweets.copy()

        # checking if user follows original tweet user
        def check_relations_with_follows(ids_list: list, tid: int):
            if tid in ids_list:
                return True
            else:
                return False

        # checking if user has less than 5000 followers for API and building diffusion tree
        if og_user_followers_count < 5000:
            og_user_followers_id = api.get_follower_ids(user_id=og_id)
            for item in dict_of_retweets.copy():
                if check_relations_with_follows(ids_list=og_user_followers_id, tid=item):
                    try:
                        tree.create_node(str(item), str(item), parent=str(og_id))
                    except treelib.exceptions.DuplicatedNodeIdError:
                        continue
                    if og_user_followers_count > dict_of_retweets_copy[item][3]:
                        high_to_low += 1
                    else:
                        low_to_high += 1
                    dict_of_retweets.pop(item)
                    checked.append(item)
                    continue
        else:
            for item in dict_of_retweets.copy():
                relation = check_relations(item, og_id)
                if relation[0]._json['following']:
                    try:
                        tree.create_node(str(item), str(item), parent=str(og_id))
                    except treelib.exceptions.DuplicatedNodeIdError:
                        continue
                    if og_user_followers_count > dict_of_retweets_copy[item][3]:
                        high_to_low += 1
                    else:
                        low_to_high += 1

                    dict_of_retweets.pop(item)
                    checked.append(item)
                    continue

        if len(dict_of_retweets) == 1:
            for item in checked:
                try:
                    itb = list(dict_of_retweets.keys())[0]
                except:
                    continue

                relation = check_relations(itb, item)
                if relation[0]._json['following']:
                    try:
                        tree.create_node(str(itb), str(itb), parent=str(item))
                    except treelib.exceptions.DuplicatedNodeIdError:
                        continue
                    if dict_of_retweets_copy[itb][3] > dict_of_retweets_copy[item][3]:
                        high_to_low += 1
                    else:
                        low_to_high += 1
                    dict_of_retweets.pop(itb)
                    continue
                else:
                    try:
                        tree.create_node(str(itb), str(itb), parent=str(og_id))
                    except treelib.exceptions.DuplicatedNodeIdError:
                        continue
                    dict_of_retweets.pop(itb)
                    continue

        # Checking follows of followers
        if len(dict_of_retweets) > 1:
            for i in dict_of_retweets.copy():
                for item in checked:
                    relation = check_relations(i, item)
                    if relation[0]._json['following']:
                        try:
                            tree.create_node(str(i), str(i), parent=str(item))
                        except treelib.exceptions.DuplicatedNodeIdError:
                            continue
                        if dict_of_retweets_copy[i][3] > dict_of_retweets_copy[item][3]:
                            high_to_low += 1
                        else:
                            low_to_high += 1
                        dict_of_retweets.pop(i)
                        continue

        # Last id's merge to og id
        if len(dict_of_retweets) > 1:
            for item in dict_of_retweets.copy():
                try:
                    tree.create_node(str(item), str(item), parent=str(og_id))
                except treelib.exceptions.DuplicatedNodeIdError:
                    continue
                if og_user_followers_count > dict_of_retweets_copy[item][3]:
                    high_to_low += 1
                else:
                    low_to_high += 1
                dict_of_retweets.pop(item)
                continue

        # getting low-high diffusions from diffusion tree
        def low_to_high_diffusion(lth: int, htl: int):
            return lth / (lth + htl)

        # getting the longest connected component from diffusion tree
        def longest_component_diffusion(llc: int, all_nodes: int):
            return llc / (int(all_nodes) + 1)

        # getting part of nodes with urls from diffusion tree
        def nodes_with_urls_diffusion(nwu: int, all_nodes: int):
            return nwu / (int(all_nodes) + 1)

        # getting part of isolated nodes from diffusion tree
        def isolated_diffusion(isn: int, all_nodes: int):
            return isn / (int(all_nodes) + 1)

        return low_to_high_diffusion(low_to_high, width), longest_component_diffusion(tree.depth(),
                                                                                      width), nodes_with_urls_diffusion(
            url_count, width), isolated_diffusion(isolated, width)
