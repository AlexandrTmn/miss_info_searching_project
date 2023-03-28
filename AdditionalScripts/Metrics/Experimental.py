# -*- coding: utf-8 -*-
"""
Experimental metrics.

Getting experimental metrics from tweet data in database.
user_registration_year (дата регистрации пользователя (год))
repeat_symbols_and_words (повторяющиеся символы и слова)
hashtags (наличие хэштегов)
links (наличие ссылок)
spell_check (проверка правописания)
exclamation_mark (наличие восклицательного знака)
pronoun (наличие местоимений)
conditional_words (наличие слов указывающий на условность)
sensory_verbs (наличие слова указывающих на чувства)
ordinal_adjectives (наличие слова порядка)
relative_time (наличие слова указывающих на относительное время)
numbers_count (количество цифр)
quantity_adjective (наличие слова указывающих на количество)
certainty_words (слова выражающие уверенность)
"""

# imports
import warnings
import datetime as dt
import re
import enchant

# ignoring warnings from libraries
warnings.filterwarnings("ignore")

# checking if ru dict exist
try:
    checker = enchant.Dict("ru_RU")
except enchant.errors.Error:
    pass


# Getting user registration year
def user_registration_year(basic_data: dict):
    d0 = dt.datetime.strptime(basic_data['user_created_at'], '%a %b %d %H:%M:%S %z %Y')
    d1 = dt.datetime.strptime(basic_data['created_at'], '%a %b %d %H:%M:%S %z %Y')
    ury = d1 - d0
    return ury.days


# Getting repeating symbols and words in tweet text
def repeat_symbols_and_words(text):
    repeated_symbols_count = []
    pattern = re.compile(r"\b(\w+)(?:\W\1\b)+", re.DOTALL)
    repeated_words_count = len(pattern.findall(text))
    pattern_symbols = re.compile("\\b(?=\\w*(\\w)\\1)\\w+\\b", re.DOTALL)
    for i in text.split():
        if pattern_symbols.findall(i):
            repeated_symbols_count.append(True)
        else:
            repeated_symbols_count.append(False)

    return (repeated_words_count / (len(text) - 1)), (repeated_symbols_count.count(True) / (len(text) - 1))


# Getting #hashtags in tweet text
def hashtags(text):
    if '#' in text:
        return True
    else:
        return False


# Getting link in tweet text
def links(text):
    link_pattern = re.compile('https?://\S+|www\.\S+', re.DOTALL)
    if link_pattern.findall(text):
        return True
    else:
        return False


# Getting correction of tweet text
def spell_check(text: str):
    bad_spelling = []
    for i in text.split():
        if checker.check(i):
            bad_spelling.append(True)
        else:
            bad_spelling.append(False)

    spelling = (bad_spelling.count(True) / (len(text) - 1))
    return spelling


# Getting exclamation (!) marks in tweet text
def exclamation_mark(text: str):
    pattern = re.compile(r"(\!)", re.DOTALL)
    exclamation_mark_count = len(pattern.findall(text))
    if exclamation_mark_count:
        return True
    else:
        return False


# Getting pronouns in tweet text
def pronoun(text: str):
    list_of_pronouns = ['я', 'ты', 'мы', 'вы', 'он', 'она', 'оно', 'они', 'себя',
                        'мой', 'твой', 'наш', 'ваш', 'свой', 'кто', 'что', 'каков', 'который', 'чей', 'сколько',
                        'мой', 'твой', 'свой', 'ваш', 'наш', 'его', 'её', 'никто', 'ничто', 'никакой', 'ничей',
                        'некого', 'нечего', 'незачем', 'всякий', 'каждый', 'сам', 'самый', 'любой', 'иной', 'другой',
                        'весь', 'некто', 'весь', 'нечто', 'некоторый', 'несколько', 'кто-то', 'что-нибудь',
                        'какой-либо']
    if re.compile('|'.join(list_of_pronouns), re.IGNORECASE).search(text):
        return True
    else:
        return False


# Getting conditional words in tweet text
def conditional_words(text: str):
    list_of_conditional_words = ['если', 'когда', 'раз', 'кабы', 'ежели', 'как скоро', 'как', 'коли', 'коль', 'буде',
                                 'если бы', 'раз', 'коль скоро', 'кабы', 'когда бы', 'при условии', 'при том условии',
                                 'в случае', 'в том случае']
    if re.compile('|'.join(list_of_conditional_words), re.IGNORECASE).search(text):
        return True
    else:
        return False


# Getting sensory verbs in tweet text
def sensory_verbs(text: str):
    list_of_sensory_verbs = ['увидел', 'видел', 'услышал', 'слушал', 'слышал', 'слышу', 'вижу',
                             'почувствовал', 'чувствовал', 'чувствовать', 'слышать', 'видеть']

    if re.compile('|'.join(list_of_sensory_verbs), re.IGNORECASE).search(text):
        return True
    else:
        return False


# Getting ordinal adjectives in tweet text
def ordinal_adjectives(text: str):
    list_of_ordinal_adjectives = ['во первых', 'во-первых', 'во вторых', 'во-вторых', 'в третьих', 'в-третьих',
                                  'это раз', 'это два', 'это один', 'это два', 'для начала', ]
    if re.compile('|'.join(list_of_ordinal_adjectives), re.IGNORECASE).search(text):
        return True
    else:
        return False


# Getting relatives time words in tweet text
def relative_time(text: str):
    list_of_relative_time_words = ['сегодня', 'вчера', 'позавчера', 'утром', 'вечером', 'днем', 'полдень', 'полудня',
                                   'неделе', 'месяце', 'в конце', 'в начале', 'завтра', 'ночью', 'этой', 'этим']
    if re.compile('|'.join(list_of_relative_time_words), re.IGNORECASE).search(text):
        return True
    else:
        return False


# Getting count of numbers [0-9] in tweet text
def numbers_count(text: str):
    pattern = re.compile(r"[0-9]", re.DOTALL)
    numbers_counter = len(pattern.findall(text))
    if numbers_counter:
        return numbers_counter
    else:
        return 0


# Getting quantity adjectives in tweet text
def quantity_adjective(text: str):
    list_of_quantity_adjective = ['несколько', 'некоторое количеств', 'количества', 'количество', 'определенное',
                                  'число', 'достигали', 'достигли', 'целых', 'ровно', 'минимум', 'максимум', 'порядка',
                                  'больше', 'меньше']
    if re.compile('|'.join(list_of_quantity_adjective), re.IGNORECASE).search(text):
        return True
    else:
        return False


# Getting certainty words in tweet text
def certainty_words(text: str):
    list_of_certainty_words = ['знаю', 'уверен', 'точно', 'сто процентов', '100%', 'сто проц', 'наверняка', 'считаю']
    if re.compile('|'.join(list_of_certainty_words), re.IGNORECASE).search(text):
        return True
    else:
        return False
