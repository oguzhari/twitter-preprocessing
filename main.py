import pandas as pd

from utils import *

query = '"asgari ücret" lang:tr until:2022-10-20 since:2022-09-20'
# You can get that query from the website: https://twitter.com/search-advanced


tweets = get_tweets(query, 100, readable_csv=True)


# For detect the language of tweets
tweets['detected_language'] = language_detect(tweets['content'])


# For detect the language of tweets
tweets['detected_language'] = language_detect(tweets['content'])

# For detect the language of tweets
tweets['detected_language'] = language_detect(tweets['content'])


# For remove mentions from tweets
tweets['content_rmv_mention'] = preprocessing(tweets['content'], remove_mentions=True)

# For remove links from tweets
tweets['content_rmv_link'] = preprocessing(tweets['content'], remove_links=True)

# For remove links, hashtags and make lowercase from tweets
tweets['content_rmv_link_hashtag_uppercase'] = preprocessing(tweets['content'], remove_links=True, remove_mentions=True,
                                                             lowercase=True, remove_hashtag=True)

# For remove punctuation from tweets
tweets['content_rmv_punctuation'] = preprocessing(tweets['content_rmv_link_hashtag_uppercase'], remove_punctuation=True)

# For remove rare words from tweets, you can change rare_limit value
tweets['content_rmv_rare_words'] = preprocessing(tweets['content_rmv_punctuation'], remove_rare_words=True, rare_limit=0)

# for remove short texts from tweets
tweets['content_rmv_short_text'] = preprocessing(tweets['content_rmv_rare_words'], remove_short_text=True)

# For remove stopwords from tweets
tweets['content_rmv_stopwords'] = preprocessing(tweets['content_rmv_short_text'], remove_stopwords=True)

# For extract sentiment labels and scores from tweets
tweets[['label', 'score']] = sentiment(tweets['content'])


# For extract sentiment labels and scores from tweets

print("test")

translates, error = translator(tweets['content'], 'en', secure_translations=True)

# if error return False, translates add to tweets dataframe
if not error:
    tweets['translated'] = translates

ogrenme_seti = pd.read_csv('thing/ml_test.csv', encoding='utf-8')

y = ogrenme_seti['Zafer Kutlaması']

tfidf_dataframe = create_tfidf(ogrenme_seti['content'])

get_models(tfidf_dataframe, y, classification=True)

