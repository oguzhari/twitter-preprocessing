from utils import *

query = '"asgari ücret" lang:tr until:2022-10-20 since:2022-09-20'
# You can get that query from the website: https://twitter.com/search-advanced

tweets = get_tweets(query, 100)
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======

# For detect the language of tweets
tweets['detected_language'] = language_detect(tweets['content'])
>>>>>>> Stashed changes
=======

# For detect the language of tweets
tweets['detected_language'] = language_detect(tweets['content'])
>>>>>>> Stashed changes

# For remove mentions from tweets
tweets['content_rmv_mention'] = preprocessing(tweets['content'], remove_mentions=True)

# For remove links from tweets
tweets['content_rmv_link'] = preprocessing(tweets['content'], remove_links=True)

# For remove links, hashtags and make lowercase from tweets
tweets['content_rmv_link_hashtag_uppercase'] = preprocessing(tweets['content'], remove_links=True,
                                                             lowercase=True, remove_hashtag=True)

# For remove punctuation from tweets
tweets['content_rmv_punctuation'] = preprocessing(tweets['test_content3'], remove_punctuation=True)

# For remove rare words from tweets, you can change rare_limit value
tweets['content_rmv_rare_words'] = preprocessing(tweets['content'], remove_rare_words=True)

# For remove stopwords from tweets
tweets['content_rmv_stopwords'] = preprocessing(tweets['content'], remove_stopwords=True)

# For extract sentiment labels and scores from tweets
tweets[['label', 'score']] = sentiment(tweets['content'])

translates, error = translator(tweets['content'], 'en', secure_translations=True)

if not error:
    tweets['translated'] = translates

