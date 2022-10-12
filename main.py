from utils import *


tweets_list = []
query = "(zelenski OR zelensky OR zelenskiy) lang:tr until:2022-02-12 since:2022-01-12"
# You can get that query from the website: https://twitter.com/search-advanced

tweets = get_tweets(query, 1000)

# For remove mentions from tweets
tweets['test_content'] = preprocessing(tweets['content'], remove_mentions=True)

# For remove links from tweets
tweets['test_content2'] = preprocessing(tweets['content'], remove_links=True)

# For remove links, hashtags and make lowercase from tweets
tweets['test_content3'] = preprocessing(tweets['content'], remove_links=True, lowercase=True, remove_hashtag=True)

# For remove punctuation from tweets
tweets['test_content4'] = preprocessing(tweets['test_content3'], remove_punctuation=True)

# For remove rare words from tweets, you can change rare_limit value
tweets['test_content5'] = preprocessing(tweets['content'], remove_rare_words=True)

# For remove stopwords from tweets
tweets['test_content6'] = preprocessing(tweets['content'], remove_stopwords=True)

# For extract sentiment labels and scores from tweets
tweets[['label', 'score']] = sentiment(tweets['content'])

tweets['translated'] = translator(tweets['content'], 'en', secure_translations=True)

