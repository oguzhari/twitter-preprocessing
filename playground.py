from utils import *

query = '"asgari ücret" lang:tr until:2022-10-20 since:2022-09-20'
# You can get that query from the website: https://twitter.com/search-advanced

tweets = get_tweets(query, 100)

tweets['content_rm_link_hashtag_mention'] = preprocessing(tweets['content'], remove_links=True, remove_punctuation=True,
                                                          remove_hashtag=True, remove_mentions=True, lowercase=True)

tweets['content_rm_link_hashtag_mention']


