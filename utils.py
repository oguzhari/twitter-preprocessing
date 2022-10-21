import pandas as pd
import snscrape.modules.twitter as sntwitter
import re
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from mtranslate import translate
from timeit import default_timer as timer
from datetime import timedelta
from langdetect import detect


def get_tweets(query, limit=1000000000, also_csv=False, csv_name='tweets.csv'):
    """
    Tasks
    -----
        Gets tweets from Twitter.

    Parameters
    ----------
    query: str
        The query to be searched on Twitter.
    limit: int (default=1000000000)
        The limit of tweets to be searched.
    also_csv: bool (default=False)
        If True, saves the tweets as a csv file.
    csv_name: str (default='tweets.csv')
        The name of the csv file to be saved.
    Returns
    -------
    dataframe: pandas.DataFrame
        The dataframe containing the tweets.
    """
    tweets = []
    for i, t in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i > limit:
            break
        else:
            tweets.append(
                [t.id, t.url, t.media, t.date.strftime("%d/%m/%Y, %H:%M:%S"), t.retweetCount, t.likeCount, t.quoteCount,
                 t.hashtags, t.content, t.lang, t.user.location, t.cashtags, t.conversationId, t.coordinates,
                 t.inReplyToTweetId, t.inReplyToUser, t.mentionedUsers, t.outlinks, t.place,
                 t.quotedTweet, t.renderedContent, t.replyCount, t.retweetCount, t.retweetedTweet, t.source,
                 t.sourceLabel, t.sourceUrl, t.tcooutlinks, t.user, t.user.username,
                 t.user.created.strftime("%d-%m-%Y %H:%M:%S"), t.user.description, t.user.descriptionUrls,
                 t.user.displayname, t.user.favouritesCount, t.user.followersCount, t.user.friendsCount, t.user.id,
                 t.user.label, t.user.linkTcourl, t.user.linkUrl, t.user.listedCount, t.user.location,
                 t.user.mediaCount, t.user.profileBannerUrl, t.user.profileImageUrl, t.user.protected,
                 t.user.rawDescription, t.user.statusesCount, t.user.url, t.user.username, t.user.verified])

        if i % 100 == 0 and i != 0:
            print(f"Downloaded {i} tweets")

    dataframe = pd.DataFrame(tweets, columns=['id', 'url', 'media', 'date', 'retweet_count', 'like_count', 'quoteCount',
                                              'hashtags', 'content', 'lang', 'user_location',
                                              'cashtags', 'conversation_id', 'coordinates', 'inReplyToTweetId',
                                              'inReplyToUser', 'mentionedUsers', 'out_links', 'place',
                                              'quotedTweet', 'renderedContent', 'replyCount', 'retweetCount',
                                              'retweetedTweet', 'source', 'sourceLabel', 'sourceUrl', 'tco_out_links',
                                              'user', 'user_name', 'user_created', 'user_description',
                                              'user_descriptionUrls', 'user_display_name', 'user_favouritesCount',
                                              'user_followersCount', 'user_friendsCount', 'user_id', 'user_label',
                                              'user_link_Tco_url', 'user_linkUrl', 'user_listedCount', 'user_location',
                                              'user_media_count', 'user_profile_banner_url', 'user_profile_image_url',
                                              'user_protected', 'user_raw_description', 'user_statuses_count',
                                              'user_url', 'user_username', 'user_verified'])

<<<<<<< Updated upstream
    if also_csv:
        dataframe.to_csv(csv_name, index=False)
        print("CSV file is created")
    print(f"Dataframe has {dataframe.shape[0] - 1} tweets")
=======
    print(f"Dataframe has {dataframe.shape[0]-1} tweets")

>>>>>>> Stashed changes
    return dataframe


def preprocessing(series, remove_hashtag=False, remove_mentions=False, remove_links=False, remove_numbers=False,
                  remove_short_text=False, remove_stopwords=False, lowercase=False, remove_punctuation=False,
                  remove_rare_words=False, rare_limit=5):
    """
    Tasks
    -----
        Preprocesses the given series.

    Parameters
    ----------
    series: pandas.Series
        The series to be preprocessed.
    remove_hashtag: bool
        If True, removes hashtags (#) from the series.
    remove_mentions: bool
        If True, removes mentions (@) from the series.
    remove_links: bool
        If True, removes links from the series.
    remove_numbers: bool
        If True, removes numbers from the series.
    remove_short_text: bool
        If True, removes short texts (shorter than the given value) from the series.
    remove_stopwords: bool
        If True, removes stopwords (gets stopwords from stopwords.txt) from the series.
    lowercase: bool
        If True, converts all characters to lowercase.
    remove_punctuation:
        If True, removes punctuation from the series.
    remove_rare_words:
        If True, removes rare words (occurs less than the given value) from the series.
    rare_limit: int (default=5)
        The limit for removing rare words.

    Returns
    -------
    series: pandas.Series
        The preprocessed series.
    """
    if remove_hashtag:
        series = series.str.replace(r'((#)[^\s]*)\b', '', regex=True)

    if remove_mentions:
        series = series.str.replace(r'((@)[^\s]*)\b', '', regex=True)

    if remove_links:
        series = series.str.replace(r'\n', '', regex=True)
        series = series.apply(lambda x: re.split('https:\/\/.*', str(x))[0])

    if remove_numbers:
        series = series.str.replace(r'\d+', '', regex=True)

    if remove_short_text:
        series = series.apply(lambda x: re.sub(r'\b\w{1,2}\b', '', x))

    if remove_stopwords:
        with open('assets/stopwords.txt', 'r') as f:
            stop = [line.strip() for line in f]
        series = series.apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

    if lowercase:
        series = series.str.lower()

    if remove_punctuation:
        series = series.str.replace(r"((')[^\s]*)\b", '', regex=True)
        series = series.str.replace(r'[^\w\s]', '', regex=True)

    if remove_rare_words:
        all_ = [x for y in series for x in y.split(' ')]
        a, b = np.unique(all_, return_counts=True)
        print(f"Average word count: {np.mean(b)}")
        to_remove = a[b <= rare_limit]
        series = [' '.join(np.array(y.split(' '))[~np.isin(y.split(' '), to_remove)]) for y in series]
        print(f"Removed {len(to_remove)} rare words")

    return series


def sentiment(series, model_name="savasy/bert-base-turkish-sentiment-cased"):
    """
    Tasks
    -----
        Calculates the sentiment of the given series.

    Parameters
    ----------
    series: pandas.Series
        The series to be preprocessed.
    model_name: str (default="savasy/bert-base-turkish-sentiment-cased")
        The name of the model to be used for sentiment analysis.

    Returns
    -------
    label: list
        The sentiment label of the given series.
    score: list
        The sentiment score of the given series.
    """
    label, score = [], []
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sa = pipeline("sentiment-analysis", tokenizer=tokenizer, model=model)
    for i, k in enumerate(series):
        result = sa(k)
        label.append(result[0]['label'])
        score.append(result[0]['score'])
        if i % 100 == 0 and i != 0:
            print(f"Sentiment analysis of {i} tweets is done")
    return pd.DataFrame({'label': label, 'score': score})


def translator(series, target_language="en", from_language="auto", secure_translations=False, secure_frequency=1000):
    """
    Tasks
    -----
        Translate given series to target language. The process is too slow due to the API limit.
        if you want to secure your translations, you can use secure_translations=True. It will save the translations
        in every secure_frequency. Due to the API limit, it is recommended to use secure_translations=True.
        if API give you an error and secure_translations=False you will lose all the translations.
        Enable secure_translations may slow down the process.

    Parameters
    ----------
    series: pandas.Series
        The series to be translated.
    target_language: str (default="en")
        The target language to be translated.
    from_language: str (default="auto")
        The language of the given series. If "auto", the language will be detected automatically.
    secure_translations: bool (default=False)
        If True, the translation will be done in batches.
    secure_frequency: int (default=1000)
        The frequency of the batches.

    Returns
    -------
    series: pandas.Series
        The translated series.
    """
    secure_list, translate_list = [], []
    start = timer()
    print("Translation started, It's too slow due to API limit.")
    print("100 tweets nearly takes 40 seconds")
    print("You will see the progress in the console soon.")
    for i, k in enumerate(series):
        try:
            translate_list.append(translate(k, target_language, from_language))
            if secure_translations:
                if i % secure_frequency == 0 and i != 0:
                    print(f"{i} tweets are translated and secured")
                    secure_list = translate_list.copy()
                elif i % 100 == 0 and i != 0:
                    print(f"{i} tweets are translated and {timedelta(seconds=(timer() - start))} time elapsed")
            elif i % 100 == 0 and i != 0:
                print(f"{i} tweets are translated and {timedelta(seconds=(timer() - start))} time elapsed")
        except Exception as e:
            print(e)
            return secure_list
    print(f"Translation is done in {timedelta(seconds=(timer() - start))}")
    return translate_list


def language_detect(series):
    """
    Tasks
    -----
        Detects the language of the given series.

    Parameters
    ----------
    series: pandas.Series
        The series to be detected.
    Returns
    -------
    series: pandas.Series
        The language of the given series.
    """
    detected = [detect(k) for k in series]
    print(pd.Series(detected).value_counts(normalize=True))
    return detected

