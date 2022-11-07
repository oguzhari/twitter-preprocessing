import pandas as pd
import re
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
from alive_progress import alive_bar


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

    import snscrape.modules.twitter as sntwitter

    tweets = []
    with alive_bar(limit+1, force_tty=True) as bar:
        for i, t in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i > limit:
                break
            else:
                tweets.append(
                    [t.id, t.url, t.media, t.date.strftime("%d/%m/%Y, %H:%M:%S"), t.retweetCount, t.likeCount,
                     t.quoteCount,
                     t.hashtags, t.content, t.lang, t.user.location, t.cashtags, t.conversationId, t.coordinates,
                     t.inReplyToTweetId, t.inReplyToUser, t.mentionedUsers, t.outlinks, t.place,
                     t.quotedTweet, t.renderedContent, t.replyCount, t.retweetCount, t.retweetedTweet, t.source,
                     t.sourceLabel, t.sourceUrl, t.tcooutlinks, t.user, t.user.username,
                     t.user.created.strftime("%d-%m-%Y %H:%M:%S"), t.user.description, t.user.descriptionUrls,
                     t.user.displayname, t.user.favouritesCount, t.user.followersCount, t.user.friendsCount, t.user.id,
                     t.user.label, t.user.linkTcourl, t.user.linkUrl, t.user.listedCount, t.user.location,
                     t.user.mediaCount, t.user.profileBannerUrl, t.user.profileImageUrl, t.user.protected,
                     t.user.rawDescription, t.user.statusesCount, t.user.url, t.user.username, t.user.verified])
            bar()

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

    if also_csv:
        dataframe.to_csv(csv_name, index=False)
        print("CSV file is created")

    print(f"Dataframe has {dataframe.shape[0]-1} tweets")
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
        print("Removing hashtags...")
        start = timer()
        series = series.str.replace(r'((#)[^\s]*)\b', '', regex=True)
        print(f"Removed in {timedelta(seconds=timer() - start)}")

    if remove_mentions:
        print("Removing mentions...")
        start = timer()
        series = series.str.replace(r'((@)[^\s]*)\b', '', regex=True)
        print(f"Removed in {timedelta(seconds=timer() - start)}")

    if remove_links:
        print("Removing links...")
        start = timer()
        series = series.str.replace(r'\n', '', regex=True)
        series = series.apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        print(f"Removed in {timedelta(seconds=timer() - start)}")

    if remove_numbers:
        print("Removing numbers...")
        start = timer()
        series = series.str.replace(r'\d+', '', regex=True)
        print(f"Removed in {timedelta(seconds=timer() - start)}")

    if remove_short_text:
        print("Removing short texts...")
        start = timer()
        series = series.apply(lambda x: re.sub(r'\b\w{1,2}\b', '', x))
        print(f"Removed in {timedelta(seconds=timer() - start)}")

    if remove_stopwords:
        print("Removing stopwords...")
        start = timer()
        with open('assets/stopwords.txt', 'r') as f:
            stop = [line.strip() for line in f]
        series = series.apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
        print(f"Stopwords are removed in {timedelta(seconds=timer() - start)}")

    if lowercase:
        print("Lowercasing...")
        start = timer()
        series = series.str.lower()
        print(f"Lowercasing is done in {timedelta(seconds=timer() - start)}")

    if remove_punctuation:
        print("Removing punctuation...")
        start = timer()
        series = series.str.replace(r"((')[^\s]*)\b", '', regex=True)
        series = series.str.replace(r'[^\w\s]', '', regex=True)
        print(f"Removed in {timedelta(seconds=timer() - start)}")

    if remove_rare_words:
        print("Removing rare words...")
        start = timer()
        whole_count = pd.Series(" ".join(series).split()).value_counts()
        print(f"There are {whole_count.count()} words in the series")
        print(f"%{round(whole_count[whole_count <= rare_limit].count() / whole_count.count() * 100, 2)} of "
              f"words appear less than {rare_limit} times")
        print(f"Average word count: {np.mean(whole_count)}")
        to_remove = whole_count[whole_count <= rare_limit]
        print(f"Removing rare words...")
        series = series.apply(lambda x: " ".join(x for x in x.split() if x not in to_remove))
        print(f"{len(to_remove)} rare words removed")
        print(f"Removed in {timedelta(seconds=timer() - start)}")

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

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

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

    from mtranslate import translate

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

    from langdetect import detect

    detected = [detect(k) for k in series]
    print(pd.Series(detected).value_counts(normalize=True))
    return detected


def get_models(x, y, test_size=0.25, random_state=10, classification=False, average='binary', order_type='acc'):
    """
    Tasks
    -----
        This functions returns scores of baseline models for classification and regression problems.
    Parameters
    ----------
    x: pandas.DataFrame
        The features of the dataset.
    y: pandas.Series
        The target of the dataset.
    test_size: float (default=0.2)
        The size of the test set.
    random_state: int (default=42)
        The random state of the train test split.
    classification
        If True, the function will work on classification and returns their score.
    average: str (default='binary')
        The average method of the classification report.
    order_type: str (default='acc')
        The order type of the scores. If 'acc', the scores will be ordered by accuracy. If 'f1', the scores will be
        ordered by f1 score. If 'precision', the scores will be ordered by precision score. If 'recall', the scores will
        be ordered by recall score. If 'time' scores will be ordered by time.

    Returns
    -------
    print: str
        The scores of the baseline models.
    """

    from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    # Tum Base Modeller (Classification)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC

    # Tum Base Modeller (Regression)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    all_models = []
    order_types = {'acc': 1, 'precision': 2, 'recall': 3, 'f1': 4, 'time': 5}
    if classification:
        models = [('LR', LogisticRegression(random_state=random_state)),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('SVM', SVC(gamma='auto', random_state=random_state)),
                  ('XGB', GradientBoostingClassifier(random_state=random_state)),
                  ("LightGBM", LGBMClassifier(random_state=random_state)),
                  ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

        for name, model in models:
            start = timer()
            print(f"{name} is training")
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            acc_test = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average=average)
            recall = recall_score(y_test, y_pred, average=average)
            f1 = f1_score(y_test, y_pred, average=average)
            values = dict(name=name, acc_test=acc_test, precision=precision, recall=recall, f1=f1,
                          train_time=str(timedelta(seconds=(timer() - start)))[-15:])
            print(f"{name} is done in {timedelta(seconds=(timer() - start))}")
            all_models.append(values)
        sort_method = False

    else:
        models = [('LR', LinearRegression()),
                  ("Ridge", Ridge()),
                  ("Lasso", Lasso()),
                  ("ElasticNet", ElasticNet()),
                  ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor()),
                  ('RF', RandomForestRegressor()),
                  ('SVR', SVR()),
                  ('GBM', GradientBoostingRegressor()),
                  ("XGBoost", XGBRegressor()),
                  ("LightGBM", LGBMRegressor()),
                  ("CatBoost", CatBoostRegressor(verbose=False))]

        for name, model in models:
            print(f"{name} is training")
            model.fit(x_train, y_train)
            y_pred_test = model.predict(x_test)
            y_pred_train = model.predict(x_train)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
            all_models.append(values)
        sort_method = True

    all_models_df = pd.DataFrame(all_models)
    if order_type == 'time':
        sort_method = True
    all_models_df = all_models_df.sort_values(all_models_df.columns[order_types[order_type]], ascending=sort_method)
    print(f"\nAll models are done --- ordered by {order_type}")
    print(all_models_df.to_markdown())
    return None


def create_tfidf(series):
    """
    Tasks
    -----
        Creates tfidf matrix for the given series.

    Parameters
    ----------
    series: pandas.Series
        The series to be transformed.

    Returns
    -------
    df_tfidf_vect: pandas.DataFrame
        The tfidf matrix.
    """

    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorized = TfidfVectorizer(analyzer='word')
    tfidf_wm = tfidf_vectorized.fit_transform(series)
    tfidf_tokens = tfidf_vectorized.get_feature_names_out()
    df_tfidf_vect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens)

    return df_tfidf_vect


def describe_series(series):
    """
    Tasks
    -----
        Describes the given series.

    Parameters
    ----------
    series: pandas.Series
        The series to be described.

    Returns
    -------
    print: str
        The description of the given series.
    """
    most_repeated_word = pd.Series(" ".join(series).split()).value_counts().index[1]
    most_repeated_count = pd.Series(" ".join(series).split()).value_counts()[1]

    print(f"""
    Missing Values: {series.isnull().sum()}
    Series has {len(series)} rows.
    Series has {pd.Series(" ".join(series).split()).value_counts().count()} unique words.
    Most repeated word is "{most_repeated_word}".
    "{most_repeated_word}" is repeated {most_repeated_count} times.
    """)
    return None

