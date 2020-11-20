from collections import Counter
from datetime import datetime, timedelta
import time
from urllib import parse

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def _get_all_tweets(token, search_keyword, fields, duration):
    print('Fetching recent tweets ...')
    fields_ = set(['text', 'referenced_tweets', 'lang'] + fields)
    # end_time = datetime.now()
    # start_time = (end_time - timedelta(hours=24))
    url = 'https://api.twitter.com/2/tweets/search/recent'
    auth_header = {'Authorization': f'Bearer {token}'}
    search_params = {
        'query': search_keyword,
        'max_results': 100,
        'tweet.fields': ','.join(fields_),
        'start_time': '2020-11-18T22:59:59Z',
        'end_time': '2020-11-18T23:59:59Z',
    }
    response = requests.get(
        url=url,
        params=search_params,
        headers=auth_header,
    )
    print(response.headers)
    response.raise_for_status()
    # print(response.headers)
    data = response.json()
    # from pprint import pprint
    # pprint(data)
    df = pd.DataFrame.from_dict(data['data'])
    st.write(f'Found {df.shape[0]} tweets so far')

    page_count = 1
    while data.get('meta', {}).get('next_token', None) and page_count < 10:
        print(f'\tfetching page {page_count}')
        page_count += 1
        next_token = data['meta']['next_token']
        search_params['next_token'] = next_token
        response = requests.get(
            url=url,
            params=search_params,
            headers=auth_header,
        )
        print(response.headers)
        response.raise_for_status()
        data = response.json()
        df = df.append(pd.DataFrame.from_dict(data['data']), ignore_index=True)
        st.write(f'Found {df.shape[0]} tweets so far')
        time.sleep(5)
    print(f'Done getting recent tweets, found: {df.shape[0]} tweets')
    return df


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def _get_recent_tweets(token, search_keyword, fields, lang='en', include_retweets=False, duration="Last 24 hours", max_count=50):
    print('Fetching recent tweets ...')
    fields_ = set(['text', 'referenced_tweets', 'lang'] + fields)
    end_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    d = {
        'Last 4 hours': timedelta(hours=4),
        'Last 8 hours': timedelta(hours=8),
        'Last 12 hours': timedelta(hours=12),
        'Last 24 hours': timedelta(hours=24),
        'Last 48 hours': timedelta(hours=48),
        'Last 72 hours': timedelta(hours=72),
    }
    start_time = (datetime.now() - d[duration]).strftime('%Y-%m-%dT%H:%M:%SZ')
    url = 'https://api.twitter.com/2/tweets/search/recent'
    auth_header = {'Authorization': f'Bearer {token}'}
    search_params = {
        'query': search_keyword,
        'max_results': 100,
        'tweet.fields': ','.join(fields_),
        'start_time': start_time,
        'end_time': end_time,
    }
    response = requests.get(
        url=url,
        params=search_params,
        headers=auth_header,
    )
    # print(response.headers)
    response.raise_for_status()
    # print(response.headers)
    data = response.json()
    # from pprint import pprint
    # pprint(data)
    df = pd.DataFrame.from_dict(data['data'])

    if include_retweets is False:
        df = df.loc[pd.isna(df['referenced_tweets'])]
    df = df.loc[df['lang'] == lang]

    st.write(f'Found {df.shape[0]} tweets so far')

    page_count = 1
    while data.get('meta', {}).get('next_token', None) and df.shape[0] < max_count:
        print(f'\tfetching page {page_count}')
        page_count += 1
        next_token = data['meta']['next_token']
        search_params['next_token'] = next_token
        response = requests.get(
            url=url,
            params=search_params,
            headers=auth_header,
        )
        print(response.headers)
        response.raise_for_status()
        data = response.json()
        df = df.append(pd.DataFrame.from_dict(data['data']), ignore_index=True)

        if include_retweets is False:
            df = df.loc[pd.isna(df['referenced_tweets'])]
        df = df.loc[df['lang'] == lang]

        st.write(f'Found {df.shape[0]} tweets so far')
        time.sleep(5)

    print(f'After filtering for language, found: {df.shape[0]} tweets')

    if 'referenced_tweets' not in fields:
        df = df.drop('referenced_tweets', axis=1)

    if 'lang' not in fields:
        df = df.drop('lang', axis=1)

    return df


@st.cache(allow_output_mutation=True)
def _get_zero_shot_classifier():
    print('Getting zero-shot classifier')
    return pipeline('zero-shot-classification', model='joeddav/bart-large-mnli-yahoo-answers')


@st.cache()
def _get_zero_shot_classification(text, labels):
    print('Generating category predictions')
    clf = _get_zero_shot_classifier()
    return clf(text, labels, multi_class=False)


def fetch_and_analyze_tweets(
    token, search_keyword, fields, tweet_language, include_retweets, duration, max_count,
):
    if token == "":
        st.write("Please enter a Twitter API bearer token to continue.")
        st.stop()

    if search_keyword == "":
        st.write("Please enter a search keyword (example: covid)")
        st.stop()

    candidate_categories = st.text_input(
        "Comma-separated categories to use for zero-shot classification (example: school,work,travel,health,vacation",
        "",
    )

    if candidate_categories == "":
        st.write("Please enter candidate categories")
        st.stop()

    st.write(f'Fetching recent tweets related to {search_keyword}')
    df = _get_recent_tweets(token, search_keyword, fields, tweet_language, include_retweets, duration, max_count)
    
    st.write(f'Analyzing recent tweets related to {search_keyword}')
    print('Generating histogram of tweets')
    hist = Counter()
    predicted_categories = []
    prediction_confidence = []

    c1, c2 = st.beta_columns(2)

    with c1:
        st.table(df)

    with c2:
        predictions = _get_zero_shot_classification(
            df['text'].tolist(),
            candidate_categories,
        )
        print(predictions)
        for prediction in predictions:
            labels = prediction['labels']
            scores = prediction['scores']
            if scores[0] > 0.5:
                predicted_categories.append(labels[0])
                prediction_confidence.append(scores[0])
                hist.update({labels[0]: 1})
            else:
                predicted_categories.append('unknown')
                prediction_confidence.append(0)
                hist.update({'unknown': 1})

        df['predicted_category'] = predicted_categories
        df['prediction_confidence'] = prediction_confidence
        st.write(hist)
        st.bar_chart(pd.DataFrame.from_dict(hist, orient='index').reset_index())
