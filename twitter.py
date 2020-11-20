from collections import Counter
from datetime import datetime, timedelta
from urllib import parse

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline


# @st.cache(allow_output_mutation=True, ttl=60*60)
def _get_recent_tweets(token, search_keyword, fields, lang='en', include_retweets=False, max_count=100):
    print(f'Getting up to {max_count} recent tweets ')
    fields_ = set(['text', 'referenced_tweets', 'lang'] + fields)
    # end_time = datetime.now()
    # start_time = (end_time - timedelta(hours=24))
    url = 'https://api.twitter.com/2/tweets/search/recent'
    auth_header = {'Authorization': f'Bearer {token}'}
    search_params = {
        'query': search_keyword,
        'max_results': 100,
        'tweet.fields': ','.join(fields_),
        # 'start_time': '2020-11-18T00:00:00Z',
        'end_time': '2020-11-19T16:00:00Z',
    }
    response = requests.get(
        url=url,
        params=search_params,
        headers=auth_header,
    )
    response.raise_for_status()
    print(response.headers)
    data = response.json()
    from pprint import pprint
    pprint(data)
    df = pd.DataFrame.from_dict(data['data'])

    page_count = 1
    while data.get('meta', {}).get('next_token', None):
        print(f'\tfetching page {page_count}')
        page_count += 1
        next_token = data['meta']['next_token']
        search_params['next_token'] = next_token
        response = requests.get(
            url=url,
            params=search_params,
            headers=auth_header,
        )
        response.raise_for_status()
        data = response.json()
        df = df.append(pd.DataFrame.from_dict(data['data']), ignore_index=True)

    print(f'Done getting recent tweets, found: {df.shape[0]} tweets')
    if include_retweets is False:
        df = df.loc[pd.isna(df['referenced_tweets'])]
        print(f'After excluding retweets, found: {df.shape[0]}')

    df = df.loc[df['lang'] == lang]
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
    token, search_keyword, fields, tweet_language, max_count, include_retweets, zero_shot_categories,
):
    if token == "":
        st.write("Please enter a Twitter API bearer token to continue.")
        st.stop()

    candidate_categories = zero_shot_categories

    df = _get_recent_tweets(token, search_keyword, fields, tweet_language, include_retweets, max_count)
    
    print('Generating histogram of tweets')
    hist = Counter()
    predicted_categories = []
    for idx, row in df.iterrows():
        print(f'\tProcessing {idx}')
        res = _get_zero_shot_classification(row["text"], candidate_categories)
        if res['scores'][0] > 0.5:
            predicted_categories.append(res['labels'][0])
            hist.update({res['labels'][0]: 1})
        else:
            predicted_categories.append('unknown')
            hist.update({'unknown': 1})

    df['predicted_category'] = predicted_categories
    st.write(hist)
    st.bar_chart(pd.DataFrame.from_dict(hist, orient='index').reset_index())
    with st.beta_expander('Expand to see the raw tweets data'):
        st.table(df)
