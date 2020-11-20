import os
from urllib import parse

import pandas as pd
import requests
import streamlit as st
from transformers import pipeline

from twitter import fetch_and_analyze_tweets

st.title("Streamlit Twitter Feed Analyzer")

with st.sidebar.beta_expander("Twitter Dataset", expanded=True):
    token_from_env = os.environ['TWITTER_API_TOKEN']
    token = token_from_env if token_from_env else st.text_input("Enter Twitter API bearer token")
    search_keyword = st.text_input("Enter a search keyword", value="#streamlit")
    tweet_language = st.selectbox("Select language", options=["en"], index=0)
    max_count = st.slider("Max number ot tweets", min_value=10, max_value=100, value=50, step=10)
    include_retweets = st.radio("Include retweets", options=[False, True])
    fields = st.multiselect(
        "Fields to show",
        options=["source", "created_at", "lang", "referenced_tweets"])
    zero_shot_categories = st.text_input(
        "Comma-separated categories to use for zero-shot classification",
        "finance,medicine,politics,machinelearning,sports"
    )

fetch_and_analyze_tweets(
    token=token,
    search_keyword=search_keyword,
    fields=fields,
    tweet_language=tweet_language,
    max_count=max_count,
    include_retweets=include_retweets,
    zero_shot_categories=zero_shot_categories.split(','),
)

#import streamlit.components.v1 as components
#components.iframe('<blockquote class="twitter-tweet"><p lang="en" dir="ltr">just setting up my twttr</p>&mdash; jack (@jack) <a href="https://twitter.com/jack/status/20?ref_src=twsrc%5Etfw">March 21, 2006</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>')
    