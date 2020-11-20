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
    search_keyword = st.text_input("Enter a search keyword", value="")
    tweet_language = st.selectbox("Select language", options=["en"], index=0)
    include_retweets = st.radio("Include retweets", options=[False, True])
    fields = st.multiselect(
        "Fields to show",
        options=["source", "created_at", "lang", "referenced_tweets"])
    duration = st.selectbox(
        "Duration of tweets",
        options=["Last 4 hours", "Last 8 hours", "Last 24 hours", "Last 48 hours", "Last 72 hours"])
    max_count = st.slider("Max number tweets to analyze", min_value=10, max_value=500, value=50, step=10)

fetch_and_analyze_tweets(
    token=token,
    search_keyword=search_keyword,
    fields=fields,
    tweet_language=tweet_language,
    include_retweets=include_retweets,
    duration=duration,
    max_count=max_count,
)
    