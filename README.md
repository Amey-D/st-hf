## Streamlit 🎈 + Hugging Face 🤗 = Fun 🎉

This app demonstrates a live Twitter dataset explorer using HuggingFace's Zero-Shot text classification capability.

### Installing

```
git clone git@github.com:Amey-D/st-hf.git && cd st-hf
pip install -r requirements.txt
```

### Running

You will need to provide an auth token so that the app can access Twitter's Search API endpoint. You can get a token by
applying for developer account at https://developer.twitter.com/en.

Once you have the token, you can either run the app and enter it via the app UI, or preferably you can export it as an
enviroment variable before running the app.

```
export TWITTER_API_TOKEN=<your auth token>
streamlit run streamlit_app.py
```


