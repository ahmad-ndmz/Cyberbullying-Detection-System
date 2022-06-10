import tweepy
import pandas as pd


# Twitter API Credentials
api_key = 'XXXXXXXXXXXXX'
api_key_secret = 'XXXXXXXXXXXXX'
access_token = 'XXXXXXXXXXXXX'
access_token_secret = 'XXXXXXXXXXXXX'

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)


# Twitter API
api = tweepy.API(auth)


# Get 10 latest tweets based on Twitter username
def tweetAnalyzer(username):
    username_df = []
    try:
        tweets = api.user_timeline(screen_name=username, count=100, exclude_replies=True, include_rts=False,
                                   tweet_mode='extended')
        for tweet in tweets[:10]:
            username_df.append({'Tweet': tweet.full_text})
        return pd.DataFrame.from_dict(username_df)

    except BaseException as e:
        print('User Not Found, ERROR', str(e))
        return pd.DataFrame()


