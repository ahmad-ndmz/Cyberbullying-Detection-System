from flask import Blueprint, render_template, request, flash
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from system import twitter_api
import pickle
import re

views = Blueprint('views', __name__)


# Home Page
@views.route('/')
def home():
    return render_template("home.html")


# Machine Learning
# Load the pickle model
with open("system\svm_model.pkl", "rb") as pkl:
    model = pickle.load(pkl)
with open("system\svm_vector.pkl", "rb") as vec:
    vectorizer = pickle.load(vec)


# Data Preprocessing
def preprocessing(text):
    stopW = stopwords.words('english')

    # Remove negation words in the stopword list
    stopWRemove = []
    for i in stopW:
        x = re.findall(r'[\']t$|not?', i)
        if x:
            stopWRemove.append(i)

    for i in stopWRemove:
        if i in stopW:
            stopW.remove(i)

    lemma = WordNetLemmatizer()

    text = re.sub(r'RT[\s]', ' ', text) # remove RT keyword
    text = re.sub(r'@[a-zA-Z0-9_]+', ' ', text) # remove username
    text = re.sub(r'https?://[a-zA-Z0-9/.]+', ' ', text) # remove URL
    text = re.sub(r'[^\w\s]', ' ', text) # remove punctuation
    text = re.sub(r'\d+', ' ', text) # remove digits
    text = text.lower() # convert text to lower case
    text = text.split() # tokenize the word
    text = [lemma.lemmatize(t) for t in text] # lemmatize the word
    text = [word for word in text if word not in stopW] # remove stopwords
    return text


# Convert list to string
def listToString(s):
    string = " "
    return (string.join(s))


# Get value from list
def output_text(user_pred):
    for i in user_pred:
        return str(i)


# Pre-process text and make prediction
def prediction(text):
    input_vector = []
    prep_text = preprocessing(text)
    input_vector.append(listToString(prep_text))
    new_test = vectorizer.transform(input_vector)
    pred = model.predict(new_test)
    final_pred = output_text(pred)
    return final_pred


# User Type Page (user type their own tweet)
@views.route('/user-type', methods=['GET', 'POST'])
def userType():
    if request.method == 'POST':
        new_text = request.form.get('text')
        if len(new_text) < 3:
            flash('Tweet is too short!', category='error')
        else:
            final_pred = prediction(new_text)
            return render_template("result.html", data=final_pred, text=new_text)

    return render_template("userType.html")


# Twitter User Search Page (Based on twitter user handler)
@views.route('/twitter-user-search', methods=['GET', 'POST'])
def twitterSearch():
    if request.method == 'POST':
        username = request.form.get('text')
        if len(username) < 3:
            flash('Twitter Handle is too short!', category='error')
        else:
            username_df = twitter_api.tweetAnalyzer(username)
            if not username_df.empty:
                username_df['Sentiment'] = username_df.apply(lambda x: prediction(x['Tweet']), axis=1)
                return render_template("result.html", username_df=username_df, username=username)
            else:
                flash('Twitter Handle is not available!', category='error')

    return render_template("twitterUserSearch.html")
