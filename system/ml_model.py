import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import ADASYN
import pickle


# Load the csv files
data1 = pd.read_csv('dataset_hatespeech.csv')
data2 = pd.read_csv('dataset_sexism.csv')


# Data Preprocessing
# Dropping rows that has null values (missing value)
data2.dropna(inplace=True)


# Dataset 1 (hatespeech)
# Rename Column
data1.rename(columns={'class': 'Sentiment', 'tweet': 'Tweet'}, inplace=True)

# Selecting the important collumn
data1 = data1[['Tweet', 'Sentiment']]

# Replacing values to True or False
data1['Sentiment'] = data1['Sentiment'].replace(0, 1).replace(2, 0)


# Dataset 2 (sexism)
# Rename columns
data2.rename(columns={'oh_label': 'Sentiment'}, inplace=True)
data2.rename(columns={'Text': 'Tweet'}, inplace=True)

# Selecting the important column
data2 = data2[['Tweet', 'Sentiment']]

# Change the data type
data2['Sentiment'] = data2['Sentiment'].astype(int)


# Combining the 2 datasets
df = pd.concat([data1, data2], ignore_index=True)


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


# Create a new column for the cleaned text
prep_text = df.apply(lambda x: preprocessing(x['Tweet']), axis=1)

combinedText = []
def listToString(list):
    for i in list:
        i = ' '.join(map(str, i))
        combinedText.append(i)

listToString(prep_text.values)
df['CleanedTweet'] = combinedText
print(df.head)


# Select independent and target variable
x_class = df['CleanedTweet'].values
y_class = df['Sentiment'].values


# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(x_class, y_class, test_size=0.2, random_state=42)


# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)


# Solve class imbalance
# Oversampling using ADASYN
ada = ADASYN(random_state=42)
X_train_ada, y_train_ada = ada.fit_resample(X_train, y_train)


# Instantiate the Selected Machine Learning Model (Linear Support Vector Machines)
svm = LinearSVC(random_state=42)
svm.fit(X_train_ada, y_train_ada)
y_predict_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_predict_svm)


# Results
print("\n----REPORT----")
print("\nMachine Learning Model (SVM) Accuracy: ", accuracy_svm)
print("\nClassification Report:\n\n", metrics.classification_report(y_test, y_predict_svm))


# Create a pickle file
with open('svm_model.pkl', 'wb') as pkl:
    pickle.dump(svm,pkl)

with open('svm_vector.pkl', 'wb') as vec:
    pickle.dump(tfidf,vec)





