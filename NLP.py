# Libraries
import numpy as np 
import pandas as pd 
# Read data
data = pd.read_csv('gender_classifier.csv',encoding = 'latin1')
data = pd.concat([data.gender, data.description], axis=1)
data.dropna(axis = 0,inplace = True)
# Convert genders to male and female (instead of brand)
data.gender = [1 if gender == "female" else 0 for gender in data.gender]

# NLP
import nltk
import re
description_list = [] # we created a list so we after these steps, we will append into this list
for description in data.description:
    # Sub, change non-letter into space
    description = re.sub("[^a-zA-Z]", " ", description)
    # All letters must be lowercase. Because e is not equall to E
    description = description.lower()
    description = nltk.word_tokenize(description)
    # We have to turn the sentence into a word list
    lemma = nltk.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    # We found the roots of each words with lemma
    description = " ".join(description)
    # After all these steps,we joined the words together.
    description_list.append(description)
    
from sklearn.feature_extraction.text import CountVectorizer

max_features = 5000

# Removes unnecessary words like the,an,of,etc.
count_vectorizer = CountVectorizer(max_features = max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

#print("Top Used {} Words: {}".format(max_features,count_vectorizer.get_feature_names()))

# Slicing
y = data.iloc[:,0].values
x = sparce_matrix

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
# Accuracy
y_pred = rf.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred)*100)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter = 2000)
lr.fit(X_train,y_train)
# Accuracy
y_pred = lr.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred)*100)