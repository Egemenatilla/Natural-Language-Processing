import numpy as np
import pandas as pd

# Read data from csv file
data = pd.read_csv('Restaurant_Reviews.csv',error_bad_lines = False)
# Preprocessing
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputed_values = data.iloc[:, -1:].values

imputer = imputer.fit(imputed_values[:, -1:])

imputed_values[:, -1:] = imputer.transform(imputed_values[:, -1:])

liked = pd.DataFrame(data=imputed_values, index= range(716),columns = ['Liked'])

print(liked)

review1 = data.iloc[:,0:1].values

review = pd.DataFrame(data = review1, index= range(716),columns = ['Review'])

reviews = pd.concat([review, liked],axis=1)
#%% Import libraries

# Regular Expression -Sparce Matrix and Punctuation Marks
import re

# Stopwords
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords

# Porter Stemmer
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#%% NLTK Section - Preprocessing
corpus = []

for i in range(716):     
    # Regular Expression -Sparce Matrix and Punctuation Marks
    review = re.sub('[^a-zA-Z]',' ',reviews['Review'][i])
    # Uppercase - Lowercase
    review = review.lower()
    review = review.split()
    # Stopwords and Porter Stemmer
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Make string to list 
    review = ' '.join(review)
    corpus.append(review)

#%% Feature Extraction - Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = (1000))

X = cv.fit_transform(corpus).toarray()
y = reviews.iloc[:,1].values

#%% Machine Learning -Classification
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)