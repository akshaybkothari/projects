import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

dataset = pd.read_csv('IMDB_Dataset.csv')

#FEATURE EXTRACTION
cleaned_data = []
'''for i in range(0, 20000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['review'][i]) #to remove the punctuations and html tags
    review= review.lower()
    review= review.split()  #converts sentence into words and converts it into lower case
    ps = PorterStemmer()
    review = [ps.stem(w) for w in review if not w in set(stopwords.words('english'))] #stemmazising the words
    review = ' '.join(review)
    cleaned_data.append(review)


with open("cleaned_data.pickle","wb") as f:
  pickle.dump(cleaned_data,f)'''
pickle_in = open("cleaned_data.pickle","rb")
cleaned_data = pickle.load(pickle_in)




vectorizer = CountVectorizer(max_features = 5000)#Convert a collection of text documents to a matrix of token counts
X = vectorizer.fit_transform(cleaned_data).toarray()#gives a vector for each string which corresponds to the number of times a token is repeated
y = dataset.iloc[:, 1].values
y=y[:20000]
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


model_nb = MultinomialNB()
model_nb.fit(X_train,y_train)
y_pred = model_nb.predict(X_test) #predicting the output using test data


# verifying  the model performance for new reviews.
reviews = ["I love this movie","This movie is bad","I was going to say something awesome or great or good, but I simply can't because the movie is so bad.","It might have bad actors, but everything else is good.","This movie turned out to be better than I had expected it to be. Some parts were pretty funny. It was nice to have a movie with a new plot.","First one was much better, I had enjoyed it a lot. This one has not even produced a smile. The idea was showing how deep down can human kind fall, but in reference to the characters not the film-maker."]
for review in reviews:
    tag=model_nb.predict(vectorizer.transform([review]).toarray())
    if tag==[0] :
        print(review,'=','negative')
    else :
        print(review,'=','positive')


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred)) #gives the confusion matrix op
print(classification_report(y_test,y_pred))  #gives the precision recall and other values
print('accuracy : ',accuracy_score(y_test, y_pred)) #ACCURACY
