import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score


messages = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"])
# print(messages)

ps = PorterStemmer()
lemma = WordNetLemmatizer()

corpus = []

for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    # review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = [lemma.lemmatize(word) for word in review if not word in stopwords.words('english')]
    
    review = ' '.join(review)
    corpus.append(review)
    
# print(corpus)
    
# # Creating the Bag of Words model
cv = CountVectorizer(max_features=2500)

# features/independent variables
X = cv.fit_transform(corpus).toarray() 
# print(len(cv.get_feature_names_out()))


y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values
# print(y)

# # Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# # Training model using Naive bayes classifier
model_spam = MultinomialNB()
spam_detect_model = model_spam.fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)
print(y_test, y_pred )

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy_score = accuracy_score(y_test, y_pred)
print(accuracy_score)

