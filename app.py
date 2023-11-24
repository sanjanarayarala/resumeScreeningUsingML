import pandas as pd
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

def cleanResume(resumeText):
  resumeText = resumeText.lower()
  resumeText = re.sub('http\S+\s*', ' ', resumeText) # removing https
  resumeText = re.sub('#\S+', '', resumeText) # remove words starting with '#'
  resumeText = re.sub('@\S+', ' ', resumeText) # remove words starting with '@'
  resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText) # remove punctuation
  resumeText = re.sub('[^\x00-\x7f]', ' ', resumeText) # remove Non-ASCII characters
  resumeText = re.sub('\d+', '', resumeText)  # remove any numbers
  resumeText = re.sub('\s+', ' ', resumeText) # remove extra spaces
  return resumeText

def tokenize_sentence(sentence):
    word_token = nltk.word_tokenize(sentence)
    return ' '.join(word_token)

df = pd.read_csv('UpdatedResumeDataSet.csv')
df['cleaned_resume'] = df['Resume'].apply(lambda x: cleanResume(x))
df['tokenized_resume'] = df['cleaned_resume'].apply(tokenize_sentence)

lenc = LabelEncoder()
df['Category_Labelled'] = lenc.fit_transform(df['Category'])

tokens = df['tokenized_resume'].values
op_labels = df['Category_Labelled'].values

X_train, X_test, y_train, y_test = train_test_split(tokens, op_labels, test_size= 0.25, random_state=1)

word_vectorizer = TfidfVectorizer(max_features= 1600)
X_train = word_vectorizer.fit_transform(X_train)
X_test = word_vectorizer.transform(X_test)

mnb = MultinomialNB()
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
lsvm = LinearSVC()
sgd = SGDClassifier()
knn = KNeighborsClassifier()

for clf in (mnb, lsvm, lr, knn, sgd, dt, rf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)

def pdf2text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    full_text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        full_text += page.extract_text()
    return full_text

def ml_prediction(text):
    clean_text = cleanResume(text)
    tokenized_text = tokenize_sentence(clean_text)
    word_features = word_vectorizer.transform([tokenized_text])
    ypred = dt.predict(word_features)
    decoded_op = lenc.inverse_transform(ypred)
    return decoded_op[0]

def main():
    st.title('Resume Screening using Machine Learning Models')

    uploaded_file = st.file_uploader('Upload a PDF version of resume', type='pdf')
    st.write('\n')
    st.write('\n')

    if uploaded_file is not None:
        text = pdf2text(uploaded_file) 
        predictions = ml_prediction(text) 

        if predictions is not None:

            st.subheader('This resume belongs to job category - ')
            st.write(predictions)

        else:
            st.subheader('Could not predict the job category of resume')

if __name__ == '__main__':
    main()


