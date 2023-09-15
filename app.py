import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import string

mnb = pickle.load(open('mnb.pkl', 'rb'))
tfidf = pickle.load(open('vectors.pkl', 'rb'),encoding='utf-8')

st.title('E-mail Spam Detection')

st.write("Example for spam sms is : *SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info.")
st.write("you may copy this sms for use purpose")
# text input
sms = st.text_area("Enter your E-mail here.")

ps = PorterStemmer()
# convert text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)



if st.button('CHECK'):
    text = transform_text(sms)
    # vectorize the text
    X = tfidf.transform([text])
    # predict
    result = mnb.predict(X)[0]
    if result==0:
        st.success("The E-mail is not a Spam one")
    else:
        st.warning("The E-mail is a Spam")