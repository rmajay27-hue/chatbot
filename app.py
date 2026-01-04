import random
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

train_texts = [
    "hi","hello","hey","good morning","good evening",
    "bye","goodbye","see you later",
    "thanks","thank you","that was helpful",
    "what is your name","who are you","what can you do"
]

train_labels = [
    "greet","greet","greet","greet","greet",
    "bye","bye","bye",
    "thanks","thanks","thanks",
    "about_bot","about_bot","about_bot"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_texts)

clf = LogisticRegression(max_iter=1000)
clf.fit(X, train_labels)

RESPONSES = {
    "greet": [
        "Hello! How can I help you?",
        "Hi there!",
        "Hey, nice to meet you!"
    ],
    "bye": [
        "Goodbye!",
        "See you later!",
        "Bye, have a great day!"
    ],
    "thanks": [
        "You're welcome!",
        "No problem!",
        "Glad I could help!"
    ],
    "about_bot": [
        "I am a simple machine learning chatbot.",
        "I am a tiny ML demo bot built in Python.",
        "I can reply to greetings, thanks, and simple questions."
    ],
    "fallback": [
        "Sorry, I didn't understand that.",
        "Can you say that in another way?",
        "I'm not sure about that yet."
    ]
}

def predict_intent(text):
    X_test = vectorizer.transform([text])
    probs = clf.predict_proba(X_test)[0]
    intent = clf.classes_[probs.argmax()]
    if probs.max() < 0.4:
        return "fallback"
    return intent

def get_reply(intent):
    return random.choice(RESPONSES[intent])

st.title("💬 ML Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("You:")

if st.button("Send") and user_input:
    intent = predict_intent(user_input)
    reply = get_reply(intent)

    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", reply))

for sender, message in st.session_state.chat:
    st.markdown(f"**{sender}:** {message}")
