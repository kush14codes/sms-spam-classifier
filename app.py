
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# ---------------- NLTK SETUP ----------------
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

ps = PorterStemmer()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    color: #4F46E5;
}

.sub-title {
    text-align: center;
    color: #6B7280;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
    font-size: 14px;
}

.stButton > button {
    width: 100%;
    height: 3.2em;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- PREPROCESSING ----------------
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

# ---------------- LOAD FILES ----------------
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("📊 Project Overview")

    st.write("""
    This application uses Machine Learning and
    Natural Language Processing (NLP) techniques
    to classify Email/SMS messages as Spam or Not Spam.
    """)

    st.divider()

    st.subheader("🛠 Technologies")

    st.markdown("""
    - Python
    - Scikit-Learn
    - TF-IDF Vectorizer
    - Multinomial Naive Bayes
    - NLTK
    - Streamlit
    """)

    st.divider()

    st.subheader("📌 Example Spam")

    st.code(
        "Congratulations! You won ₹50,000. "
        "Claim your reward now."
    )

    st.subheader("📌 Example Ham")

    st.code(
        "Hey, are we still meeting tomorrow at 5 PM?"
    )

# ---------------- HEADER ----------------
st.markdown(
    "<div class='main-title'>📧 Email / SMS Spam Classifier</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Machine Learning Powered Spam Detection System</div>",
    unsafe_allow_html=True
)

# ---------------- METRICS ----------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Model", "Multinomial NB")

with col2:
    st.metric("Features", "TF-IDF")

st.markdown("---")

# ---------------- INPUT ----------------
input_sms = st.text_area(
    "Enter Message",
    height=180,
    placeholder="Type or paste your SMS / Email message here..."
)

# ---------------- BUTTON ----------------
if st.button("🔍 Analyze Message"):

    if len(input_sms.strip()) == 0:
        st.warning("Please enter a message first.")
    else:

        transformed_sms = transform_text(input_sms)

        vector_input = tfidf.transform([transformed_sms])

        result = model.predict(vector_input)[0]

        # Probability
        confidence = None

        try:
            probs = model.predict_proba(vector_input)[0]
            confidence = max(probs) * 100
        except:
            pass

        st.markdown("---")

        if result == 1:

            st.error("🚨 SPAM MESSAGE DETECTED")

            st.write(
                "This message appears suspicious and "
                "contains spam-like characteristics."
            )

            if confidence:
                st.progress(int(confidence))
                st.metric(
                    "Spam Detection Confidence",
                    f"{confidence:.2f}%"
                )

        else:

            st.success("✅ LEGITIMATE MESSAGE")

            st.write(
                "This message appears safe and does "
                "not contain spam characteristics."
            )

            if confidence:
                st.progress(int(confidence))
                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

# ---------------- FOOTER ----------------
st.markdown(
    "<div class='footer'>Developed by Kushagra Chaubey</div>",
    unsafe_allow_html=True
)
