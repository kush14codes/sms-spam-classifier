# 📧 Email / SMS Spam Classifier

A Machine Learning-powered web application that classifies Email and SMS messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) techniques and a Multinomial Naive Bayes classifier.

## 🚀 Live Demo

Add your Streamlit deployment link here after deployment:

```text
https://your-app-name.streamlit.app
```

---

## 📌 Features

* Detects Spam and Legitimate messages
* NLP-based text preprocessing
* TF-IDF Vectorization
* Multinomial Naive Bayes Classifier
* Confidence Score for Predictions
* Interactive Streamlit Web Interface
* Real-time Message Analysis

---

## 🛠 Tech Stack

* Python
* Streamlit
* Scikit-Learn
* NLTK
* Pandas
* NumPy

---

## ⚙️ Machine Learning Pipeline

### 1. Text Preprocessing

The input message undergoes:

* Lowercase conversion
* Tokenization
* Removal of special characters
* Stopword removal
* Stemming using Porter Stemmer

### 2. Feature Extraction

TF-IDF Vectorization is used to convert text into numerical feature vectors.

### 3. Model Training

A Multinomial Naive Bayes classifier is trained on the SMS Spam Collection Dataset.

### 4. Prediction

The trained model predicts whether the message is:

* Spam 🚨
* Not Spam ✅

---

## 📂 Project Structure

```text
Spam-Classifier/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── spam classifier.ipynb
```

---

## 📸 Application Preview

Add screenshots here after deployment.

### Home Page

```text
Upload Screenshot Here
```

### Prediction Result

```text
Upload Screenshot Here
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/sms-spam-classifier.git
```

Move into the project directory:

```bash
cd sms-spam-classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📊 Model Information

| Component                | Technique               |
| ------------------------ | ----------------------- |
| Text Processing          | NLP                     |
| Feature Extraction       | TF-IDF                  |
| Classification Algorithm | Multinomial Naive Bayes |
| Interface                | Streamlit               |

---

## 🎯 Future Improvements

* Deep Learning based Spam Detection
* Multi-language Support
* Email Subject Classification
* Explainable AI Predictions
* Model Monitoring Dashboard

---

## 👨‍💻 Author

**Kushagra Chaubey**

Computer Science Engineering Student | Data Science & Machine Learning Enthusiast

LinkedIn: https://www.linkedin.com/in/kushagra-chaubey-a86049291/

GitHub: https://github.com/kush14codes/
---

⭐ If you found this project useful, consider giving it a star.
