# EchoMind

# 💬 AI-Based Chatbot Using NLP and Deep Learning

## 📌 Overview
This project is an AI-powered chatbot developed using Python, Natural Language Processing (NLP), and Deep Learning techniques. The chatbot understands user inputs, identifies the intent behind messages, and generates relevant responses in real time.

The system uses custom training data from an `intents.json` file and applies NLP preprocessing techniques to improve text understanding and response accuracy.

---

## ✨ Features
- Intent-based intelligent chatbot
- Real-time response generation
- Natural Language Processing (NLP)
- Text preprocessing and cleaning
- Custom training dataset support
- Context-aware conversation handling
- Emotion-aware interaction capability
- Model saving and reuse functionality
- Scalable deployment support

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- NLTK
- NumPy
- JSON
- Pickle

---

## 🧠 Concepts Used
- **Natural Language Processing (NLP)**  
  Tokenization, lemmatization, and text preprocessing

- **Bag-of-Words (BoW)**  
  Converting text into numerical vectors

- **Deep Neural Networks**  
  Intent classification and prediction

- **Supervised Learning**  
  Training using custom labeled data

- **Conversation Memory & Emotion Detection**  
  Personalized and context-aware responses

- **Model Persistence**  
  Saving trained models using Pickle and `.h5` files

---

## 📂 Project Structure

```bash
Chatbot-Project/
│
├── intents.json
├── chatbot_model.h5
├── words.pkl
├── classes.pkl
├── train_chatbot.py
├── chatbot.py
├── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/chatbot-project.git
cd chatbot-project
```

### Install dependencies

```bash
pip install tensorflow
pip install nltk
pip install numpy
pip install scikit-learn
```

---

## ▶️ Run the Project

Train the chatbot:

```bash
python train_chatbot.py
```

Run the chatbot:

```bash
python chatbot.py
```

---

## 💡 How It Works

1. Reads user patterns and responses from `intents.json`
2. Preprocesses text using NLP techniques
3. Converts text into numerical vectors using Bag-of-Words
4. Trains a Deep Neural Network model
5. Predicts user intent from input
6. Returns the most relevant response

---

## 🚀 Future Scope

- Integration with advanced language models
- Voice-enabled interaction
- Web and mobile deployment
- Enhanced contextual understanding
- Multi-language support

---

## 👩‍💻 Author

**Varshith**
