# ----------------------------------------------------------------------
# EchoMind: AI Emotion & Wellness Journal (Chat-Enabled Version)
# ----------------------------------------------------------------------
#
# To run this app:
# 1. Save this code as 'app.py'
# 2. Install required libraries:
#    pip install streamlit pandas plotly transformers torch vaderSentiment
# 3. Run in your terminal:
#    streamlit run app.py
#
# ----------------------------------------------------------------------

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# ----------------------------------------------------------------------
# 1. SETUP & INITIALIZATION
# ----------------------------------------------------------------------

# Load AI models only once
@st.cache_resource
def load_models():
    """Load the NLP models for emotion and sentiment."""
    emotion_classifier = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base", 
        return_all_scores=False
    )
    vader_analyzer = SentimentIntensityAnalyzer()
    return emotion_classifier, vader_analyzer

emotion_model, vader_model = load_models()


# ----------------------------------------------------------------------
# 2. DATABASE MODULE (SQLite)
# ----------------------------------------------------------------------

DB_NAME = "echomind.db"

def init_db():
    """Initialize the SQLite database and create the journal table."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            entry_text TEXT NOT NULL,
            detected_emotion TEXT,
            stress_score REAL,
            ai_suggestion TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_entry(entry_text, emotion, stress_score, suggestion):
    """Add a new journal entry to the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO journal (entry_text, detected_emotion, stress_score, ai_suggestion) VALUES (?, ?, ?, ?)",
        (entry_text, emotion, stress_score, suggestion)
    )
    conn.commit()
    conn.close()

def get_all_entries():
    """Retrieve all journal entries as a Pandas DataFrame."""
    conn = sqlite3.connect(DB_NAME)
    # Ensure we get the correct data for the charts
    df = pd.read_sql_query("SELECT timestamp, detected_emotion, stress_score FROM journal ORDER BY timestamp ASC", conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Initialize the database on first run
init_db()


# ----------------------------------------------------------------------
# 3. AI / NLP MODULE (Emotion & Stress)
# ----------------------------------------------------------------------

def analyze_emotion(text):
    """Detect emotion using Hugging Face Transformers."""
    try:
        result = emotion_model(text)
        return result[0]['label'].capitalize()
    except Exception as e:
        return "Unknown"

def analyze_stress(text):
    """Analyze stress using VADER's negative score."""
    scores = vader_model.polarity_scores(text)
    neg_score = scores['neg']
    stress_level = round(neg_score * 10, 1) # Scale negative score (0.0 to 1.0) to 0-10
    return stress_level, scores['compound']


# ----------------------------------------------------------------------
# 4. AI SUGGESTIONS MODULE
# ----------------------------------------------------------------------

def get_ai_suggestion(emotion, stress_level):
    """Generate a simple motivational or calming suggestion."""
    if stress_level > 7:
        return "I'm sensing a high level of stress in your words. Please take a moment for yourself. A simple 5-minute deep breathing exercise can make a big difference."
    if emotion == "Sadness":
        return "It's completely okay to feel sad. Thank you for sharing that. Be extra kind to yourself today. Maybe listen to some music that comforts you?"
    if emotion == "Anger":
        return "Feeling angry is a powerful, normal emotion. It's a signal. If you feel restless, a quick walk or just tensing and relaxing your muscles can help."
    if emotion == "Anxiety" or emotion == "Fear":
        return "I understand that anxiety is an overwhelming feeling. Let's try to ground ourselves. Can you name 3 things you can see and 2 things you can hear right now?"
    if emotion == "Joy":
        return "That's wonderful to hear! I'm so glad you're feeling joyful. Take a moment to really soak in that feeling. What a great moment."
    if emotion == "Calm" or emotion == "Surprise":
        return "It sounds like a moment of calm (or surprise!). These are great times for quiet reflection or just enjoying the peace."
    
    return "Thank you for sharing that with me. Acknowledging your feelings is a huge and positive step."


# ----------------------------------------------------------------------
# 5. MAIN APP UI (Chat-Enabled)
# ----------------------------------------------------------------------

st.set_page_config(page_title="EchoMind AI Journal", layout="centered", initial_sidebar_state="expanded")
st.title("🧠 EchoMind: Your Emotion-Aware AI Journal")
st.write("Welcome. Chat about your day, your thoughts, or your feelings. I'll listen, analyze, and help you reflect.")

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How are you feeling today? You can write as much or as little as you like."}
    ]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Processing ---
if prompt := st.chat_input("Write about your day..."):
    # 1. Add user's message to chat and session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run AI/NLP Analysis on the user's prompt
    detected_emotion = analyze_emotion(prompt)
    stress_level, _ = analyze_stress(prompt)
    
    # 3. Get AI Suggestion
    ai_suggestion = get_ai_suggestion(detected_emotion, stress_level)
    
    # 4. Save the user's entry (and the AI's analysis) to the Database
    add_entry(prompt, detected_emotion, stress_level, ai_suggestion)

    # 5. Format the AI's response
    # This is the "Emotion-Aware Chat Journal" feature
    ai_response = f"""
    Thanks for sharing. I've processed your entry:

    * **Detected Emotion:** {detected_emotion}
    * **Stress Score:** {stress_level}/10

    **My Reflection:** {ai_suggestion}
    """

    # 6. Add AI's response to chat and session state
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            time.sleep(1) # Small delay to feel more "real"
            st.markdown(ai_response)
    
    # 7. Rerun to update the dashboard in the sidebar immediately
    st.rerun()


# ----------------------------------------------------------------------
# 6. VISUALIZATION DASHBOARD (in Sidebar)
# ----------------------------------------------------------------------
st.sidebar.header("Your Wellness Dashboard")
st.sidebar.write("Your emotional trends and stress levels, updated in real-time.")

all_entries_df = get_all_entries()

if all_entries_df.empty:
    st.sidebar.info("Your dashboard will appear here once you make your first entry.")
else:
    # --- Chart 1: Stress Over Time (Line Chart) ---
    st.sidebar.subheader("Stress Trend")
    stress_chart = px.line(
        all_entries_df, 
        x='timestamp', 
        y='stress_score', 
        title="Stress Score Over Time",
        markers=True
    )
    stress_chart.update_yaxes(range=[0, 10]) # Set Y-axis from 0 to 10
    st.sidebar.plotly_chart(stress_chart, use_container_width=True)

    # --- Chart 2: Emotion Distribution (Pie Chart) ---
    st.sidebar.subheader("Emotion Distribution")
    emotion_counts = all_entries_df['detected_emotion'].value_counts().reset_index()
    emotion_counts.columns = ['emotion', 'count']
    
    pie_chart = px.pie(
        emotion_counts, 
        names='emotion', 
        values='count', 
        title="Overall Emotion Distribution"
    )
    st.sidebar.plotly_chart(pie_chart, use_container_width=True)

    # --- (Optional) Display All Entries ---
    with st.sidebar.expander("View All Journal Entries"):
        st.dataframe(
            all_entries_df.sort_values(by="timestamp", ascending=False), 
            use_container_width=True
        )