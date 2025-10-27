# ----------------------------------------------------------------------
# EchoMind: AI Wellness Companion (Interactive & Actionable)
# ----------------------------------------------------------------------
#
# This version includes:
# - Interactive chat UI
# - "New Chat Session" button
# - AI replies that quote the user
# - Specific exercises & overcoming solutions
# - Session-based dashboard
# - Overall history dashboard
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
    df = pd.read_sql_query("SELECT timestamp, detected_emotion, stress_score FROM journal ORDER BY timestamp ASC", conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Initialize the database on first run
init_db()


# ----------------------------------------------------------------------
# 3. AI / NLP MODULE
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
# 4. AI SUGGESTIONS MODULE (UPGRADED)
# ----------------------------------------------------------------------

def get_ai_suggestion(emotion, stress_level):
    """
    Generate a more interactive reflection and a specific exercise.
    Returns a tuple: (reflection, exercise)
    """
    
    reflection = ""
    exercise = ""

    # --- High Stress (Overrides emotion) ---
    if stress_level > 8:
        reflection = "I'm sensing a very high level of stress in your words. That sounds incredibly tough, and it's completely valid to feel overwhelmed right now."
        exercise = ("**Actionable Tip: Box Breathing**\n\n"
                    "Let's try to reset your nervous system. \n"
                    "1. Inhale slowly for 4 seconds. \n"
                    "2. Hold your breath for 4 seconds. \n"
                    "3. Exhale slowly for 4 seconds. \n"
                    "4. Hold the exhale for 4 seconds. \n"
                    "Repeat this 5 times.")
    
    # --- Emotion-Specific Responses ---
    elif emotion == "Sadness":
        reflection = "Thank you for sharing that you're feeling sad. It's a heavy feeling, and I want you to know it's okay to feel this way."
        exercise = ("**Actionable Tip: Self-Kindness**\n\n"
                    "Let's try a small act of self-kindness. Can you do one small, gentle thing for yourself right now? \n\n"
                    "*Maybe make a warm cup of tea, stretch your arms for 30 seconds, or listen to one song that usually comforts you.*")
    
    elif emotion == "Anger":
        reflection = "Feeling angry is a powerful, normal emotion. It's a signal that something isn't right. Thanks for letting it out here instead of holding it in."
        exercise = ("**Actionable Tip: Progressive Muscle Relaxation**\n\n"
                    "Let's channel that physical energy. \n"
                    "1. Clench your fists as tight as you can for 5 seconds. \n"
                    "2. Release them and feel the tension leave. \n"
                    "3. Do the same with your shoulders (shrug them up to your ears). \n"
                    "Repeat this a few times to release that tension.")
    
    elif emotion == "Anxiety" or emotion == "Fear":
        reflection = f"I hear that you're feeling {emotion.lower()}. That anxious or fearful energy can be really overwhelming and hijack your thoughts."
        exercise = ("**Actionable Tip: 5-4-3-2-1 Grounding**\n\n"
                    "Let's ground ourselves in the present moment. \n"
                    "- **5:** Name 5 things you can **see**. \n"
                    "- **4:** Name 4 things you can **feel**. \n"
                    "- **3:** Name 3 things you can **hear**. \n"
                    "- **2:** Name 2 things you can **smell**. \n"
                    "- **1:** Name 1 thing you can **taste**.")
    
    elif emotion == "Joy":
        reflection = "That's wonderful to hear! Feeling joy is a fantastic experience, and I'm genuinely happy for you."
        exercise = ("**Actionable Tip: Savoring**\n\n"
                    "Let's try to hold on to this feeling. \n"
                    "Take 60 seconds to close your eyes and *really* think about what's making you happy. Why does it feel good? Try to lock that positive feeling in your memory.")
        
    elif emotion == "Calm" or emotion == "Surprise":
        reflection = f"It sounds like a moment of {emotion.lower()}. Those can be great times for quiet reflection or just enjoying the peace."
        exercise = ("**Actionable Tip: Mindful Check-in**\n\n"
                    "This is a great time to simply check in. Take 30 seconds to just notice your breath. You don't need to change it, just observe. It's a simple, powerful act of mindfulness.")

    else: # Default/Other emotions (like 'Love', 'Optimism', etc.)
        reflection = f"Thank you for sharing that you're feeling {emotion.lower()}. It's always good to acknowledge our feelings."
        exercise = ("**Actionable Tip: Quick Breath**\n\n"
                    "As a simple check-in, let's take one deep, slow breath. Inhale through your nose, and exhale slowly through your mouth. A nice little reset.")

    return reflection, exercise


# ----------------------------------------------------------------------
# 5. MAIN APP UI (Chat-Enabled with Session Logic)
# ----------------------------------------------------------------------

st.set_page_config(page_title="EchoMind AI Journal", layout="centered", initial_sidebar_state="expanded")
st.title("🧠 EchoMind: Your AI Wellness Companion")

# --- Session Controls in Sidebar ---
st.sidebar.title("Chat Controls")
if st.sidebar.button("Start New Chat Session"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to a new session. How are you feeling today?"}
    ]
    st.session_state.session_scores = [] # Reset session scores
    st.rerun()

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How are you feeling today? You can write as much or as little as you like."}
    ]
if "session_scores" not in st.session_state:
    st.session_state.session_scores = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Processing ---
if prompt := st.chat_input("Write about your day..."):
    # 1. Add user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run AI/NLP Analysis
    detected_emotion = analyze_emotion(prompt)
    stress_level, _ = analyze_stress(prompt)
    
    # 3. Get AI Suggestion (NOW returns 2 items)
    ai_reflection, ai_exercise = get_ai_suggestion(detected_emotion, stress_level)
    
    # 4. Save to *permanent* Database
    ai_suggestion_full = f"{ai_reflection}\n{ai_exercise}"
    add_entry(prompt, detected_emotion, stress_level, ai_suggestion_full)

    # 5. Session Logic
    st.session_state.session_scores.append(stress_level)
    session_avg_stress = sum(st.session_state.session_scores) / len(st.session_state.session_scores)
    
    # 6. Overall Logic
    all_entries = get_all_entries() 
    overall_avg_stress = 0.0
    if not all_entries.empty:
        overall_avg_stress = all_entries['stress_score'].mean()

    # 7. --- NEW: Format the Interactive AI response ---
    # Shorten the user's prompt for the quote
    quoted_prompt = (prompt[:75] + "...") if len(prompt) > 75 else prompt

    ai_response = f"""
    Thank you for sharing that. It sounds like you're dealing with a lot, especially when you say: *"{quoted_prompt}"*

    {ai_reflection}

    ---
    **Here's a small exercise you can try right now:**
    
    {ai_exercise}

    ---
    **Your Analysis & Stats:**
    * **Detected Emotion:** {detected_emotion}
    * **This Entry's Stress:** {stress_level}/10
    * **Session Average Stress:** {session_avg_stress:.1f}/10
    * **Overall Average Stress:** {overall_avg_stress:.1f}/10
    """

    # 8. Add AI's response to chat
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            time.sleep(1) 
            st.markdown(ai_response)
    
    # 9. Rerun to update the sidebars immediately
    st.rerun()


# ----------------------------------------------------------------------
# 6. VISUALIZATION DASHBOARDS (in Sidebar)
# ----------------------------------------------------------------------

# --- Current Session Dashboard ---
st.sidebar.header("Current Session Dashboard")
if not st.session_state.session_scores:
    st.sidebar.info("Your session dashboard will appear here once you send a message.")
else:
    session_avg = sum(st.session_state.session_scores) / len(st.session_state.session_scores)
    st.sidebar.metric(label="Session Average Stress", value=f"{session_avg:.1f}/10")
    
    st.sidebar.subheader("Session Stress Trend")
    session_df = pd.DataFrame({
        'Prompt #': range(1, len(st.session_state.session_scores) + 1),
        'Stress Score': st.session_state.session_scores
    })
    session_chart = px.line(session_df, x='Prompt #', y='Stress Score', markers=True)
    session_chart.update_yaxes(range=[0, 10]) 
    st.sidebar.plotly_chart(session_chart, use_container_width=True)


# --- Overall History Dashboard (in an expander) ---
with st.sidebar.expander("View Overall History Dashboard"):
    st.write("This shows all entries from all your sessions.")
    
    all_entries_df = get_all_entries()

    if all_entries_df.empty:
        st.sidebar.info("Your overall history will appear here after your first session.")
    else:
        overall_avg_stress = all_entries_df['stress_score'].mean()
        st.metric(label="Overall Average Stress", value=f"{overall_avg_stress:.1f}/10")

        st.subheader("Overall Stress Trend")
        stress_chart = px.line(
            all_entries_df, 
            x='timestamp', 
            y='stress_score', 
            title="Stress Score (All Time)",
            markers=True
        )
        stress_chart.update_yaxes(range=[0, 10])
        st.plotly_chart(stress_chart, use_container_width=True)

        st.subheader("Overall Emotion Distribution")
        emotion_counts = all_entries_df['detected_emotion'].value_counts().reset_index()
        emotion_counts.columns = ['emotion', 'count']
        
        pie_chart = px.pie(
            emotion_counts, 
            names='emotion', 
            values='count', 
            title="Emotion Distribution (All Time)"
        )
        st.plotly_chart(pie_chart, use_container_width=True)