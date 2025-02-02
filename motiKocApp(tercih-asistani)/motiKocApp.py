import streamlit as st
import google.generativeai as genai
from datetime import datetime, timedelta
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import time
import sqlite3
import hashlib
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
import base64
from io import BytesIO
import nest_asyncio
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()



nest_asyncio.apply()
# Page Configuration
st.set_page_config(
    page_title="MotiKoç - YKS Hazırlık Asistanı",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        /* Dark Theme Base */
        :root {
            --text-color: #ffffff;
            --accent-color: #ff4b4b;
            --success-color: #00cc99;
            --warning-color: #ffbb33;
        }
        
        /* Main Container */
        .main {
            color: var(--text-color);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: var(--secondary-background);
        }
        
        /* Cards */
        .custom-card {
            background-color: var(--secondary-background);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin: 10px 0;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: var(--accent-color);
            color: white;
            border-radius: 20px;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        }
        
        /* Text Inputs */
        .stTextInput>div>div>input {
            background-color: var(--secondary-background);
            color: var(--text-color);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Progress Bars */
        .stProgress>div>div>div {
            background-color: var(--accent-color);
        }
        
        /* Select Boxes */
        .stSelectbox>div>div {
            background-color: var(--secondary-background);
            color: var(--text-color);
        }
        
        /* Tabs */
        .stTabs {
            background-color: var(--secondary-background);
            border-radius: 10px;
        }
        
        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade-in {
            animation: fadeIn 0.5s ease-out forwards;
        }
        
        /* Tables */
        .dataframe {
            background-color: var(--secondary-background);
            color: var(--text-color);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Charts */
        .plotly-graph-div {
            background-color: var(--secondary-background) !important;
        }

        /* Modern Dark Theme */
        :root {
            --primary-color: #4F46E5;
            --secondary-color: #7C3AED;
            --background-color: #1A1A1A;
            --secondary-background: #2D2D2D;
            --text-color: #FFFFFF;
            --accent-color: #10B981;
            --error-color: #EF4444;
            --warning-color: #F59E0B;
        }
        
        /* Main Container Styling */
        .main {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        /* Enhanced Card Design */
        .custom-card {
            background: linear-gradient(145deg, 
                       rgba(45, 45, 45, 0.9), 
                       rgba(35, 35, 35, 0.9));
            padding: 1.5rem;
            border-radius: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
                       0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin: 1rem 0;
        }
        
        .custom-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1),
                       0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        /* Modern Button Styling */
        .stButton > button {
            background: linear-gradient(90deg, 
                       var(--primary-color), 
                       var(--secondary-color));
            color: white;
            border-radius: 0.75rem;
            padding: 0.5rem 1.5rem;
            border: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }
        
        /* Enhanced Input Fields */
        .stTextInput > div > div > input {
            background-color: var(--secondary-background);
            color: var(--text-color);
            border-radius: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 0.75rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2);
        }
        
        /* Progress Bars */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, 
                       var(--primary-color), 
                       var(--accent-color));
            border-radius: 1rem;
        }
        
        /* Badge Design */
        .badge-card {
            background: linear-gradient(135deg, 
                       rgba(79, 70, 229, 0.1), 
                       rgba(124, 58, 237, 0.1));
            border-radius: 1rem;
            padding: 1rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .badge-card:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
        }
        
        /* Chat Interface */
        .chat-message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0.75rem;
            background-color: var(--secondary-background);
        }
        
        .user-message {
            background-color: rgba(79, 70, 229, 0.1);
            margin-left: 2rem;
        }
        
        .bot-message {
            background-color: rgba(16, 185, 129, 0.1);
            margin-right: 2rem;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade-in {
            animation: fadeIn 0.5s ease-out forwards;
        }
        
        /* Enhanced Tables */
        .dataframe {
            background-color: var(--secondary-background);
            border-radius: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .dataframe th {
            background-color: rgba(79, 70, 229, 0.1);
            padding: 0.75rem;
        }
        
        .dataframe td {
            padding: 0.75rem;
        }
        
        /* Charts and Visualizations */
        .plotly-graph-div {
            background-color: var(--secondary-background) !important;
            border-radius: 0.75rem;
            padding: 1rem;
        }
        
        /* Sidebar Enhancement */
        .css-1d391kg {
            background-color: var(--secondary-background);
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--background-color);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--primary-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--secondary-color);
        }
    </style>
""", unsafe_allow_html=True)

# Database Setup ve diğer başlangıç fonksiyonları aynen kalıyor...

# Database Setup
def init_db():
    conn = sqlite3.connect('motikoc.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, 
                  username TEXT UNIQUE,
                  password TEXT,
                  name TEXT,
                  email TEXT,
                  grade TEXT,
                  city TEXT,
                  target_university TEXT,
                  target_department TEXT,
                  target_rank INTEGER,
                  study_type TEXT)''')
    
    # Study logs
    c.execute('''CREATE TABLE IF NOT EXISTS study_logs
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  subject TEXT,
                  topic TEXT,
                  duration INTEGER,
                  date TEXT,
                  performance_rating INTEGER,
                  notes TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    # Mood logs
    c.execute('''CREATE TABLE IF NOT EXISTS mood_logs
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  mood TEXT,
                  stress_level INTEGER,
                  notes TEXT,
                  date TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    # Goals
    c.execute('''CREATE TABLE IF NOT EXISTS goals
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  title TEXT,
                  deadline TEXT,
                  progress INTEGER,
                  completed BOOLEAN,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    # Subjects and Topics
    c.execute('''CREATE TABLE IF NOT EXISTS subjects
                 (id INTEGER PRIMARY KEY,
                  name TEXT,
                  category TEXT)''')
    
    # Default subjects
    subjects = [
        ('Türkçe', 'TYT'),
        ('Matematik', 'TYT'),
        ('Fizik', 'AYT'),
        ('Kimya', 'AYT'),
        ('Biyoloji', 'AYT'),
        ('Tarih', 'TYT'),
        ('Coğrafya', 'TYT'),
        ('Felsefe', 'TYT')
    ]
    
    c.executemany('''INSERT OR IGNORE INTO subjects (name, category)
                     VALUES (?, ?)''', subjects)
    # Saved Solutions table - YENİ EKLENEN TABLO
    c.execute('''CREATE TABLE IF NOT EXISTS saved_solutions
               (id INTEGER PRIMARY KEY,
                user_id INTEGER,
                subject TEXT,
                question TEXT,
                solution TEXT,
                date TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    
    # Social features
    c.execute('''CREATE TABLE IF NOT EXISTS friendships
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  friend_id INTEGER,
                  status TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id),
                  FOREIGN KEY(friend_id) REFERENCES users(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS achievements
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  title TEXT,
                  description TEXT,
                  date TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    # Additional Tables for YKS Features
    # Study Groups
    c.execute('''CREATE TABLE IF NOT EXISTS study_groups
                 (id INTEGER PRIMARY KEY,
                  creator_id INTEGER,
                  name TEXT,
                  group_type TEXT,
                  max_members INTEGER,
                  description TEXT,
                  subjects TEXT,
                  FOREIGN KEY(creator_id) REFERENCES users(id))''')
    
    # Group Members
    c.execute('''CREATE TABLE IF NOT EXISTS group_members
                 (id INTEGER PRIMARY KEY,
                  group_id INTEGER,
                  user_id INTEGER,
                  role TEXT,
                  FOREIGN KEY(group_id) REFERENCES study_groups(id),
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    # Mock Exams
    c.execute('''CREATE TABLE IF NOT EXISTS mock_exams
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  exam_type TEXT,
                  exam_date TEXT,
                  total_time INTEGER,
                  subject_results TEXT,
                  analysis TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    # Question Stats
    c.execute('''CREATE TABLE IF NOT EXISTS question_stats
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  subject TEXT,
                  topic TEXT,
                  correct INTEGER,
                  incorrect INTEGER,
                  unanswered INTEGER,
                  average_time INTEGER,
                  last_practice TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    # YKS Subjects
    c.execute('''CREATE TABLE IF NOT EXISTS yks_subjects
                 (id INTEGER PRIMARY KEY,
                  exam_type TEXT,  -- TYT/AYT
                  subject TEXT,
                  topic TEXT,
                  subtopic TEXT,
                  difficulty INTEGER,
                  importance INTEGER)''')
    
    conn.commit()
    conn.close()

def update_database_schema():
    conn = sqlite3.connect('motikoc.db')
    c = conn.cursor()
    
    # Badges table
    c.execute('''CREATE TABLE IF NOT EXISTS badges
                 (id INTEGER PRIMARY KEY,
                  name TEXT,
                  description TEXT,
                  icon TEXT,
                  category TEXT,
                  requirement_type TEXT,
                  requirement_value INTEGER)''')
    
    # User badges
    c.execute('''CREATE TABLE IF NOT EXISTS user_badges
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  badge_id INTEGER,
                  earned_date TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id),
                  FOREIGN KEY(badge_id) REFERENCES badges(id))''')
    
    # User level system
    c.execute('''CREATE TABLE IF NOT EXISTS user_levels
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  current_level INTEGER DEFAULT 1,
                  current_xp INTEGER DEFAULT 0,
                  total_xp INTEGER DEFAULT 0,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    # Daily tasks table
    c.execute('''CREATE TABLE IF NOT EXISTS daily_tasks
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  task_type TEXT,
                  description TEXT,
                  xp_reward INTEGER,
                  completed BOOLEAN DEFAULT FALSE,
                  date_created TEXT,
                  date_completed TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    # Competitions tables
    c.execute('''CREATE TABLE IF NOT EXISTS competitions
                 (id INTEGER PRIMARY KEY,
                  name TEXT,
                  description TEXT,
                  competition_type TEXT,
                  start_date TEXT,
                  end_date TEXT,
                  status TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS competition_participants
                 (id INTEGER PRIMARY KEY,
                  competition_id INTEGER,
                  user_id INTEGER,
                  score INTEGER DEFAULT 0,
                  FOREIGN KEY(competition_id) REFERENCES competitions(id),
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
   # Updated default badges with easier requirements
    default_badges = [
        ('Hoş Geldin!', 'MotiKoç ailesine katıldın', '👋', 'başlangıç', 'registration', 1),
        ('İlk Adım', 'İlk çalışma seansını tamamla', '🎯', 'başlangıç', 'study_sessions', 1),
        ('Günün Kahramanı', 'Bir günde 2 saat çalış', '⭐', 'çalışma', 'daily_hours', 2),
        ('Azimli Öğrenci', '5 saat çalış', '📚', 'çalışma', 'study_hours', 5),
        ('Matematik Sever', 'Matematik konularında 3 saat çalış', '🔢', 'matematik', 'math_hours', 3),
        ('Sosyal Kelebek', '3 arkadaş edin', '🦋', 'sosyal', 'friends', 3),
        ('Devamlılık', '3 gün üst üste çalış', '🎯', 'devamlılık', 'streak_days', 3),
        ('Planlı Çalışan', '5 günlük çalışma planı oluştur', '📅', 'planlama', 'study_plans', 5),
        ('Hedef Odaklı', 'İlk hedefini belirle', '🎯', 'hedefler', 'set_goals', 1),
        ('Motivasyon Ustası', '3 motivasyon seansı tamamla', '💪', 'motivasyon', 'motivation_sessions', 3),
        ('YKS Savaşçısı', 'Toplam 20 saat çalış', '⚔️', 'ileri', 'total_hours', 20),
        ('Deneme Uzmanı', 'İlk deneme sınavını gir', '📝', 'sınav', 'mock_exams', 1),
        ('Geri Bildirim', 'İlk performans değerlendirmeni yap', '📊', 'analiz', 'performance_review', 1)
    ]
    
    c.executemany('''INSERT OR IGNORE INTO badges 
                     (name, description, icon, category, requirement_type, requirement_value)
                     VALUES (?, ?, ?, ?, ?, ?)''', default_badges)
    
    conn.commit()
    conn.close()

def update_database_for_tasks():
    conn = sqlite3.connect('motikoc.db')
    c = conn.cursor()
    
    # Daily tasks table
    c.execute('''CREATE TABLE IF NOT EXISTS daily_tasks
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  task_type TEXT,
                  description TEXT,
                  xp_reward INTEGER,
                  completed BOOLEAN DEFAULT FALSE,
                  date_created TEXT,
                  date_completed TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

# Initialize all database tables
init_db()
update_database_schema()
update_database_for_tasks()

# Gemini AI Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
# Configure with the API key from environment variables
API_KEY = GEMINI_API_KEY

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-002')

def get_ai_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Üzgünüm, bir hata oluştu: {str(e)}"

# Authentication Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_user(username, password):
    conn = sqlite3.connect('motikoc.db')
    c = conn.cursor()
    c.execute('SELECT id, password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    
    if result and result[1] == hash_password(password):
        return result[0]
    return None

def register_user(username, password, name, email, grade, city,
                 target_university, target_department, target_rank, study_type):
    try:
        conn = sqlite3.connect('motikoc.db')
        c = conn.cursor()
        hashed_password = hash_password(password)
        
        # Insert user
        c.execute('''INSERT INTO users 
                    (username, password, name, email, grade, city,
                     target_university, target_department, target_rank, study_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (username, hashed_password, name, email, grade, city,
                  target_university, target_department, target_rank, study_type))
        
        user_id = c.lastrowid
        
        # Initialize user_levels for the new user
        c.execute('''INSERT INTO user_levels 
                    (user_id, current_level, current_xp, total_xp)
                    VALUES (?, 1, 0, 0)''', (user_id,))
        
        # Find and award welcome badge
        c.execute('''SELECT id FROM badges WHERE requirement_type = 'registration' LIMIT 1''')
        welcome_badge = c.fetchone()
        if welcome_badge:
            c.execute('''INSERT INTO user_badges (user_id, badge_id, earned_date)
                        VALUES (?, ?, ?)''',
                     (user_id, welcome_badge[0], datetime.now().strftime('%Y-%m-%d')))
            
            # Award XP for first badge
            c.execute('''UPDATE user_levels 
                        SET current_xp = current_xp + 100, total_xp = total_xp + 100
                        WHERE user_id = ?''', (user_id,))
        
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None

# Date Helper Function
def format_date(date_obj):
    if isinstance(date_obj, (datetime, pd.Timestamp)):
        return date_obj.strftime('%Y-%m-%d')
    return date_obj

# Clean and Parse JSON
def clean_and_parse_json(json_string):
    """
    Clean and parse AI-generated JSON string with better error handling and string cleaning.
    """
    try:
        # Remove code block markers if present
        json_string = json_string.strip()
        if json_string.startswith('```json'):
            json_string = json_string[7:]
        if json_string.startswith('```'):
            json_string = json_string[3:]
        if json_string.endswith('```'):
            json_string = json_string[:-3]
        json_string = json_string.strip()
        
        # Try parsing the cleaned JSON string
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {str(e)}")
        print(f"Problematic JSON string: {json_string}")
        return None

# User Interface Functions
def show_login_page():
    st.title("🎯 MotiKoç'a Hoş Geldiniz")
    
    tab1, tab2 = st.tabs(["Giriş Yap", "Kayıt Ol"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Kullanıcı Adı")
            password = st.text_input("Şifre", type="password")
            submit = st.form_submit_button("Giriş Yap")
            
            if submit:
                if username and password:
                    user_id = login_user(username, password)
                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.success("Giriş başarılı!")
                        st.rerun()
                    else:
                        st.error("Hatalı kullanıcı adı veya şifre!")
                else:
                    st.error("Lütfen tüm alanları doldurun!")

    with tab2:
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                username = st.text_input("Kullanıcı Adı*")
                password = st.text_input("Şifre*", type="password")
                name = st.text_input("Ad Soyad*")
                email = st.text_input("E-posta*")
                city = st.text_input("Şehir")
            
            with col2:
                grade = st.selectbox(
                    "Sınıf Düzeyi*",
                    ["9. Sınıf", "10. Sınıf", "11. Sınıf", "12. Sınıf", "Mezun"]
                )
                study_type = st.selectbox(
                    "Alan*",
                    ["Sayısal", "Eşit Ağırlık", "Sözel", "Dil"]
                )
                target_university = st.text_input("Hedef Üniversite")
                target_department = st.text_input("Hedef Bölüm")
                target_rank = st.number_input("Hedef Sıralama", min_value=1, step=1)
            
            submit = st.form_submit_button("Kayıt Ol")
            
            if submit:
                if username and password and name and email and grade and study_type:
                    user_id = register_user(
                        username, password, name, email, grade, city,
                        target_university, target_department, target_rank, study_type
                    )
                    if user_id:
                        st.success("Kayıt başarılı! Giriş yapabilirsiniz.")
                    else:
                        st.error("Bu kullanıcı adı zaten kullanımda!")
                else:
                    st.error("Lütfen zorunlu alanları doldurun!")

def show_home_page():
    st.title(f"Hoş Geldin, {st.session_state.username}! 👋")
    
    # Fetch user data
    conn = sqlite3.connect('motikoc.db')
    user_data = pd.read_sql_query('''
        SELECT grade, target_university, target_department, target_rank, study_type
        FROM users WHERE id = ?
    ''', conn, params=(st.session_state.user_id,))
    
    # Fetch recent study stats
    study_stats = pd.read_sql_query('''
        SELECT subject, SUM(duration) as total_duration
        FROM study_logs
        WHERE user_id = ? AND date >= date('now', '-7 days')
        GROUP BY subject
    ''', conn, params=(st.session_state.user_id,))
    conn.close()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # YKS Countdown
        exam_date = datetime(2025, 6, 15)
        days_left = (exam_date - datetime.now()).days
        
        st.markdown(f"""
        <div class="custom-card animate-fade-in">
            <h2>⏳ YKS'ye Kalan Süre</h2>
            <h1 style="color: var(--accent-color); text-align: center;">{days_left} Gün</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Study Summary
        st.markdown("""
        <div class="custom-card animate-fade-in">
            <h3>📚 Haftalık Çalışma Özeti</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if not study_stats.empty:
            fig = px.pie(
                study_stats,
                values='total_duration',
                names='subject',
                title='Ders Dağılımı (Son 7 Gün)'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig)
        else:
            st.info("Henüz çalışma kaydı bulunmuyor. Hadi çalışmaya başlayalım! 📚")
        
        # Daily Goals
        safe_show_daily_tasks()

    with col2:
        # Profile Summary
        st.markdown("""
        <div class="custom-card animate-fade-in">
            <h3>👤 Profil Özeti</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(f"🎓 Sınıf: {user_data.iloc[0]['grade']}")
        st.write(f"📚 Alan: {user_data.iloc[0]['study_type']}")
        if user_data.iloc[0]['target_university']:
            st.write(f"🎯 Hedef: {user_data.iloc[0]['target_university']}")
            st.write(f"📝 Bölüm: {user_data.iloc[0]['target_department']}")
            target_rank = user_data.iloc[0]['target_rank']
            if target_rank is not None:
                st.write(f"🏅 Hedef Sıralama: {target_rank:,}")
            else:
                st.write("🏅 Hedef Sıralama: Belirtilmemiş")
        
        # Motivation Message
        st.markdown("""
        <div class="custom-card animate-fade-in">
            <h3>💪 Motivasyon Mesajı</h3>
        </div>
        """, unsafe_allow_html=True)
        
        ai_prompt = f"""
        YKS'ye hazırlanan bir öğrenci için kısa ve motive edici bir mesaj yaz.
        Sınava {days_left} gün kaldı.
        """
        motivation = get_ai_response(ai_prompt)
        st.write(motivation)
        
        # Quick Actions
        st.markdown("""
        <div class="custom-card animate-fade-in">
            <h3>⚡ Hızlı İşlemler</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Yeni Çalışma"):
                st.session_state.page = "study_calendar"
                st.rerun()
        
        with col2:
            if st.button("🧘‍♂️ Nefes Egzersizi"):
                st.session_state.page = "voice_guidance"
                st.rerun()
        
        # Show gamification stats
        show_gamification_stats()

def show_study_calendar():
    st.title("📅 Çalışma Takvimi")
    
    # Calendar View
    st.markdown("""
    <div class="custom-card">
        <h3>Aylık Görünüm</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch study logs
    conn = sqlite3.connect('motikoc.db')
    study_logs = pd.read_sql_query('''
        SELECT subject, topic, duration, date, performance_rating
        FROM study_logs 
        WHERE user_id = ?
        ORDER BY date DESC
    ''', conn, params=(st.session_state.user_id,))
    conn.close()
    
    # Group by date for calendar view
    daily_logs = study_logs.groupby('date').agg({
        'duration': 'sum',
        'subject': lambda x: ', '.join(x)
    }).reset_index()
    
    # Display calendar (simplified version)
    current_month = datetime.now().month
    current_year = datetime.now().year

    # Handle December month to prevent month+1=13
    if current_month == 12:
        next_month = 1
        next_year = current_year + 1
    else:
        next_month = current_month + 1
        next_year = current_year

    # Create month calendar
    month_days = pd.date_range(
        start=f"{current_year}-{current_month}-01",
        end=f"{next_year}-{next_month}-01",
        freq='D'
    )[:-1]
    
    # Display calendar grid
    cols = st.columns(7)
    days = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz']
    for i, day in enumerate(days):
        cols[i].markdown(f"**{day}**")
    
    for i in range(0, len(month_days), 7):
        cols = st.columns(7)
        for j, date in enumerate(month_days[i:i+7]):
            date_str = date.strftime('%Y-%m-%d')
            study_data = daily_logs[daily_logs['date'] == date_str]
            
            if not study_data.empty:
                cols[j].markdown(f"""
                <div style='background-color: var(--accent-color); padding: 10px; border-radius: 5px;'>
                    {date.day}<br>
                    {study_data.iloc[0]['duration']} dk
                </div>
                """, unsafe_allow_html=True)
            else:
                cols[j].write(date.day)
    
    # Add New Study Session
    st.markdown("""
    <div class="custom-card">
        <h3>Yeni Çalışma Seansı</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("add_study_session"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            subjects = ["Matematik", "Fizik", "Kimya", "Biyoloji", "Türkçe", "Tarih", "Coğrafya", "Felsefe"]
            subject = st.selectbox("Ders", subjects)
            topic = st.text_input("Konu")
        
        with col2:
            date = st.date_input("Tarih")
            duration = st.number_input("Süre (dakika)", min_value=15, max_value=180, step=15)
        
        with col3:
            performance = st.slider("Performans Değerlendirmesi", 1, 5, 3)
            notes = st.text_area("Notlar")
        
        submit = st.form_submit_button("Ekle")
        
        if submit:
            if not isinstance(date, datetime):
                date = datetime.strptime(date, '%Y-%m-%d') if isinstance(date, str) else datetime.now()
            
            # Use the new add_study_session function instead of direct database insertion
            earned_badges = add_study_session(
                user_id=st.session_state.user_id,
                subject=subject,
                duration=duration,
                date=date.strftime('%Y-%m-%d'),
                performance_rating=performance,
                notes=notes
            )
            
            st.success("Çalışma seansı eklendi!")
            
            # Display any earned badges
            if earned_badges:
                st.balloons()
                for badge in earned_badges:
                    st.success(f"🎉 Yeni rozet kazandın: {badge[1]} {badge[2]}")
                
                # Add some encouragement based on the badge earned
                encouragements = {
                    'study_sessions': "Harika! İlk adımı attın, devam et! 💪",
                    'study_hours': "Çalışma hedefine ulaştın! 🎯",
                    'daily_hours': "Bugün çok verimli çalıştın! ⭐",
                    'streak_days': "Düzenli çalışma alışkanlığın gelişiyor! 📈",
                    'math_hours': "Matematik çalışmalarında ilerliyorsun! 🔢"
                }
                
                for badge in earned_badges:
                    if badge[4] in encouragements:  # badge[4] is requirement_type
                        st.info(encouragements[badge[4]])
            
            time.sleep(1)
            st.rerun()

# Helper function to format the encouragement message based on study duration
def get_study_encouragement(duration):
    if duration >= 120:  # 2 hours or more
        return "Muhteşem bir çalışma seansı! 🌟 Bu tempoda devam et!"
    elif duration >= 60:  # 1 hour or more
        return "Harika bir çalışma süresi! 💪 Kendini geliştirmeye devam et!"
    else:
        return "İyi bir başlangıç! 🎯 Her çalışma seni hedefine yaklaştırır!"

def show_performance_analytics():
    st.title("📊 Performans Analizi")
    
    # Fetch user data and study logs
    conn = sqlite3.connect('motikoc.db')
    study_data = pd.read_sql_query('''
        SELECT subject, topic, duration, performance_rating, date
        FROM study_logs
        WHERE user_id = ?
        ORDER BY date DESC
    ''', conn, params=(st.session_state.user_id,))
    
    user_data = pd.read_sql_query('''
        SELECT grade, target_university, target_department, target_rank
        FROM users
        WHERE id = ?
    ''', conn, params=(st.session_state.user_id,))
    conn.close()
    
    if not study_data.empty:
        # Overview Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_hours = study_data['duration'].sum() / 60
            st.metric("Toplam Çalışma", f"{total_hours:.1f} saat")
        
        with col2:
            avg_performance = study_data['performance_rating'].mean()
            if pd.notnull(avg_performance):
                st.metric("Ortalama Performans", f"{avg_performance:.1f}/5")
            else:
                st.metric("Ortalama Performans", "Veri yok")
        
        with col3:
            most_studied = study_data.groupby('subject')['duration'].sum().idxmax()
            st.metric("En Çok Çalışılan", most_studied)
        
        with col4:
            daily_average = study_data.groupby('date')['duration'].sum().mean() / 60
            st.metric("Günlük Ortalama", f"{daily_average:.1f} saat")
        
        # Detailed Analysis Tabs
        tab1, tab2, tab3 = st.tabs(["Ders Analizi", "Zaman Analizi", "AI Değerlendirme"])
        
        with tab1:
            st.markdown("""
            <div class="custom-card">
                <h3>Ders Bazında Analiz</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Study time distribution
                fig = px.pie(
                    study_data.groupby('subject')['duration'].sum().reset_index(),
                    values='duration',
                    names='subject',
                    title='Çalışma Süresi Dağılımı'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig)
            
            with col2:
                # Performance by subject
                performance_by_subject = study_data.groupby('subject')['performance_rating'].mean()
                fig = px.bar(
                    performance_by_subject,
                    title='Ders Başarı Ortalamaları',
                    color=performance_by_subject.values,
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig)
        
        with tab2:
            st.markdown("""
            <div class="custom-card">
                <h3>Zaman Bazlı Analiz</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Time series analysis
            daily_study = study_data.groupby('date')['duration'].sum().reset_index()
            fig = px.line(
                daily_study,
                x='date',
                y='duration',
                title='Günlük Çalışma Süresi Trendi'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig)
            
            # Weekly patterns
            study_data['weekday'] = pd.to_datetime(study_data['date']).dt.day_name()
            weekly_pattern = study_data.groupby('weekday')['duration'].mean().reset_index()
            # To ensure correct order of weekdays
            weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_pattern['weekday'] = pd.Categorical(weekly_pattern['weekday'], categories=weekdays_order, ordered=True)
            weekly_pattern = weekly_pattern.sort_values('weekday')
            
            fig = px.bar(
                weekly_pattern,
                x='weekday',
                y='duration',
                title='Haftalık Çalışma Düzeni',
                labels={'weekday': 'Haftanın Günü', 'duration': 'Ortalama Süre (dk)'}
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig)
        
        with tab3:
            st.markdown("""
            <div class="custom-card">
                <h3>🤖 MotiKoç'un Değerlendirmesi</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate AI analysis
            study_summary = f"""
            Çalışma Verileri:
            - Toplam çalışma: {total_hours:.1f} saat
            - Günlük ortalama: {daily_average:.1f} saat
            - En çok çalışılan ders: {most_studied}
            - Ortalama performans: {avg_performance:.1f}/5
            
            Öğrenci Hedefleri:
            - Hedef Üniversite: {user_data.iloc[0]['target_university']}
            - Hedef Bölüm: {user_data.iloc[0]['target_department']}
            - Hedef Sıralama: {user_data.iloc[0]['target_rank']}
            """
            
            ai_prompt = f"""
            Bir YKS koçu olarak öğrencinin çalışma verilerini analiz et ve önerilerde bulun.
            {study_summary}
            
            Lütfen şu başlıklarda değerlendirme yap:
            1. Güçlü Yönler
            2. Gelişim Alanları
            3. Öneriler
            """
            
            analysis = get_ai_response(ai_prompt)
            st.write(analysis)
    
    else:
        st.info("Henüz çalışma kaydınız bulunmuyor. Analiz için çalışma seansları eklemeye başlayın!")

def show_social_features():
    st.title("👥 Sosyal Özellikler")
    
    tab1, tab2, tab3 = st.tabs(["Arkadaşlar", "Başarılar", "Liderlik Tablosu"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="custom-card">
                <h3>Arkadaş Ekle</h3>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("add_friend"):
                username = st.text_input("Kullanıcı Adı")
                submit = st.form_submit_button("Arkadaş Ekle")
                
                if submit and username:
                    conn = sqlite3.connect('motikoc.db')
                    c = conn.cursor()
                    
                    # Check if user exists
                    c.execute('SELECT id FROM users WHERE username = ?', (username,))
                    friend = c.fetchone()
                    
                    if friend:
                        # Check if friendship already exists
                        c.execute('''
                            SELECT * FROM friendships 
                            WHERE (user_id = ? AND friend_id = ?) 
                            OR (user_id = ? AND friend_id = ?)
                        ''', (st.session_state.user_id, friend[0], 
                             friend[0], st.session_state.user_id))
                        
                        if not c.fetchone():
                            c.execute('''
                                INSERT INTO friendships (user_id, friend_id, status)
                                VALUES (?, ?, 'pending')
                            ''', (st.session_state.user_id, friend[0]))
                            conn.commit()
                            st.success("Arkadaşlık isteği gönderildi!")
                        else:
                            st.warning("Bu kullanıcı zaten arkadaş listenizde!")
                    else:
                        st.error("Kullanıcı bulunamadı!")
                    
                    conn.close()
        
        with col2:
            st.markdown("""
            <div class="custom-card">
                <h3>Arkadaş Listesi</h3>
            </div>
            """, unsafe_allow_html=True)
            
            conn = sqlite3.connect('motikoc.db')
            # Fetch confirmed friends
            friends = pd.read_sql_query('''
                SELECT u.username, f.status
                FROM friendships f
                JOIN users u ON u.id = CASE 
                                        WHEN f.user_id = ? THEN f.friend_id
                                        ELSE f.user_id
                                      END
                WHERE f.user_id = ? OR f.friend_id = ?
                AND f.status = 'confirmed'
            ''', conn, params=(st.session_state.user_id,
                               st.session_state.user_id,
                               st.session_state.user_id))
            conn.close()
            
            if not friends.empty:
                for _, friend in friends.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"👤 {friend['username']}")
                    with col2:
                        st.write(f"✅ {friend['status']}")
            else:
                st.write("Henüz arkadaşınız yok.")
    
    with tab2:
        st.markdown("""
        <div class="custom-card">
            <h3>Başarılarım</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("add_achievement"):
            achievement = st.text_input("Yeni Başarı")
            description = st.text_area("Açıklama")
            submit = st.form_submit_button("Paylaş")
            
            if submit and achievement:
                conn = sqlite3.connect('motikoc.db')
                c = conn.cursor()
                c.execute('''
                    INSERT INTO achievements (user_id, title, description, date)
                    VALUES (?, ?, ?, ?)
                ''', (st.session_state.user_id, achievement, description,
                      datetime.now().strftime('%Y-%m-%d')))
                conn.commit()
                conn.close()
                
                # Award XP for sharing achievement
                update_user_xp(st.session_state.user_id, calculate_xp_for_activity('achievement_shared'))
                
                st.success("Başarı paylaşıldı!")
        
        # Display achievements
        conn = sqlite3.connect('motikoc.db')
        achievements = pd.read_sql_query('''
            SELECT title, description, date
            FROM achievements
            WHERE user_id = ?
            ORDER BY date DESC
        ''', conn, params=(st.session_state.user_id,))
        conn.close()
        
        if not achievements.empty:
            for _, achievement in achievements.iterrows():
                st.markdown(f"""
                <div class="custom-card">
                    <h4>🏆 {achievement['title']}</h4>
                    <p>{achievement['description']}</p>
                    <small>{achievement['date']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("Henüz başarı paylaşımınız yok.")
    
    with tab3:
        st.markdown("""
        <div class="custom-card">
            <h3>Liderlik Tablosu</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate user scores based on study time and performance
        conn = sqlite3.connect('motikoc.db')
        leaderboard_data = pd.read_sql_query('''
            SELECT u.username,
                   COUNT(s.id) as study_sessions,
                   COALESCE(SUM(s.duration), 0) as total_duration,
                   COALESCE(AVG(s.performance_rating), 0) as avg_performance
            FROM users u
            LEFT JOIN study_logs s ON u.id = s.user_id
            GROUP BY u.id, u.username
            ORDER BY total_duration DESC
            LIMIT 10
        ''', conn)
        conn.close()
        
        if not leaderboard_data.empty:
            # Calculate score
            leaderboard_data['score'] = (
                leaderboard_data['total_duration'] * 0.5 +
                leaderboard_data['avg_performance'] * 200 +
                leaderboard_data['study_sessions'] * 10
            ).round()
            
            # Sort by score descending
            leaderboard_data = leaderboard_data.sort_values(by='score', ascending=False)
            
            # Display leaderboard
            st.table(leaderboard_data[['username', 'score']]
                    .rename(columns={'username': 'Kullanıcı', 'score': 'Puan'}))
        else:
            st.write("Henüz liderlik tablosu oluşturulmadı.")
def show_settings():
    st.title("⚙️ Ayarlar")
    st.write("Bu sayfa, uygulamanın ayarlarını içerir. Ek ayar seçenekleri burada tanımlanabilir.")
    # Add any settings options you want here, for example:
    theme = st.selectbox("Tema Seçimi", ["Karanlık Mod", "Açık Mod"])
    notifications = st.checkbox("Bildirimleri Aktif Et")
    
    # Save settings to session state if needed
    st.session_state['theme'] = theme
    st.session_state['notifications'] = notifications

    st.success("Ayarlar kaydedildi.")

def show_voice_guidance():
    st.title("🎧 Sesli Rehberlik")
    
    guidance_type = st.selectbox(
        "Rehberlik Türü",
        ["Meditasyon", "Nefes Egzersizi", "Motivasyon", "Konu Anlatımı"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if guidance_type == "Meditasyon":
            st.markdown("""
            <div class="custom-card">
                <h3>Meditasyon Seansı</h3>
                <p>Sınav kaygısını azaltmak için özel olarak tasarlanmış meditasyon seansları.</p>
            </div>
            """, unsafe_allow_html=True)
            
            meditation_type = st.selectbox(
                "Meditasyon Türü",
                ["Sınav Kaygısı Azaltma (10 dk)",
                 "Odaklanma Meditasyonu (5 dk)",
                 "Rahatlama Seansı (15 dk)"]
            )
            
            if st.button("Seansı Başlat"):
                duration = int(meditation_type.split("(")[1].split(" dk")[0]) * 60
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                meditation_prompts = [
                    "Derin bir nefes al...",
                    "Zihnini sakinleştir...",
                    "Kendini güvende hisset...",
                    "Stresini bırak...",
                    "Başaracağına inan..."
                ]
                
                for i in range(duration):
                    progress = (i + 1) / duration
                    progress_bar.progress(progress)
                    
                    if i % 12 == 0:
                        status_text.write(
                            meditation_prompts[int(i/12) % len(meditation_prompts)]
                        )
                    time.sleep(1)
                
                st.success("Meditasyon tamamlandı! 🧘‍♂️")
                
                # Update meditation count
                if "meditation_count" not in st.session_state:
                    st.session_state.meditation_count = 0
                st.session_state.meditation_count += 1
        elif guidance_type == "Nefes Egzersizi":
            st.markdown("""
            <div class="custom-card">
                <h3>Nefes Egzersizleri</h3>
                <p>Sınav öncesi ve çalışma aralarında rahatlamanıza yardımcı olacak nefes teknikleri.</p>
            </div>
            """, unsafe_allow_html=True)
            
            breathing_type = st.selectbox(
                "Nefes Tekniği",
                ["4-7-8 Tekniği", "Kutu Nefes", "Sakinleştirici Nefes"]
            )
            
            if st.button("Egzersizi Başlat"):
                cycles = 5
                if breathing_type == "4-7-8 Tekniği":
                    steps = [
                        ("Nefes Al", 4),
                        ("Tut", 7),
                        ("Nefes Ver", 8)
                    ]
                else:  # Kutu Nefes veya Sakinleştirici Nefes
                    steps = [
                        ("Nefes Al", 4),
                        ("Tut", 4),
                        ("Nefes Ver", 4),
                        ("Bekle", 4)
                    ]
                
                progress_bar = st.progress(0)
                instruction_text = st.empty()
                
                total_duration = sum(step[1] for step in steps) * cycles
                current_time = 0
                
                for cycle in range(cycles):
                    for instruction, duration in steps:
                        instruction_text.markdown(f"""
                        <div style='text-align: center; font-size: 24px;'>
                            {instruction}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for _ in range(duration):
                            current_time += 1
                            progress = current_time / total_duration
                            progress_bar.progress(progress)
                            time.sleep(1)
                
                st.success("Egzersiz tamamlandı! 🌬️")
        
        elif guidance_type == "Motivasyon":
            st.markdown("""
            <div class="custom-card">
                <h3>Günlük Motivasyon</h3>
                <p>YKS hazırlık sürecinde size güç verecek motivasyon mesajları.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Fetch user data for personalized motivation
            conn = sqlite3.connect('motikoc.db')
            user_data = pd.read_sql_query('''
            SELECT name, grade, target_university, target_department, target_rank
            FROM users WHERE id = ?
        ''', conn, params=(st.session_state.user_id,))

            conn.close()
            
            days_to_exam = (datetime(2025, 6, 15) - datetime.now()).days
            
            ai_prompt = f"""
            YKS'ye hazırlanan bir öğrenci için motive edici bir konuşma hazırla.
            
            Öğrenci Bilgileri:
            - Öğrenci adı {user_data.iloc[0]['name']}
            - Sınıf: {user_data.iloc[0]['grade']}
            - Hedef: {user_data.iloc[0]['target_university']}
            - Bölüm: {user_data.iloc[0]['target_department']}
            - Sınava kalan gün: {days_to_exam}
            
            Lütfen samimi, cesaret verici ve gerçekçi bir konuşma hazırla.
            """
            
            motivation_speech = get_ai_response(ai_prompt)
            
            st.markdown(f"""
            <div class="custom-card" style="background-color: #2d3748; padding: 20px; border-radius: 10px;">
                <h4 style="color: #f8f9fa;">💪 Motivasyon Konuşması</h4>
                <p style="color: #f8f9fa; font-size: 18px;">{motivation_speech}</p>
            </div>
            """, unsafe_allow_html=True)
        
        elif guidance_type == "Konu Anlatımı":
            st.markdown("""
            <div class="custom-card">
                <h3>Sesli Konu Anlatımı</h3>
                <p>Önemli konuların özet anlatımları ve tekrar rehberi.</p>
            </div>
            """, unsafe_allow_html=True)
            
            subject = st.selectbox(
                "Ders Seçin",
                ["Matematik", "Fizik", "Kimya", "Biyoloji", "Türkçe", "Tarih"]
            )
            
            topics = {
                "Matematik": ["Türev", "İntegral", "Limit", "Fonksiyonlar"],
                "Fizik": ["Kuvvet", "Hareket", "Elektrik", "Manyetizma"],
                "Kimya": ["Asitler", "Gazlar", "Karışımlar", "Tepkimeler"],
                "Biyoloji": ["Hücre", "Sistemler", "Kalıtım", "Evrim"],
                "Türkçe": ["Paragraf", "Dil Bilgisi", "Anlatım", "Şiir"],
                "Tarih": ["Osmanlı", "Cumhuriyet", "İnkılaplar", "Çağdaş"]
            }
            
            topic = st.selectbox("Konu Seçin", topics[subject])
            
            if st.button("Anlatımı Başlat"):
                explanation_prompt = f"""
                {subject} dersinin {topic} konusunu YKS adayı bir öğrenciye anlatır gibi
                kısa ve öz bir şekilde açıkla. Önemli noktaları vurgula ve örneklerle destekle.
                """
                
                explanation = get_ai_response(explanation_prompt)
                
                st.markdown(f"""
                <div class="custom-card">
                    <h4>📚 {subject} - {topic}</h4>
                    <p>{explanation}</p>
                </div>
                """, unsafe_allow_html=True)

# Database connection manager
def get_db_connection():
    """Get a database connection with proper timeout and isolation level"""
    conn = sqlite3.connect('motikoc.db', timeout=30)
    conn.execute('PRAGMA journal_mode=WAL')  # Write-Ahead Logging mode
    conn.row_factory = sqlite3.Row
    return conn

def add_study_session(user_id, subject, duration, date, performance_rating, notes):
    """Add study session with improved connection handling"""
    conn = get_db_connection()
    
    try:
        with conn:  # This automatically handles commit/rollback
            cursor = conn.cursor()
            
            # Add study session
            cursor.execute('''INSERT INTO study_logs 
                            (user_id, subject, duration, date, performance_rating, notes)
                            VALUES (?, ?, ?, ?, ?, ?)''',
                         (user_id, subject, duration, date, performance_rating, notes))
            
            # Calculate XP
            base_xp = 50  # Base XP for completing a study session
            duration_xp = duration // 15 * 10  # 10 XP per 15 minutes
            performance_xp = performance_rating * 10  # XP based on performance rating
            total_xp = base_xp + duration_xp + performance_xp
            
            # Update user XP in the same transaction
            cursor.execute('''UPDATE user_levels 
                            SET current_xp = current_xp + ?,
                                total_xp = total_xp + ?
                            WHERE user_id = ?''',
                         (total_xp, total_xp, user_id))
            
            # Check and award badges
            earned_badges = []
            
            # Get available badges
            cursor.execute('''SELECT b.* FROM badges b
                            WHERE NOT EXISTS (
                                SELECT 1 FROM user_badges ub 
                                WHERE ub.badge_id = b.id 
                                AND ub.user_id = ?
                            )''', (user_id,))
            available_badges = cursor.fetchall()
            
            for badge in available_badges:
                badge_id = badge[0]
                requirement_type = badge[5]
                requirement_value = badge[6]
                
                # Check badge requirements
                if requirement_type == 'study_sessions':
                    cursor.execute('''SELECT COUNT(*) FROM study_logs
                                    WHERE user_id = ?''', (user_id,))
                    count = cursor.fetchone()[0]
                    if count >= requirement_value:
                        earned_badges.append(badge)
                
                elif requirement_type in ['study_hours', 'math_hours']:
                    subject_filter = "AND subject = 'Matematik'" if requirement_type == 'math_hours' else ""
                    cursor.execute(f'''SELECT COALESCE(SUM(duration), 0) FROM study_logs
                                     WHERE user_id = ? {subject_filter}''', (user_id,))
                    minutes = cursor.fetchone()[0]
                    hours = minutes / 60
                    if hours >= requirement_value:
                        earned_badges.append(badge)
                
                elif requirement_type == 'daily_hours':
                    cursor.execute('''SELECT COALESCE(SUM(duration), 0) FROM study_logs
                                    WHERE user_id = ? AND date = date('now')''', (user_id,))
                    minutes = cursor.fetchone()[0]
                    if minutes >= (requirement_value * 60):
                        earned_badges.append(badge)
                
                elif requirement_type == 'streak_days':
                    cursor.execute('''SELECT DISTINCT date FROM study_logs
                                    WHERE user_id = ?
                                    ORDER BY date DESC''', (user_id,))
                    dates = cursor.fetchall()
                    if dates:
                        dates = [datetime.strptime(d[0], '%Y-%m-%d') for d in dates]
                        streak = 1
                        for i in range(len(dates)-1):
                            if (dates[i] - dates[i+1]).days == 1:
                                streak += 1
                            else:
                                break
                        if streak >= requirement_value:
                            earned_badges.append(badge)
            
            # Award earned badges
            for badge in earned_badges:
                try:
                    cursor.execute('''INSERT INTO user_badges (user_id, badge_id, earned_date)
                                    VALUES (?, ?, ?)''',
                                 (user_id, badge[0], datetime.now().strftime('%Y-%m-%d')))
                except sqlite3.IntegrityError:
                    continue  # Skip if already awarded
            
            return earned_badges
    
    except Exception as e:
        print(f"Error in add_study_session: {str(e)}")
        raise
    
    finally:
        conn.close()
def show_gamification_stats():
    st.markdown("""
    <div class="custom-card">
        <h3>🎮 Oyun İstatistikleri</h3>
    </div>
    """, unsafe_allow_html=True)
    
    conn = sqlite3.connect('motikoc.db')
    
    try:
        # Get user level info
        level_data = pd.read_sql_query('''
            SELECT current_level, current_xp, total_xp
            FROM user_levels WHERE user_id = ?
        ''', conn, params=(st.session_state.user_id,))
        
        if level_data.empty:
            # Initialize user levels if not exists
            c = conn.cursor()
            c.execute('''INSERT INTO user_levels (user_id, current_level, current_xp, total_xp)
                        VALUES (?, 1, 0, 0)''', (st.session_state.user_id,))
            conn.commit()
            level_data = pd.read_sql_query('''
                SELECT current_level, current_xp, total_xp
                FROM user_levels WHERE user_id = ?
            ''', conn, params=(st.session_state.user_id,))
        
        current_level = level_data.iloc[0]['current_level']
        current_xp = level_data.iloc[0]['current_xp']
        xp_for_next = current_level * 1000
        
        # Main stats columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Level Stats
            st.metric("Seviye", current_level)
            progress = min(1.0, current_xp / xp_for_next) if xp_for_next else 0
            st.progress(progress)
            st.write(f"XP: {current_xp}/{xp_for_next}")
            
            # Study Stats
            study_stats = pd.read_sql_query('''
                SELECT 
                    COUNT(DISTINCT date) as study_days,
                    COUNT(*) as total_sessions,
                    COALESCE(SUM(duration), 0) as total_minutes,
                    COALESCE(AVG(performance_rating), 0) as avg_performance
                FROM study_logs 
                WHERE user_id = ?
            ''', conn, params=(st.session_state.user_id,))
            
            if not study_stats.empty:
                stats = study_stats.iloc[0]
                st.markdown("### 📊 Çalışma İstatistikleri")
                st.write(f"🗓️ Toplam Çalışma Günü: {stats['study_days']}")
                st.write(f"📚 Toplam Çalışma Seansı: {stats['total_sessions']}")
                total_hours = stats['total_minutes'] / 60
                st.write(f"⏱️ Toplam Çalışma Süresi: {total_hours:.1f} saat")
                st.write(f"⭐ Ortalama Performans: {stats['avg_performance']:.1f}/5")
        
        with col2:
            st.markdown("### 🏆 Rozetleriniz")
            
            # Get earned badges with no duplicates
            badges = pd.read_sql_query('''
                SELECT DISTINCT b.name, b.icon, b.description, 
                       MIN(ub.earned_date) as earned_date,
                       b.category
                FROM badges b
                JOIN user_badges ub ON b.id = ub.badge_id
                WHERE ub.user_id = ?
                GROUP BY b.id
                ORDER BY ub.earned_date DESC
            ''', conn, params=(st.session_state.user_id,))
            
            if not badges.empty:
                # Group badges by category
                categories = badges['category'].unique()
                for category in categories:
                    cat_badges = badges[badges['category'] == category]
                    if not cat_badges.empty:
                        st.markdown(f"#### {category.title()} Rozetleri")
                        for _, badge in cat_badges.iterrows():
                            # Use markdown for badge display instead of nested columns
                            st.markdown(f"""
                            <div style='padding: 10px; border-radius: 5px; background-color: rgba(255, 255, 255, 0.1);'>
                                <span style='font-size: 24px;'>{badge['icon']}</span>
                                <strong style='margin-left: 10px;'>{badge['name']}</strong>
                                <br/>
                                <span style='color: #B0B0B0; font-size: 14px;'>{badge['description']}</span>
                                <br/>
                                <span style='color: #888888; font-size: 12px;'>Kazanıldı: {badge['earned_date']}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown("<br/>", unsafe_allow_html=True)
            else:
                # Check for new badges
                earned_badges = check_and_award_badges(st.session_state.user_id)
                if earned_badges:
                    st.rerun()
                else:
                    st.info("Henüz rozet kazanılmadı! Çalışmaya devam et!")
                    
                    # Show available badges
                    st.markdown("#### 🎯 Kazanılabilecek Rozetler")
                    available_badges = pd.read_sql_query('''
                        SELECT name, description, icon, requirement_value
                        FROM badges
                        WHERE id NOT IN (
                            SELECT badge_id FROM user_badges WHERE user_id = ?
                        )
                    ''', conn, params=(st.session_state.user_id,))
                    
                    for _, badge in available_badges.iterrows():
                        st.markdown(f"""
                        <div style='padding: 10px; border-radius: 5px; background-color: rgba(255, 255, 255, 0.05);'>
                            <span style='font-size: 24px;'>{badge['icon']}</span>
                            <strong style='margin-left: 10px;'>{badge['name']}</strong>
                            <br/>
                            <span style='color: #B0B0B0; font-size: 14px;'>{badge['description']}</span>
                            {f"<br/><span style='color: #888888; font-size: 12px;'>Gereksinim: {badge['requirement_value']}</span>" if badge['requirement_value'] else ""}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<br/>", unsafe_allow_html=True)
            
            # Show achievement stats
            total_badges = len(badges)
            if total_badges > 0:
                st.markdown("#### 🌟 Başarı Özeti")
                st.metric("Toplam Rozet", total_badges)
                
                # Calculate completion percentage
                total_possible_badges = pd.read_sql_query('''
                    SELECT COUNT(*) as total FROM badges
                ''', conn).iloc[0]['total']
                
                completion = (total_badges / total_possible_badges) * 100
                st.progress(completion / 100)
                st.caption(f"Tüm rozetlerin %{completion:.1f}'ini kazandınız")
    
    except Exception as e:
        st.error(f"İstatistikler yüklenirken bir hata oluştu: {str(e)}")
        print(f"Hata detayı: {str(e)}")
    
    finally:
        conn.close()
def calculate_xp_for_activity(activity_type, duration=None):
    """XP hesaplama fonksiyonunu güncelle"""
    xp_rates = {
        # Mevcut XP oranları
        'study_session': 50,
        'study_minute': 1,
        'performance_bonus': 20,
        'streak_bonus': 100,
        'badge_earned': 200,
        'achievement_shared': 150,
        
        # Forum XP oranları
        'forum_question': 75,    # Soru sorma
        'forum_answer': 100,     # Cevap verme
        'answer_accepted': 200,  # Cevabı kabul edildi
        'upvote_received': 10,   # Beğeni alma
        'answer_upvoted': 5      # Beğeni verme
    }
    
    if activity_type == 'study_session' and duration:
        base_xp = xp_rates['study_session']
        minute_xp = min(duration * xp_rates['study_minute'], 500)
        return base_xp + minute_xp
    
    return xp_rates.get(activity_type, 0)

def update_user_xp(user_id, xp_earned):
    """Update user XP with level progression handling"""
    with sqlite3.connect('motikoc.db') as conn:
        c = conn.cursor()
        
        # Get current level and XP
        c.execute('''SELECT current_level, current_xp, total_xp 
                     FROM user_levels WHERE user_id = ?''', (user_id,))
        result = c.fetchone()
        
        if result:
            current_level, current_xp, total_xp = result
        else:
            # Initialize new user
            c.execute('''INSERT INTO user_levels (user_id, current_level, current_xp, total_xp)
                         VALUES (?, 1, 0, 0)''', (user_id,))
            current_level, current_xp, total_xp = 1, 0, 0
        
        # Update XP
        new_total_xp = total_xp + xp_earned
        new_current_xp = current_xp + xp_earned
        
        # Check for level up
        while True:
            xp_for_next_level = current_level * 1000  # Each level requires 1000 * level XP
            if new_current_xp >= xp_for_next_level:
                new_current_xp -= xp_for_next_level
                current_level += 1
            else:
                break
        
        # Update database
        c.execute('''UPDATE user_levels 
                     SET current_level = ?, current_xp = ?, total_xp = ?
                     WHERE user_id = ?''',
                  (current_level, new_current_xp, new_total_xp, user_id))
        
        conn.commit()
    
    return current_level, new_current_xp, xp_for_next_level
# Badge checking function
def check_and_award_badges(user_id):
    """Enhanced badge checking function with duplicate prevention"""
    conn = sqlite3.connect('motikoc.db')
    c = conn.cursor()
    
    try:
        # Get badges that haven't been earned yet
        c.execute('''SELECT b.* FROM badges b
                     WHERE NOT EXISTS (
                         SELECT 1 FROM user_badges ub 
                         WHERE ub.badge_id = b.id 
                         AND ub.user_id = ?
                     )''', (user_id,))
        available_badges = c.fetchall()
        
        earned_badges = []
        for badge in available_badges:
            badge_id, name, description, icon, category, req_type, req_value = badge
            
            # Skip if badge already earned
            c.execute('''SELECT 1 FROM user_badges 
                        WHERE user_id = ? AND badge_id = ?''', 
                     (user_id, badge_id))
            if c.fetchone():
                continue
            
            # Check requirements based on type
            meets_requirements = False
            
            if req_type == 'registration':
                meets_requirements = True
                
            elif req_type == 'study_sessions':
                c.execute('''SELECT COUNT(DISTINCT id) FROM study_logs
                            WHERE user_id = ?''', (user_id,))
                count = c.fetchone()[0]
                meets_requirements = count >= req_value
                
            elif req_type in ['study_hours', 'math_hours']:
                subject_filter = "AND subject = 'Matematik'" if req_type == 'math_hours' else ""
                c.execute(f'''SELECT COALESCE(SUM(duration), 0) FROM study_logs
                             WHERE user_id = ? {subject_filter}''', (user_id,))
                minutes = c.fetchone()[0]
                hours = minutes / 60
                meets_requirements = hours >= req_value
                
            elif req_type == 'daily_hours':
                c.execute('''SELECT COALESCE(SUM(duration), 0) FROM study_logs
                            WHERE user_id = ? 
                            AND date = date('now')
                            GROUP BY date''', (user_id,))
                result = c.fetchone()
                daily_minutes = result[0] if result else 0
                meets_requirements = daily_minutes >= (req_value * 60)
            
            elif req_type == 'streak_days':
                c.execute('''SELECT DISTINCT date FROM study_logs
                            WHERE user_id = ?
                            ORDER BY date DESC''', (user_id,))
                dates = [datetime.strptime(d[0], '%Y-%m-%d') for d in c.fetchall()]
                
                streak = 1
                for i in range(len(dates)-1):
                    if (dates[i] - dates[i+1]).days == 1:
                        streak += 1
                    else:
                        break
                meets_requirements = streak >= req_value

            elif req_type == 'mock_exams':
                c.execute('''SELECT COUNT(*) FROM mock_exams
                            WHERE user_id = ?''', (user_id,))
                count = c.fetchone()[0]
                if count >= req_value:
                    earned_badges.append(badge)
            
            elif req_type == 'set_goals':
                c.execute('''SELECT COUNT(*) FROM goals
                            WHERE user_id = ?''', (user_id,))
                count = c.fetchone()[0]
                if count >= req_value:
                    earned_badges.append(badge)
            
            elif req_type == 'motivation_sessions':
                c.execute('''SELECT COUNT(*) FROM mood_logs
                            WHERE user_id = ?''', (user_id,))
                count = c.fetchone()[0]
                if count >= req_value:
                    earned_badges.append(badge)
            
            # Award badge if requirements are met
            if meets_requirements:
                try:
                    c.execute('''INSERT INTO user_badges (user_id, badge_id, earned_date)
                               VALUES (?, ?, ?)''',
                             (user_id, badge_id, datetime.now().strftime('%Y-%m-%d')))
                    earned_badges.append(badge)
                except sqlite3.IntegrityError:
                    continue  # Skip if already awarded
                    
        conn.commit()
        return earned_badges
        
    except Exception as e:
        print(f"Error in check_and_award_badges: {str(e)}")
        return []
    
    finally:
        conn.close()

def show_badges(user_id):
    """Improved badge display function"""
    conn = sqlite3.connect('motikoc.db')
    try:
        # Get earned badges with no duplicates
        badges = pd.read_sql_query('''
            SELECT DISTINCT b.name, b.icon, b.description, 
                   MIN(ub.earned_date) as earned_date
            FROM badges b
            JOIN user_badges ub ON b.id = ub.badge_id
            WHERE ub.user_id = ?
            GROUP BY b.id
            ORDER BY ub.earned_date DESC
        ''', conn, params=(user_id,))
        
        if not badges.empty:
            for _, badge in badges.iterrows():
                st.markdown(f"{badge['icon']} **{badge['name']}**")
                st.caption(f"{badge['description']}")
        else:
            st.info("Henüz rozet kazanılmadı! Çalışmaya devam et!")
            
    except Exception as e:
        st.error(f"Rozetler görüntülenirken bir hata oluştu: {str(e)}")
        
    finally:
        conn.close()

# Daily Tasks Functions
def generate_daily_tasks(user_id):
    conn = sqlite3.connect('motikoc.db')
    print("conn")
    # Get user's study history and preferences
    user_data = pd.read_sql_query('''
        SELECT u.grade, u.study_type, u.target_university,
               GROUP_CONCAT(DISTINCT s.subject) as recent_subjects,
               COALESCE(AVG(s.performance_rating), 0.0) as avg_performance
        FROM users u
        LEFT JOIN study_logs s ON u.id = s.user_id
        WHERE u.id = ?
        GROUP BY u.id
    ''', conn, params=(user_id,))
    print(user_data)

    if user_data.empty:
        conn.close()
        return []
    
    user_info = user_data.iloc[0]
    user_info['avg_performance'] = user_info['avg_performance'] if not pd.isnull(user_info['avg_performance']) else 0.0
    user_info['recent_subjects'] = user_info['recent_subjects'] if not pd.isnull(user_info['recent_subjects']) else "Veri Yok"
    
    
    # Create AI prompt for task generation
    prompt = f"""
    Bir YKS öğrencisi için günlük görevler oluştur. Lütfen aşağıdaki formatta JSON döndür:

    Önemli: Lütfen JSON yanıtının şu özelliklere sahip olmasına dikkat edin:
    1. Metin içeriği için tek tırnak kullanın
    2. JSON yapısı için çift tırnak kullanın
    3. Anahtar isimlerinde özel karakter kullanmayın
    4. Kesinlikle JSON formatına uyun
    5. Türkçe karakterleri düzgün kullanın
    6. Özel karakter içermemeli

    [
        {{
            "type": "study",
            "description": "görev açıklaması",
            "xp_reward": sayısal_değer
        }},
        ...
    ]
    
    Öğrenci Bilgileri:
    - Sınıf: {user_info['grade']}
    - Alan: {user_info['study_type']}
    - Hedef: {user_info['target_university']}
    - Son çalışılan dersler: {user_info['recent_subjects']}
    - Ortalama performans: {user_info['avg_performance']:.1f}/5
    
    4 farklı kategoride birer görev oluştur:
    1. Çalışma görevi (type: study)
    2. Sosyal görev (type: social)
    3. Motivasyon görevi (type: motivation)
    4. Alıştırma görevi (type: practice)
    
    Her görev için:
    - XP ödülü 50-200 arası olmalı
    - Görev açıklaması net ve anlaşılır olmalı
    """
    
    try:
        response = get_ai_response(prompt)
        print(f"AI Response: {response}")  # AI yanıtını logla
        tasks = clean_and_parse_json(response)
        
        if not tasks or not isinstance(tasks, list):
            raise ValueError("Görev listesi oluşturulamadı veya beklenen formatta değil.")
        
        # Insert tasks into database
        c = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        valid_tasks = []
        for task in tasks:
            task_type = task.get('type')
            description = task.get('description')
            xp_reward = task.get('xp_reward')
            
            # Zorunlu alanların kontrolü
            if not task_type or not description:
                print(f"Invalid task: {task}")
                continue  # Eksik alan varsa atla
            
            # XP Reward'ın doğrulanması
            if not isinstance(xp_reward, int) or not (50 <= xp_reward <= 200):
                print(f"Invalid or missing xp_reward for task: {task}")
                xp_reward = 50  # Varsayılan XP değeri
            
            valid_tasks.append({
                'type': task_type,
                'description': description,
                'xp_reward': xp_reward
            })
            
            c.execute('''
                INSERT INTO daily_tasks 
                (user_id, task_type, description, xp_reward, date_created)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, task_type, description, xp_reward, today))
        
        conn.commit()
        conn.close()
        return valid_tasks
        
    except Exception as e:
        st.error(f"Görev oluşturulurken hata: {str(e)}")
        print(f"Tam hata: {str(e)}")  # Debugging için
        try:
            print(f"AI Response: {response}")  # Debugging için
        except NameError:
            print("AI yanıtı alınamadı.")
        conn.close()
        return []
        
@st.cache_resource
def get_connection():
    return sqlite3.connect('motikoc.db', check_same_thread=False)

def safe_show_daily_tasks():
    try:
        st.markdown("""
        <div class="custom-card">
            <h3>📋 Günlük Görevler</h3>
        </div>
        """, unsafe_allow_html=True)
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Tek bir bağlantı kullanın
        conn = get_connection()
        c = conn.cursor()
        
        tasks = pd.read_sql_query('''
            SELECT * FROM daily_tasks
            WHERE user_id = ? AND date_created = ?
            ORDER BY completed ASC, task_type ASC
        ''', conn, params=(st.session_state.user_id, today))
        
        if tasks.empty:
            # Görev oluşturma işlemi
            new_tasks = generate_daily_tasks(st.session_state.user_id)
            if new_tasks:
                st.rerun()
            else:
                st.error("Görevler oluşturulamadı. Lütfen sayfayı yenileyin.")
                return
        
        for _, task in tasks.iterrows():
            try:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    icon = {
                        'study': '📚',
                        'social': '👥',
                        'motivation': '💪',
                        'practice': '✍️'
                    }.get(task['task_type'], '🎯')
                    
                    st.write(f"{icon} {task['description']}")
                
                with col2:
                    xp_reward = task['xp_reward'] if task['xp_reward'] is not None else 0
                    st.write(f"💫 {xp_reward} XP")
                
                with col3:
                    if not task['completed']:
                        if st.button("Tamamla", key=f"task_{task['id']}"):
                            # Aynı bağlantıyı kullanarak güncelleme yapın
                            c.execute('''
                                UPDATE daily_tasks 
                                SET completed = TRUE, date_completed = ?
                                WHERE id = ?
                            ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), task['id']))
                            conn.commit()
                            
                            # XP güncellemesi
                            update_user_xp(st.session_state.user_id, xp_reward)
                            
                            st.success("Görev tamamlandı! XP kazandın!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.write("✅ Tamamlandı")
            
            except Exception as e:
                print(f"Task display error: {str(e)}")
                continue
        
    except Exception as e:
        st.error("Görevler yüklenirken bir hata oluştu. Lütfen sayfayı yenileyin.")
        print(f"Daily tasks error: {str(e)}")

# Study Plan Generator
class StudyPlanGenerator:
    def __init__(self):
        self.subjects = {
            'TYT': [
                'Türkçe', 'Matematik', 'Fizik', 'Kimya', 'Biyoloji',
                'Tarih', 'Coğrafya', 'Felsefe', 'Din Kültürü'
            ],
            'AYT': {
                'Sayısal': ['Matematik', 'Fizik', 'Kimya', 'Biyoloji'],
                'Eşit Ağırlık': ['Matematik', 'Edebiyat', 'Tarih', 'Coğrafya'],
                'Sözel': ['Edebiyat', 'Tarih', 'Coğrafya', 'Felsefe'],
                'Dil': ['İngilizce']
            }
        }
    
    def generate_study_plan(self, user_id):
        conn = sqlite3.connect('motikoc.db')
        user_data = pd.read_sql_query('''
            SELECT grade, study_type, target_university, target_department,
                   target_rank
            FROM users WHERE id = ?
        ''', conn, params=(user_id,))
        
        # Get recent study history
        study_history = pd.read_sql_query('''
            SELECT subject, AVG(performance_rating) as avg_performance,
                   COUNT(*) as study_count
            FROM study_logs
            WHERE user_id = ? AND date >= date('now', '-30 days')
            GROUP BY subject
        ''', conn, params=(user_id,))
        
        conn.close()

        if user_data.empty:
            return None

        user_info = user_data.iloc[0]
        
        # AI Prompt for study plan
        prompt = f"""
        Bir YKS öğrencisi için detaylı haftalık çalışma planı oluştur.
        Lütfen aşağıdaki JSON formatında yanıt ver:
        Önemli: Lütfen JSON yanıtının şu özelliklere sahip olmasına dikkat edin:
        1. Metin içeriği için tek tırnak kullanın
        2. JSON yapısı için çift tırnak kullanın
        3. Anahtar isimlerinde özel karakter kullanmayın
        4. Kesinlikle JSON formatına uyun
        5. Türkçe karakterleri düzgün kullanın
        6. Özel karakter içermemeli
        {{
            "weekly_schedule": [
                {{
                    "day": "Pazartesi",
                    "sessions": [
                        {{
                            "time": "14:00-15:30",
                            "subject": "ders_adı",
                            "topic": "konu",
                            "duration": 90,
                            "priority": 1-3,
                            "notes": "öneriler"
                        }}
                    ]
                }}
            ],
            "focus_areas": [
                {{
                    "subject": "ders_adı",
                    "topics": ["öncelikli_konular"],
                    "weekly_hours": saat
                }}
            ],
            "recommendations": [
                "özel_öneriler"
            ]
        }}

        Öğrenci Bilgileri:
        - Sınıf: {user_info['grade']}
        - Alan: {user_info['study_type']}
        - Hedef: {user_info['target_university']} - {user_info['target_department']}
        - Hedef Sıralama: {user_info['target_rank']}
        - Geçmiş Çalışma Verileri:
        {study_history.to_json(orient='records') if not study_history.empty else 'Veri yok'}

        Lütfen şunlara dikkat et:
        1. YKS konularının güncel müfredata uygun olması
        2. TYT-AYT dengesinin gözetilmesi
        3. Düzenli tekrar ve pratik süreleri
        4. Öğrencinin hedefine uygun süre dağılımı
        5. Her gün için makul çalışma süreleri (max 8 saat)
        6. Dinlenme araları ve sosyal aktivite zamanı
        """
        
        try:
            response = get_ai_response(prompt)
            plan = clean_and_parse_json(response)
            
            if not plan:
                raise ValueError("Plan oluşturulamadı")
            
            return plan
            
        except Exception as e:
            print(f"Plan oluşturma hatası: {str(e)}")
            return None

# Chatbot Class
class MotiKocChatbot:
    def __init__(self):
        self.context = []
        self.max_context = 5  # Son 5 mesajı hatırla
    
    def add_message(self, role, content):
        self.context.append({"role": role, "content": content})
        if len(self.context) > self.max_context:
            self.context.pop(0)
    
    def generate_response(self, user_message, user_id):
        # Kullanıcı bilgilerini al
        conn = sqlite3.connect('motikoc.db')
        user_data = pd.read_sql_query('''
            SELECT grade, study_type, target_university, target_department,
                   target_rank
            FROM users WHERE id = ?
        ''', conn, params=(user_id,))
        conn.close()

        if user_data.empty:
            return "Kullanıcı bilgileri bulunamadı."

        user_info = user_data.iloc[0]
        
        # Chatbot prompt
        system_prompt = f"""
        Sen MotiKoç'sun - YKS'ye hazırlanan öğrenciler için bir yapay zeka eğitim koçu.
        
        Öğrenci Bilgileri:
        - Sınıf: {user_info['grade']}
        - Alan: {user_info['study_type']}
        - Hedef: {user_info['target_university']} - {user_info['target_department']}
        
        Özellikler:
        1. Samimi ve motive edici bir dil kullan
        2. YKS müfredatına ve sınav sistemine hakimsin
        3. Öğrencinin hedeflerine uygun öneriler sun
        4. Stres yönetimi ve motivasyon konularında destek ol
        5. Net ve uygulanabilir tavsiyeler ver
        
        Yanıt verirken:
        - Kısa ve öz cevaplar ver
        - Gerektiğinde örnekler kullan
        - Motivasyonu yüksek tut
        - Türkiye'deki eğitim sistemine uygun konuş
        """

        conversation_prompt = f"{system_prompt}\n\nGeçmiş Mesajlar:\n"
        for msg in self.context:
            conversation_prompt += f"{msg['role']}: {msg['content']}\n"
        
        conversation_prompt += f"\nÖğrenci: {user_message}\n\nYanıtın:"
    
        try:
            response = get_ai_response(conversation_prompt)
            self.add_message("user", user_message)
            self.add_message("assistant", response)
            return response
        except Exception as e:
            print(f"Chatbot hatası: {str(e)}")
            return "Üzgünüm, bir hata oluştu. Lütfen tekrar dener misin?"

# Study Groups Functions
def create_study_group():
    st.markdown("""
    <div class="custom-card">
        <h3>📚 Çalışma Grubu Oluştur</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("study_group_form"):
        group_name = st.text_input("Grup Adı")
        group_type = st.selectbox(
            "Çalışma Alanı",
            ["Sayısal", "Eşit Ağırlık", "Sözel", "Dil"]
        )
        max_members = st.number_input("Maksimum Üye Sayısı", 2, 10, 5)
        description = st.text_area("Grup Açıklaması")
        
        subjects = st.multiselect(
            "Odak Dersler",
            ["TYT Matematik", "TYT Türkçe", "AYT Matematik", "AYT Fizik", 
             "AYT Kimya", "AYT Biyoloji", "AYT Edebiyat"]
        )
        
        submit = st.form_submit_button("Grup Oluştur")
        
        if submit:
            if group_name and group_type and description and subjects:
                conn = sqlite3.connect('motikoc.db')
                c = conn.cursor()
                
                c.execute('''
                    INSERT INTO study_groups 
                    (creator_id, name, group_type, max_members, description, subjects)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (st.session_state.user_id, group_name, group_type,
                      max_members, description, ','.join(subjects)))
                
                group_id = c.lastrowid
                
                # Add creator as first member
                c.execute('''
                    INSERT INTO group_members (group_id, user_id, role)
                    VALUES (?, ?, 'admin')
                ''', (group_id, st.session_state.user_id))
                
                conn.commit()
                conn.close()
                
                st.success("Çalışma grubu oluşturuldu!")
            else:
                st.error("Lütfen tüm alanları doldurun!")

def show_study_groups():
    st.markdown("""
    <div class="custom-card">
        <h3>📚 Çalışma Grupları</h3>
    </div>
    """, unsafe_allow_html=True)
    
    create_study_group()
    
    st.markdown("### 🌟 Katıldığınız Gruplar")
    
    conn = sqlite3.connect('motikoc.db')
    groups = pd.read_sql_query('''
        SELECT sg.id, sg.name, sg.group_type, sg.description, sg.subjects, gm.role
        FROM study_groups sg
        JOIN group_members gm ON sg.id = gm.group_id
        WHERE gm.user_id = ?
    ''', conn, params=(st.session_state.user_id,))
    conn.close()
    
    if not groups.empty:
        for _, group in groups.iterrows():
            st.markdown(f"""
            <div class="custom-card" style="margin: 5px 0;">
                <h4>📖 {group['name']} ({group['group_type']})</h4>
                <p>{group['description']}</p>
                <p><strong>Odak Dersler:</strong> {group['subjects']}</p>
                <p><strong>Rolünüz:</strong> {group['role']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("Henüz katıldığınız bir çalışma grubu yok.")

# YKS Social Features
def show_yks_social_features():
    st.title("🤝 YKS Sosyal Özellikler")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Sıralama", "Çalışma Grupları", "Başarı Paylaşımları", "Arkadaş Önerileri"
    ])
    
    with tab1:
        show_yks_rankings()
    
    with tab2:
        show_study_groups()
    
    with tab3:
        show_achievements()
    
    with tab4:
        show_friend_suggestions()

def show_yks_rankings():
    st.markdown("""
    <div class="custom-card">
        <h3>📊 YKS Hazırlık Sıralaması</h3>
    </div>
    """, unsafe_allow_html=True)
    
    ranking_type = st.selectbox(
        "Sıralama Türü",
        ["Genel", "TYT Net", "AYT Net", "Çalışma Süresi"]
    )
    
    conn = sqlite3.connect('motikoc.db')
    
    if ranking_type == "Genel":
        rankings = pd.read_sql_query('''
            SELECT u.username, ul.current_level, ul.total_xp,
                   COUNT(DISTINCT s.id) as study_sessions,
                   ROUND(AVG(s.performance_rating), 2) as avg_performance
            FROM users u
            LEFT JOIN user_levels ul ON u.id = ul.user_id
            LEFT JOIN study_logs s ON u.id = s.user_id
            GROUP BY u.id, u.username
            ORDER BY ul.total_xp DESC
            LIMIT 100
        ''', conn)

        if not rankings.empty:
            for i, row in enumerate(rankings.itertuples(index=False), start=1):
                st.markdown(f"""
                <div class="custom-card" style="margin: 5px 0;">
                    <h4>#{i} {row.username}</h4>
                    <p>Seviye: {row.current_level} | XP: {row.total_xp:,}</p>
                    <p>Çalışma: {row.study_sessions} seans | 
                       Performans: {row.avg_performance}/5</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif ranking_type in ["TYT Net", "AYT Net"]:
        # Deneme sınavı sonuçlarına göre sıralama
        exam_type = "TYT" if ranking_type == "TYT Net" else "AYT"
        rankings = pd.read_sql_query('''
            SELECT u.username, 
                   MAX(e.total_net) as best_net,
                   COUNT(e.id) as exam_count,
                   ROUND(AVG(e.total_net), 2) as avg_net
            FROM users u
            JOIN mock_exams e ON u.id = e.user_id
            WHERE e.exam_type = ?
            GROUP BY u.id
            ORDER BY best_net DESC
            LIMIT 100
        ''', conn, params=(exam_type,))

        if not rankings.empty:
            for i, row in enumerate(rankings.itertuples(index=False), start=1):
                st.markdown(f"""
                <div class="custom-card" style="margin: 5px 0;">
                    <h4>#{i} {row.username}</h4>
                    <p>En İyi Net: {row.best_net}</p>
                    <p>Ortalama: {row.avg_net} | 
                       Deneme Sayısı: {row.exam_count}</p>
                </div>
                """, unsafe_allow_html=True)
    
    else:  # Çalışma Süresi
        rankings = pd.read_sql_query('''
            SELECT u.username,
                   SUM(s.duration) as total_duration,
                   COUNT(DISTINCT DATE(s.date)) as study_days,
                   ROUND(AVG(s.performance_rating), 2) as avg_performance
            FROM users u
            JOIN study_logs s ON u.id = s.user_id
            GROUP BY u.id
            ORDER BY total_duration DESC
            LIMIT 100
        ''', conn)
        
        if not rankings.empty:
            for i, row in enumerate(rankings.itertuples(index=False), start=1):
                hours = cast(float, row.total_duration)
                st.markdown(f"""
                <div class="custom-card" style="margin: 5px 0;">
                    <h4>#{i} {row.username}</h4>
                    <p>Toplam: {hours:.1f} saat</p>
                    <p>Çalışma Günü: {row.study_days} | 
                       Performans: {row.avg_performance}/5</p>
                </div>
                """, unsafe_allow_html=True)
    
    conn.close()



def show_friend_suggestions():
    st.markdown("""
    <div class="custom-card">
        <h3>👥 Arkadaş Önerileri</h3>
    </div>
    """, unsafe_allow_html=True)
    
    conn = sqlite3.connect('motikoc.db')
    
    # Get current user's info
    user_data = pd.read_sql_query('''
        SELECT grade, study_type, target_university, city
        FROM users WHERE id = ?
    ''', conn, params=(st.session_state.user_id,))
    
    if not user_data.empty:
        user_info = user_data.iloc[0]
        
        # Find similar users
        suggestions = pd.read_sql_query('''
            SELECT u.id, u.username, u.grade, u.study_type,
                   u.target_university, u.city
            FROM users u
            LEFT JOIN friendships f ON 
                (f.user_id = ? AND f.friend_id = u.id) OR
                (f.friend_id = ? AND f.user_id = u.id)
            WHERE u.id != ? AND f.id IS NULL
            AND (u.grade = ? OR u.study_type = ? OR 
                 u.target_university = ? OR u.city = ?)
            LIMIT 10
        ''', conn, params=(st.session_state.user_id, st.session_state.user_id,
                          st.session_state.user_id, user_info['grade'],
                          user_info['study_type'], user_info['target_university'],
                          user_info['city']))
        
        if not suggestions.empty:
            for _, suggestion in suggestions.iterrows():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="custom-card" style="margin: 5px 0;">
                        <h4>{suggestion['username']}</h4>
                        <p>{suggestion['grade']} | {suggestion['study_type']}</p>
                        <p>Hedef: {suggestion['target_university']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("Arkadaş Ekle", key=f"add_{suggestion['id']}"):
                        c = conn.cursor()
                        c.execute('''
                            INSERT INTO friendships (user_id, friend_id, status)
                            VALUES (?, ?, 'pending')
                        ''', (st.session_state.user_id, suggestion['id']))
                        conn.commit()
                        st.success("İstek gönderildi!")
        else:
            st.info("Öneri bulunamadı. Daha fazla arkadaş edinmek için daha fazla çalışın!")
    
    conn.close()

def show_achievements():
    st.markdown("""
    <div class="custom-card">
        <h3>🏆 YKS Başarıları</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Yeni başarı paylaşımı
    with st.form("new_achievement"):
        achievement_type = st.selectbox(
            "Başarı Türü",
            ["Deneme Sınavı", "Konu Bitirme", "Hedef Gerçekleştirme", "Diğer"]
        )
        
        description = st.text_area("Açıklama")
        image = st.file_uploader("Görsel (isteğe bağlı)", type=['png', 'jpg', 'jpeg'])
        
        submit = st.form_submit_button("Paylaş")
        
        if submit and description:
            conn = sqlite3.connect('motikoc.db')
            c = conn.cursor()
            
            c.execute('''
                INSERT INTO achievements 
                (user_id, title, description, date)
                VALUES (?, ?, ?, ?)
            ''', (st.session_state.user_id, achievement_type, description,
                  datetime.now().strftime('%Y-%m-%d')))
            
            conn.commit()
            conn.close()
            
            # Award XP for sharing achievement
            update_user_xp(st.session_state.user_id, calculate_xp_for_activity('achievement_shared'))
            
            st.success("Başarın paylaşıldı!")
            st.rerun()
    
    # Başarı akışı
    conn = sqlite3.connect('motikoc.db')
    achievements = pd.read_sql_query('''
        SELECT a.*, u.username
        FROM achievements a
        JOIN users u ON a.user_id = u.id
        ORDER BY a.date DESC
        LIMIT 50
    ''', conn)
    conn.close()
    
    if not achievements.empty:
        for _, achievement in achievements.iterrows():
            st.markdown(f"""
            <div class="custom-card" style="margin: 10px 0;">
                <h4>{achievement['username']}</h4>
                <p>{achievement['title']}</p>
                <p>{achievement['description']}</p>
                <small>{achievement['date']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("Henüz başarı paylaşımınız yok.")

class YKSHelper:
    def __init__(self):
        self.tyt_subjects = {
            'Türkçe': ['Paragraf', 'Dil Bilgisi', 'Anlatım Bozukluğu'],
            'Matematik': ['Temel Kavramlar', 'Sayılar', 'Problemler'],
            'Fen': ['Fizik', 'Kimya', 'Biyoloji'],
            'Sosyal': ['Tarih', 'Coğrafya', 'Felsefe', 'Din Kültürü']
        }
        
        self.ayt_subjects = {
            'Sayısal': {
                'Matematik': ['Fonksiyonlar', 'Türev', 'İntegral'],
                'Fizik': ['Kuvvet', 'Elektrik', 'Manyetizma'],
                'Kimya': ['Organik', 'Asit-Baz', 'Reaksiyonlar'],
                'Biyoloji': ['Hücre', 'Genetik', 'Sistemler']
            },
            'Eşit Ağırlık': {
                'Matematik': ['Fonksiyonlar', 'Türev', 'İntegral'],
                'Edebiyat': ['Şiir', 'Düz Yazı', 'Edebi Akımlar'],
                'Tarih': ['Osmanlı', 'Cumhuriyet', 'Çağdaş Tarih'],
                'Coğrafya': ['Türkiye', 'Dünya', 'Beşeri']
            },
            'Sözel': {
                'Edebiyat': ['Şiir', 'Düz Yazı', 'Edebi Akımlar'],
                'Tarih': ['Osmanlı', 'Cumhuriyet', 'Çağdaş Tarih'],
                'Coğrafya': ['Türkiye', 'Dünya', 'Beşeri'],
                'Felsefe': ['Antik', 'Ortaçağ', 'Modern']
            },
            'Dil': {
                'İngilizce': ['Grammar', 'Vocabulary', 'Reading Comprehension']
            }
        }


# YKSAIHelper Class
class YKSAIHelper:
    def __init__(self):
        self.yks_helper = YKSHelper()
    
    def generate_study_recommendations(self, user_id):
        conn = sqlite3.connect('motikoc.db')
        
        # Kullanıcı verilerini al
        user_data = pd.read_sql_query('''
            SELECT u.*, 
                   COUNT(s.id) as total_sessions,
                   COALESCE(AVG(s.performance_rating), 0.0) as avg_performance,
                   GROUP_CONCAT(DISTINCT s.subject) as studied_subjects
            FROM users u
            LEFT JOIN study_logs s ON u.id = s.user_id
            WHERE u.id = ?
            GROUP BY u.id
        ''', conn, params=(user_id,))

        print("girdiii")

        
        if user_data.empty:
            conn.close()
            return None
        
        user_info = user_data.iloc[0]
        print("ai destekli eğitim botu")
        prompt = f"""
        YKS'ye hazırlanan bir öğrenci için kişiselleştirilmiş öneriler oluştur.
        Lütfen aşağıdaki JSON formatında yanıt ver:
        Önemli: Lütfen JSON yanıtının şu özelliklere sahip olmasına dikkat edin:
        1. Metin içeriği için tek tırnak kullanın
        2. JSON yapısı için çift tırnak kullanın
        3. Anahtar isimlerinde özel karakter kullanmayın
        4. Kesinlikle JSON formatına uyun
        5. Özel karekter içermemeli.
        {{
            "öncelikli_konular": [
                {{
                    "ders": "ders_adı",
                    "konular": ["konu1", "konu2"],
                    "önem_derecesi": 1-5,
                    "tavsiyeler": ["öneri1", "öneri2"]
                }}
            ],
            "çalışma_önerileri": [
                {{
                    "başlık": "öneri_başlığı",
                    "açıklama": "detaylı_açıklama",
                    "uygulama_adımları": ["adım1", "adım2"]
                }}
            ],
            "motivasyon_tavsiyeleri": [
                "tavsiye1",
                "tavsiye2"
            ],
            "sınav_stratejileri": [
                {{
                    "strateji": "strateji_adı",
                    "uygulama": "uygulama_detayı"
                }}
            ]
        }}
        
        Öğrenci Profili:
        - Sınıf: {user_info['grade']}
        - Alan: {user_info['study_type']}
        - Hedef: {user_info['target_university']} - {user_info['target_department']}
        - Toplam Çalışma: {user_info['total_sessions']} seans
        - Ortalama Performans: {user_info['avg_performance']:.2f}/5
        - Çalışılan Dersler: {user_info['studied_subjects']}
        
        Öneriler için dikkat edilecek noktalar:
        1. TYT-AYT dengesi gözetilmeli
        2. Türk eğitim sistemine uygun olmalı
        3. Gerçekçi ve uygulanabilir olmalı
        4. Öğrencinin hedef bölümüne uygun olmalı
        5. YKS'nin güncel formatına uygun olmalı
        """
        
        try:
            response = get_ai_response(prompt)
            recommendations = clean_and_parse_json(response)
            print("recomandationss",recommendations)
            
            if not recommendations:
                raise ValueError("Öneriler oluşturulamadı")
            
            return recommendations
            
        except Exception as e:
            print(f"Öneri oluşturma hatası: {str(e)}")
            return None

    def analyze_mock_exam(self, user_id, exam_data):
        """Deneme sınavı analizi ve öneriler"""
        prompt = f"""
        YKS deneme sınavı sonuçlarını analiz et ve öneriler sun.
        Lütfen aşağıdaki JSON formatında yanıt ver: dönen JSON düzgün olmalı özel karekter içermemeli.
        {{
            "genel_analiz": {{
                "güçlü_yönler": ["madde1", "madde2"],
                "gelişim_alanları": ["madde1", "madde2"],
                "sıralama_tahmini": "tahmin_detayı"
            }},
            "ders_analizleri": [
                {{
                    "ders": "ders_adı",
                    "doğru": sayı,
                    "yanlış": sayı,
                    "net": sayı,
                    "öneriler": ["öneri1", "öneri2"],
                    "öncelikli_konular": ["konu1", "konu2"]
                }}
            ],
            "zaman_yönetimi": {{
                "problem_alanları": ["alan1", "alan2"],
                "öneriler": ["öneri1", "öneri2"]
            }},
            "gelecek_hedefler": [
                {{
                    "hedef": "hedef_açıklaması",
                    "süre": "hedef_süresi",
                    "strateji": "uygulama_stratejisi"
                }}
            ]
        }}

        Sınav Verileri:
        {json.dumps(exam_data, indent=2, ensure_ascii=False)}
        """
        
        try:
            response = get_ai_response(prompt)
            print(f"AI Response: {response}")  # Loglama için
            analysis = clean_and_parse_json(response)
            print("analysis",analysis)
            if not analysis:
                raise ValueError("Analiz oluşturulamadı")
            
            return analysis
            
        except Exception as e:
            print(f"Deneme analizi hatası: {str(e)}")
            print(f"AI Response: {response}")  # Hatanın kaynağını görmek için
            return None

    def generate_question_solutions(self, question_text, subject):
        """Soru çözüm ve açıklama üretici"""
        prompt = f"""
        YKS sorusunun çözümünü ve detaylı açıklamasını yap.
        Lütfen aşağıdaki JSON formatında yanıt ver:
        {{
            "soru_analizi": {{
                "konu": "ana_konu",
                "alt_konu": "alt_konu",
                "zorluk": 1-5,
                "sık_görülme": "sık/orta/nadir"
            }},
            "çözüm_adımları": [
                {{
                    "adım": 1,
                    "açıklama": "adım_açıklaması",
                    "formül": "kullanılan_formül",
                    "ipucu": "yararlı_ipucu"
                }}
            ],
            "benzer_soru_türleri": [
                "tür1",
                "tür2"
            ],
            "dikkat_edilecekler": [
                "nokta1",
                "nokta2"
            ]
        }}

        Soru: {question_text}
        Ders: {subject}
        """
        
        try:
            response = get_ai_response(prompt)
            solution = clean_and_parse_json(response)
            
            if not solution:
                raise ValueError("Çözüm oluşturulamadı")
            
            return solution
            
        except Exception as e:
            print(f"Soru çözümü hatası: {str(e)}")
            return None

from datetime import datetime, date
from typing import Union, cast

def get_valid_exam_date(label: str) -> date:
    selected_date = st.date_input(label, value=datetime.now())
    
    if isinstance(selected_date, tuple):
        st.warning("Lütfen geçerli bir tarih seçin.")
        st.stop()
    
    if isinstance(selected_date, datetime):
        return selected_date.date()
    elif isinstance(selected_date, date):
        return selected_date
    else:
        st.warning("Geçersiz tarih seçildi, varsayılan tarih kullanılıyor.")
        return datetime.now().date()

def show_yks_ai_features():
    st.title("🤖 YKS AI Asistanı")
    
    ai_helper = YKSAIHelper()
    
    tab1, tab2, tab3 = st.tabs([
        "Kişisel Öneriler",
        "Deneme Analizi",
        "Soru Çözüm Yardımcısı"
    ])
    
    with tab1:
        st.markdown("""
        <div class="custom-card">
            <h3>📚 Kişisel YKS Önerileri</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Önerileri Güncelle"):
            with st.spinner("Öneriler hazırlanıyor..."):
                recommendations = ai_helper.generate_study_recommendations(
                    st.session_state.user_id
                )
                print(recommendations)
                if recommendations:
                    st.session_state.yks_recommendations = recommendations
                    st.success("Öneriler hazır!")
                    st.rerun()  # st.rerun() kullanabilirsiniz
                else:
                    st.error("Öneriler oluşturulamadı. Lütfen tekrar deneyin veya destek alın.")
        
        if 'yks_recommendations' in st.session_state and st.session_state.yks_recommendations:
            recs = st.session_state.yks_recommendations
            
            # Öncelikli Konular
            st.markdown("### 📌 Öncelikli Konular")
            # `oncelikli_konular` yerine `öncelikli_konular` anahtarını kontrol et
            oncelikli_konular = recs.get('oncelikli_konular', []) or recs.get('öncelikli_konular', [])
            
            for konu in oncelikli_konular:
                with st.expander(f"📘 {konu.get('ders', 'Ders')}"):
                    # Önem Derecesi kontrolü
                    onem_derecesi = konu.get('onem_derecesi', 0) or konu.get('önem_derecesi', 0)
                    if isinstance(onem_derecesi, int) and onem_derecesi > 0:
                        st.write(f"**Önem Derecesi:** {'🌟' * onem_derecesi}")
                    else:
                        st.write("**Önem Derecesi:** Belirtilmemiş")
                    
                    # Konular kontrolü
                    konular = konu.get('konular', [])
                    if konular:
                        st.write("**Konular:**")
                        for k in konular:
                            st.write(f"- {k}")
                    else:
                        st.write("**Konular:** Belirtilmemiş")
                    
                    # Tavsiyeler kontrolü
                    tavsiyeler = konu.get('tavsiyeler', [])
                    if tavsiyeler:
                        st.write("**Tavsiyeler:**")
                        for t in tavsiyeler:
                            st.write(f"• {t}")
                    else:
                        st.write("**Tavsiyeler:** Belirtilmemiş")
            
            # Çalışma Önerileri
            st.markdown("### 💡 Çalışma Önerileri")
            # `calisma_onerileri` yerine `çalışma_önerileri` anahtarını kontrol et
            calisma_onerileri = recs.get('calisma_onerileri', []) or recs.get('çalışma_önerileri', [])
            
            for oneri in calisma_onerileri:
                with st.expander(f"🎯 {oneri.get('baslik', '') or oneri.get('başlık', 'Başlık')}"):
                    aciklama = oneri.get('aciklama', '') or oneri.get('açıklama', 'Açıklama bulunamadı.')
                    st.write(aciklama)
                    
                    # Uygulama Adımları kontrolü
                    uygulama_adimlari = oneri.get('uygulama_adimlari', []) or oneri.get('uygulama_adımları', [])
                    if uygulama_adimlari:
                        st.write("**Uygulama Adımları:**")
                        for i, adim in enumerate(uygulama_adimlari, 1):
                            st.write(f"{i}. {adim}")
                    else:
                        st.write("**Uygulama Adımları:** Belirtilmemiş")
            
            # Motivasyon Tavsiyeleri
            st.markdown("### 💪 Motivasyon Tavsiyeleri")
            for tavsiye in recs.get('motivasyon_tavsiyeleri', []):
                if tavsiye:
                    st.info(tavsiye)
                else:
                    st.write("• Motivasyon tavsiyesi bulunamadı.")
            
            # Sınav Stratejileri
            st.markdown("### 🎯 Sınav Stratejileri")
            # Her iki versiyonu da kontrol et
            sinav_stratejileri = recs.get('sinav_stratejileri', []) or recs.get('sınav_stratejileri', [])

            if sinav_stratejileri:
                for strateji in sinav_stratejileri:
                    # Her iki olası anahtar adını kontrol et
                    strateji_adi = (
                        strateji.get('strateji', None) or 
                        strateji.get('strategi', None) or 
                        'Strateji Adı'
                    )
                    uygulama_detayi = (
                        strateji.get('uygulama', None) or 
                        strateji.get('uygulama_detayi', None) or 
                        'Uygulama Detayı'
                    )
                    
                    if strateji_adi and uygulama_detayi:
                        # Özel bir card tasarımı kullanalım
                        st.markdown(f"""
                        <div style='
                            padding: 15px;
                            border-radius: 10px;
                            background-color: rgba(0, 255, 0, 0.1);
                            margin-bottom: 10px;
                        '>
                            <h4 style='color: #1a73e8;'>{strateji_adi}</h4>
                            <p>{uygulama_detayi}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("• Sınav stratejisi bilgisi eksik.")
            else:
                st.info("Henüz sınav stratejisi önerisi bulunmuyor.")

# Debug için JSON çıktısını göster (geliştirme aşamasında kullanılabilir)
# st.write("Debug - Raw Data:", recs)
        else:
            st.info("Henüz kişisel önerileriniz yok. Önerileri güncellemek için butona tıklayın.")

    with tab2:
        st.markdown("""
        <div class="custom-card">
            <h3>📊 Deneme Sınavı Analizi</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Deneme sonuçları girişi
        with st.form("mock_exam_form"):
            exam_type = st.selectbox("Sınav Türü", ["TYT", "AYT"])
            
            if exam_type == "TYT":
                col1, col2 = st.columns(2)
                with col1:
                    turkce_d = st.number_input("Türkçe Doğru", 0, 40, 0, key="turkce_d")
                    turkce_y = st.number_input("Türkçe Yanlış", 0, 40, 0, key="turkce_y")
                    mat_d = st.number_input("Matematik Doğru", 0, 40, 0, key="mat_d")
                    mat_y = st.number_input("Matematik Yanlış", 0, 40, 0, key="mat_y")
                
                with col2:
                    fen_d = st.number_input("Fen Doğru", 0, 20, 0, key="fen_d")
                    fen_y = st.number_input("Fen Yanlış", 0, 20, 0, key="fen_y")
                    sosyal_d = st.number_input("Sosyal Doğru", 0, 20, 0, key="sosyal_d")
                    sosyal_y = st.number_input("Sosyal Yanlış", 0, 20, 0, key="sosyal_y")
            
            else:  # AYT
                col1, col2 = st.columns(2)
                with col1:
                    mat_d = st.number_input("Matematik Doğru", 0, 40, 0, key="ayt_mat_d")
                    mat_y = st.number_input("Matematik Yanlış", 0, 40, 0, key="ayt_mat_y")
                    fizik_d = st.number_input("Fizik Doğru", 0, 14, 0, key="fizik_d")
                    fizik_y = st.number_input("Fizik Yanlış", 0, 14, 0, key="fizik_y")
                
                with col2:
                    kimya_d = st.number_input("Kimya Doğru", 0, 13, 0, key="kimya_d")
                    kimya_y = st.number_input("Kimya Yanlış", 0, 13, 0, key="kimya_y")
                    bio_d = st.number_input("Biyoloji Doğru", 0, 13, 0, key="bio_d")
                    bio_y = st.number_input("Biyoloji Yanlış", 0, 13, 0, key="bio_y")
            
            # Yardımcı fonksiyon ile tarih alımı
            exam_date: date = get_valid_exam_date("Sınav Tarihi")
            date_str: str = exam_date.strftime("%Y-%m-%d")
            
            total_time = st.number_input("Toplam Süre (dakika)", 30, 180, 120, key="total_time")
            
            submit = st.form_submit_button("Analiz Et")
            
            if submit:
                # Sınav verilerini hazırla
                exam_data = {
                    "type": exam_type,
                    "date": date_str,
                    "total_time": total_time,
                    "subjects": {}
                }
                
                if exam_type == "TYT":
                    exam_data["subjects"] = {
                        "Türkçe": {"doğru": turkce_d, "yanlış": turkce_y},
                        "Matematik": {"doğru": mat_d, "yanlış": mat_y},
                        "Fen": {"doğru": fen_d, "yanlış": fen_y},
                        "Sosyal": {"doğru": sosyal_d, "yanlış": sosyal_y}
                    }
                else:
                    exam_data["subjects"] = {
                        "Matematik": {"doğru": mat_d, "yanlış": mat_y},
                        "Fizik": {"doğru": fizik_d, "yanlış": fizik_y},
                        "Kimya": {"doğru": kimya_d, "yanlış": kimya_y},
                        "Biyoloji": {"doğru": bio_d, "yanlış": bio_y}
                    }
                
                with st.spinner("Analiz yapılıyor..."):
                    analysis = ai_helper.analyze_mock_exam(
                        st.session_state.user_id,
                        exam_data
                    )
                    
                    if analysis:
                        st.session_state.exam_analysis = analysis
                        st.success("Analiz tamamlandı!")
                        st.rerun()  # st.rerun() yerine st.rerun() kullanabilirsiniz
                    else:
                        st.error("Analiz oluşturulamadı. Lütfen tekrar deneyin veya destek alın.")
        
        # Analiz sonuçlarını göster
        if 'exam_analysis' in st.session_state and st.session_state.exam_analysis:
            analysis = st.session_state.exam_analysis
            
            # Genel Analiz kontrolü
            genel_analiz = analysis.get('genel_analiz', {})
            güçlü_yönler = genel_analiz.get('güçlü_yönler', [])
            gelişim_alanları = genel_analiz.get('gelişim_alanları', [])
            sıralama_tahmini = genel_analiz.get('sıralama_tahmini', 'Belirtilmemiş')
            
            st.markdown("### 📊 Genel Analiz")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Güçlü Yönler:**")
                if güçlü_yönler:
                    for güç in güçlü_yönler:
                        st.success(f"✓ {güç}")
                else:
                    st.write("• Güçlü yönler belirtilmemiş.")
            
            with col2:
                st.write("**Gelişim Alanları:**")
                if gelişim_alanları:
                    for alan in gelişim_alanları:
                        st.warning(f"⚠ {alan}")
                else:
                    st.write("• Gelişim alanları belirtilmemiş.")
            
            st.info(f"📈 Sıralama Tahmini: {sıralama_tahmini}")
            
            # Ders Analizleri kontrolü
            ders_analizleri = analysis.get('ders_analizleri', [])
            if ders_analizleri:
                st.markdown("### 📚 Ders Bazında Analiz")
                
                for ders in ders_analizleri:
                    ders_adı = ders.get('ders', 'Ders Adı')
                    doğru = ders.get('doğru', 0)
                    yanlış = ders.get('yanlış', 0)
                    net = ders.get('net', 0)
                    öneriler = ders.get('öneriler', [])
                    öncelikli_konular = ders.get('öncelikli_konular', [])
                    
                    with st.expander(f"📘 {ders_adı}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Doğru", doğru)
                        with col2:
                            st.metric("Yanlış", yanlış)
                        with col3:
                            st.metric("Net", net)
                        
                        st.write("**🎯 Öneriler:**")
                        if öneriler:
                            for öneri in öneriler:
                                st.write(f"• {öneri}")
                        else:
                            st.write("• Öneriler belirtilmemiş.")
                        
                        st.write("**📌 Öncelikli Konular:**")
                        if öncelikli_konular:
                            for konu in öncelikli_konular:
                                st.write(f"- {konu}")
                        else:
                            st.write("• Öncelikli konular belirtilmemiş.")
            else:
                st.warning("Ders bazında analiz bulunamadı.")
            
            # Zaman Yönetimi kontrolü
            zaman_yönetimi = analysis.get('zaman_yönetimi', {})
            problem_alanları = zaman_yönetimi.get('problem_alanları', [])
            öneriler = zaman_yönetimi.get('öneriler', [])
            
            st.markdown("### ⏰ Zaman Yönetimi")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Problem Alanları:**")
                if problem_alanları:
                    for problem in problem_alanları:
                        st.error(f"⚠ {problem}")
                else:
                    st.write("• Problem alanları belirtilmemiş.")
            
            with col2:
                st.write("**Öneriler:**")
                if öneriler:
                    for öneri in öneriler:
                        st.info(f"💡 {öneri}")
                else:
                    st.write("• Öneriler belirtilmemiş.")
            
            # Gelecek Hedefler kontrolü
            gelecek_hedefler = analysis.get('gelecek_hedefler', [])
            if gelecek_hedefler:
                st.markdown("### 🎯 Gelecek Hedefler")
                for hedef in gelecek_hedefler:
                    hedef_adı = hedef.get('hedef', 'Hedef Adı')
                    süre = hedef.get('süre', 'Belirtilmemiş')
                    strateji = hedef.get('strateji', 'Strateji belirtilmemiş.')
                    st.markdown(f"""
                    <div class="custom-card">
                        <h4>{hedef_adı}</h4>
                        <p>⏳ Süre: {süre}</p>
                        <p>📝 Strateji: {strateji}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Gelecek hedefler belirtilmemiş.")
        else:
            st.info("Henüz sınav analizi yapılmamış.")
    with tab3:
        st.markdown("""
        <div class="custom-card">
            <h3>🎯 Soru Çözüm Yardımcısı</h3>
            <p>YKS sorularının çözümünde size yardımcı olacak yapay zeka destekli çözüm asistanı.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Soru girişi formu
        with st.form("question_solver_form"):
            subject = st.selectbox(
                "Ders Seçin",
                ["Matematik", "Fizik", "Kimya", "Biyoloji", "Türkçe", "Tarih", "Coğrafya", "Felsefe"]
            )
            
            question_text = st.text_area(
                "Soruyu buraya yazın",
                height=150,
                placeholder="Çözmek istediğiniz soruyu buraya yazın..."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                difficulty = st.select_slider(
                    "Zorluk Seviyesi",
                    options=["Kolay", "Orta", "Zor"],
                    value="Orta"
                )
            
            with col2:
                exam_type = st.selectbox(
                    "Sınav Türü",
                    ["TYT", "AYT"]
                )
            
            submit = st.form_submit_button("Çözümü Göster")
            
            if submit and question_text:
                with st.spinner("Soru analiz ediliyor..."):
                    solution = ai_helper.generate_question_solutions(
                        question_text,
                        subject
                    )
                    
                    if solution:
                        st.session_state.current_solution = solution
                        st.success("Çözüm hazır!")
                        st.rerun()
                    else:
                        st.error("Çözüm oluşturulamadı. Lütfen tekrar deneyin.")
        
        # Çözüm sonuçlarını göster
        if 'current_solution' in st.session_state and st.session_state.current_solution:
            solution = st.session_state.current_solution
            
            # Soru Analizi
            st.markdown("### 📝 Soru Analizi")
            soru_analizi = solution.get('soru_analizi', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Konu", soru_analizi.get('konu', 'Belirtilmemiş'))
            with col2:
                st.metric("Alt Konu", soru_analizi.get('alt_konu', 'Belirtilmemiş'))
            with col3:
                zorluk = soru_analizi.get('zorluk', 0)
                st.metric("Zorluk", "⭐" * zorluk if isinstance(zorluk, int) else 'Belirtilmemiş')
            
            # Çözüm Adımları
            st.markdown("### 🔍 Çözüm Adımları")
            çözüm_adımları = solution.get('çözüm_adımları', [])
            if çözüm_adımları:
                for adım in çözüm_adımları:
                    with st.expander(f"Adım {adım.get('adım', '?')}"):
                        st.write(f"**Açıklama:** {adım.get('açıklama', 'Belirtilmemiş')}")
                        if adım.get('formül'):
                            st.latex(adım.get('formül'))
                        if adım.get('ipucu'):
                            st.info(f"💡 İpucu: {adım.get('ipucu')}")
            else:
                st.warning("Çözüm adımları bulunamadı.")
            
            # Benzer Soru Türleri
            st.markdown("### 📚 Benzer Soru Türleri")
            benzer_sorular = solution.get('benzer_soru_türleri', [])
            if benzer_sorular:
                for soru_türü in benzer_sorular:
                    st.write(f"• {soru_türü}")
            else:
                st.write("Benzer soru türü önerisi bulunamadı.")
            
            # Dikkat Edilecekler
            st.markdown("### ⚠️ Dikkat Edilecek Noktalar")
            dikkat_noktaları = solution.get('dikkat_edilecekler', [])
            if dikkat_noktaları:
                for nokta in dikkat_noktaları:
                    st.warning(nokta)
            else:
                st.write("Özel dikkat noktası belirtilmemiş.")
            
            # Çözümü Kaydet
            if st.button("Bu Çözümü Kaydet"):
                conn = sqlite3.connect('motikoc.db')
                c = conn.cursor()
                
                # Çözümleri kaydetmek için yeni bir tablo oluştur
                c.execute('''CREATE TABLE IF NOT EXISTS saved_solutions
                           (id INTEGER PRIMARY KEY,
                            user_id INTEGER,
                            subject TEXT,
                            question TEXT,
                            solution TEXT,
                            date TEXT,
                            FOREIGN KEY(user_id) REFERENCES users(id))''')
                
                # Çözümü JSON olarak kaydet
                c.execute('''INSERT INTO saved_solutions 
                           (user_id, subject, question, solution, date)
                           VALUES (?, ?, ?, ?, ?)''',
                        (st.session_state.user_id, subject, question_text,
                         json.dumps(solution), datetime.now().strftime('%Y-%m-%d')))
                
                conn.commit()
                conn.close()
                
                st.success("Çözüm kaydedildi! Daha sonra tekrar inceleyebilirsiniz.")
        
        # Kaydedilmiş çözümleri görüntüle
        st.markdown("### 📚 Kaydedilmiş Çözümleriniz")
        conn = sqlite3.connect('motikoc.db')
        saved_solutions = pd.read_sql_query('''
            SELECT * FROM saved_solutions
            WHERE user_id = ?
            ORDER BY date DESC
        ''', conn, params=(st.session_state.user_id,))
        conn.close()
        
        if not saved_solutions.empty:
            for _, solution in saved_solutions.iterrows():
                with st.expander(f"{solution['subject']} - {solution['date']}"):
                    st.write("**Soru:**")
                    st.write(solution['question'])
                    st.write("**Çözüm:**")
                    saved_solution = json.loads(solution['solution'])
                    st.json(saved_solution)
        else:
            st.info("Henüz kaydedilmiş çözümünüz bulunmuyor.")


def show_yks_dashboard():
    st.title("📊 YKS Hazırlık Paneli")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        show_yks_countdown()
        show_yks_ai_features()
    
    with col2:
        show_yks_resources()
        safe_show_daily_tasks()

def show_yks_countdown():
    """YKS geri sayım widget'ı"""
    yks_date = datetime(2025, 6, 15)  # YKS tarihi
    today = datetime.now()
    days_left = (yks_date - today).days
    
    st.markdown(f"""
    <div class="custom-card" style="text-align: center;">
        <h2>YKS'ye Kalan Süre</h2>
        <h1 style="color: {'red' if days_left < 30 else 'orange' if days_left < 90 else 'green'}">
            {days_left} GÜN
        </h1>
        <p>TYT: {yks_date.strftime('%d.%m.%Y')} Cumartesi</p>
        <p>AYT: {(yks_date + timedelta(days=1)).strftime('%d.%m.%Y')} Pazar</p>
    </div>
    """, unsafe_allow_html=True)

def show_yks_resources():
    """YKS kaynakları ve faydalı linkler"""
    st.markdown("""
    <div class="custom-card">
        <h3>📚 YKS Kaynakları</h3>
    </div>
    """, unsafe_allow_html=True)
    
    resource_types = {
        "Video Dersler": [
            "Khan Academy Türkçe",
            "TonguçAkademi",
            "Hocalara Geldik"
        ],
        "Soru Bankaları": [
            "345",
            "Aydın Yayınları",
            "Bilgi Sarmal"
        ],
        "Online Denemeler": [
            "Vitamin",
            "UdemyYKS",
            "Online Deneme"
        ]
    }
    
    for category, resources in resource_types.items():
        with st.expander(category):
            for resource in resources:
                st.write(f"• {resource}")

# YKS Specific Features
def add_yks_specific_features():
    """YKS'ye özel ek özellikler"""
    
    # Konular veritabanı tablosu
    def init_yks_tables():
        conn = sqlite3.connect('motikoc.db')
        c = conn.cursor()
        
        # YKS konuları zaten init_db() içinde oluşturuldu
        # Bu fonksiyon isterseniz ek tablolar için genişletebilirsiniz
        
        conn.commit()
        conn.close()
    
    return {
        'init_tables': init_yks_tables,
        'show_dashboard': show_yks_dashboard
    }

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import google.generativeai as genai
from dataclasses import dataclass ,field
import re
from enum import Enum, auto
import json
import os
import asyncio
import streamlit as st
import plotly.express as px
from datetime import datetime
import nest_asyncio
nest_asyncio.apply()

class IntentType(Enum):
    PROGRAM = auto()
    UNIVERSITY = auto()
    CITY = auto()
    SCORE_TYPE = auto()
    RANKING = auto()
    SCHOLARSHIP = auto()
    COMPARISON = auto()
    FACULTY = auto()
    FEES = auto()
    QUOTA = auto()

@dataclass
class FilterCriteria:
    min_ranking: Optional[int] = None
    max_ranking: Optional[int] = None
    universities: List[str] = field(default_factory=list)
    programs: List[str] = field(default_factory=list)
    cities: List[str] = field(default_factory=list)
    score_types: List[str] = field(default_factory=list)
    scholarship_percentage: Optional[int] = None
    faculty_types: List[str] = field(default_factory=list)
    university_types: List[str] = field(default_factory=list)
    max_fee: Optional[float] = None
    min_quota: Optional[int] = None
    language_types: List[str] = field(default_factory=list)
    min_score: Optional[float] = None  # Yeni özellik
    max_score: Optional[float] = None  # Yeni özellik
    
    def __post_init__(self):
        self.universities = self.universities or []
        self.programs = self.programs or []
        self.cities = self.cities or []
        self.score_types = self.score_types or []
        self.faculty_types = self.faculty_types or []
        self.university_types = self.university_types or []
        self.language_types = self.language_types or []

class UniversityRecommender:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro-002"):
        """Initialize the recommender with Gemini API key and model."""
        if api_key is None:
            api_key = API_KEY
            if api_key is None:
                raise ValueError("Gemini API key must be provided or set as an environment variable.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.system_prompt = """
    Sen bir üniversite öneri sistemi asistanısın. Kullanıcının verdiği soruyu analiz et ve kullanıcının tercih kriterlerini ayrıştırarak aşağıdaki bilgileri çıkar. 
    Kullanıcının ifadesini en yalın haliyle anla. Örneğin, "yazılımcılık" ifadesi geldiğinde bunu "yazılım mühendisi" olarak algıla.

    Özellikle "fakülte" ile ilgili tercihleri "fakulte_tercihleri" altında, spesifik program tercihlerini ise "tercih_edilen_programlar_bolumler" alanında 
    tanımla. 

    Kullanıcı "mühendislik", "hukuk", "tıp" gibi fakülte isimlerini belirttiğinde, bunu "fakulte_tercihleri" altında değerlendir. Ancak, 
    spesifik bir program adı verdiğinde (örneğin "bilgisayar mühendisliği"), bunu "tercih_edilen_programlar_bolumler" altında değerlendir. Bu ayrımı 
    sağlamak için aşağıdaki şablonu takip et.

    Çıkarılacak Bilgiler:
    - tercih_edilen_universiteler: Örneğin, "İstanbul Üniversitesi", "Boğaziçi Üniversitesi".
    - tercih_edilen_programlar_bolumler: Örneğin, "bilgisayar mühendisliği", "endüstri mühendisliği".
    - tercih_edilen_sehirler: Örneğin, "İstanbul", "Ankara".
    - taban_puan_araligi_veya_belirli_bir_puan_degeri: Puan değeri veya aralık (örneğin 450.0 veya 400.0-500.0).
    - basari_sirasi_araligi_veya_belirli_bir_siralama: Belirli bir başarı sırası veya aralık (örneğin 5000 veya 3000-10000).
    - burs_gereksinimleri: Yüzde olarak burs gereksinimi (örneğin %50 veya %100).
    - fakulte_tercihleri: Örneğin, "Mühendislik Fakültesi", "Hukuk Fakültesi".
    - universite_turu: Üniversite türü (örneğin, "devlet", "vakıf", "kıbrıs").
    - puan türü: SAY, EA, SÖZ, DİL, TYT gibi. Kullanıcı "sayısal" derse SAY, "eşit ağırlık" derse EA, "sözel" derse SÖZ, "dil" derse DİL olarak döndür. 
      2 yıllık veya TYT puan türü belirtilmişse, TYT olarak döndür.
    - maksimum_ucret: Kullanıcının belirttiği maksimum ücret (örneğin, 50000.0).
    - dil_tercihi: Program dili (örneğin, "İngilizce", "Fransızca", "Almanca").
    - minimum_kontenjan_gereksinimleri: En az kontenjan değeri (örneğin, 10).

    **Dikkat Edilmesi Gerekenler**:
    - Çıktıyı JSON formatında döndür ve tüm alanları ekle. Eğer bilgi verilmediyse, ilgili alanı boş string ("") olarak döndür.
    - JSON formatı şu şekilde olmalıdır:
      {
          "tercih_edilen_universiteler": [],
          "tercih_edilen_programlar_bolumler": [],
          "tercih_edilen_sehirler": [],
          "taban_puan_araligi_veya_belirli_bir_puan_degeri": "",
          "basari_sirasi_araligi_veya_belirli_bir_siralama": "",
          "burs_gereksinimleri": "",
          "fakulte_tercihleri": [],
          "universite_turu": "",
          "puan_turu": "",
          "maksimum_ucret": "",
          "dil_tercihi": [],
          "minimum_kontenjan_gereksinimleri": ""
      }

    **Örnek Kullanıcı Soruları**:
    - "İstanbul'da mühendislik fakültesinde %50 burslu bilgisayar mühendisliği istiyorum."
    - "Yazılımcılık bölümü istiyorum." (Bu durumda "yazılım mühendisliği" olarak algılanmalı.)
    - "Tıp fakültesi için başarı sıralamam 5000'den iyi olan programlar."
    - "Eşit ağırlık ile %100 burslu, 400-500 puan aralığında hukuk bölümleri."

    Kullanıcının sorusunu analiz ederek, bu bilgileri doğru JSON formatında ver.
"""



    def _analyze_intent(self, question: str) -> Dict[str, Any]:
        """Analyze user question using LLM to detect intents and filtering criteria."""
        try:
            prompt = f"{self.system_prompt}\n\nKullanıcı sorusu: {question}"
            response = self.model.generate_content(prompt)
            
            # Log the raw response for debugging
            response_text = response.text.strip() if response.text else "(No content)"
            print("Raw response:", response_text)
            
            # Clean the response by removing backticks and unnecessary formatting
            cleaned_response = response_text.replace("```json", "").replace("```", "").strip()
            print("Cleaned response:", cleaned_response)  # For debugging
            intent_data = json.loads(cleaned_response)
            return intent_data
                
        except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw response: {response_text}")
                return {}
        except Exception as e:
                print(f"Error in intent analysis: {e}")
                return {}
    def _create_filter_criteria(self, intent_data: Dict[str, Any]) -> FilterCriteria:
        """Convert LLM intent analysis into structured filter criteria."""
        criteria = FilterCriteria()
        
        criteria.programs = intent_data.get('tercih_edilen_programlar_bolumler', [])
        criteria.cities = intent_data.get('tercih_edilen_sehirler', [])
        criteria.score_types = [intent_data.get('puan_turu', '')] if intent_data.get('puan_turu', '') else []
        
        basari_sirasi = intent_data.get('basari_sirasi_araligi_veya_belirli_bir_siralama', '')

        # Eğer aralık varsa, min ve max olarak ayır
        if '-' in basari_sirasi:
            try:
                min_ranking, max_ranking = map(int, basari_sirasi.split('-'))
                criteria.min_ranking = min_ranking
                criteria.max_ranking = max_ranking
            except ValueError:
                print("Geçersiz aralık formatı:", basari_sirasi)
        elif basari_sirasi !='':
            # Tek bir sıralama değeri varsa onu hem min hem de max olarak al
            try:
                ranking = int(basari_sirasi)
                criteria.min_ranking = ranking
            except ValueError:
                print("Geçersiz sıralama formatı:", basari_sirasi)
            # Taban puan aralığı ekleme
        taban_puan = intent_data.get('taban_puan_araligi_veya_belirli_bir_puan_degeri', '')
        
        # Eğer puan aralığı bir dize ise "-" işaretine göre bölün
        if isinstance(taban_puan, str) and '-' in taban_puan:
            try:
                min_score, max_score = map(float, taban_puan.split('-'))
                criteria.min_score = min_score
                criteria.max_score = max_score
            except ValueError:
                print("Geçersiz puan aralığı formatı:", taban_puan)
        elif isinstance(taban_puan, (float, int)):  # Tek bir puan değeri varsa, bunu max_score olarak al
            criteria.max_score = float(taban_puan)


        criteria.scholarship_percentage = int(intent_data.get('burs_gereksinimleri', 0)) if intent_data.get('burs_gereksinimleri', '') else None
        criteria.faculty_types = intent_data.get('fakulte_tercihleri', [])
        criteria.university_types = [intent_data.get('universite_turu', '')] if intent_data.get('universite_turu', '') else []
        criteria.max_fee = float(intent_data.get('maksimum_ucret', 0)) if intent_data.get('maksimum_ucret', '') else None
        criteria.min_quota = int(intent_data.get('minimum_kontenjan_gereksinimleri', 0)) if intent_data.get('minimum_kontenjan_gereksinimleri', '') else None
        criteria.language_types = intent_data.get('dil_tercihi', [])

        return criteria

    def _apply_filters(self, df: pd.DataFrame, criteria: FilterCriteria) -> pd.DataFrame:
        """Apply filtering criteria to the dataframe."""
        filtered_df = df.copy()
        
        try:
            if criteria.min_ranking is not None:
                filtered_df = filtered_df[filtered_df['Başarı Sırası 2023'] >= criteria.min_ranking]
            if criteria.max_ranking is not None:
                filtered_df = filtered_df[filtered_df['Başarı Sırası 2023'] <= criteria.max_ranking]
            if criteria.universities:
                pattern = '|'.join(map(re.escape, criteria.universities))
                filtered_df = filtered_df[filtered_df['Üniversite'].str.contains(pattern, case=False, na=False)]

            if criteria.programs:
                # Listeyi bir dizeye dönüştürüyoruz
                pattern = '|'.join(map(re.escape, criteria.programs))  # program isimlerini ayırarak birleştir
                filtered_df = filtered_df[filtered_df['Program Adı'].str.contains(pattern, case=False, na=False)]


                # Sonrasında str.contains() ile regex'i devre dışı bırakarak doğrudan arama yapalım
                filtered_df = filtered_df[filtered_df['Program Adı'].str.contains(pattern, case=False, na=False)]
                print("after filtering:", len(filtered_df))


            if criteria.cities:
                pattern = '|'.join(map(re.escape, criteria.cities))
                filtered_df = filtered_df[filtered_df['Şehir'].str.contains(pattern, case=False, na=False)]

            if criteria.score_types:
                print("criteria.score_types : ",criteria.score_types)
                pattern = '|'.join(map(re.escape, criteria.score_types))
                print("criteria.score_types : ",criteria.score_types)
                print("before filtering : ",len(filtered_df))
                filtered_df = filtered_df[filtered_df['Puan Türü'].str.contains(pattern, case=False, na=False)]
                print("after filtering : ",len(filtered_df))

            if criteria.language_types:
                language_pattern = '|'.join(map(re.escape, criteria.language_types))
                filtered_df = filtered_df[filtered_df['Program Adı'].str.contains(language_pattern, case=False, na=False)]

            if criteria.scholarship_percentage is not None:
                scholarship_str = f"%{criteria.scholarship_percentage}"
                filtered_df = filtered_df[filtered_df['Program Adı'].str.contains(scholarship_str, case=False, na=False)]

            if criteria.faculty_types:
                pattern = '|'.join(map(re.escape, criteria.faculty_types))
                filtered_df = filtered_df[filtered_df['Fakülte'].str.contains(pattern, case=False, na=False)]

            if criteria.university_types:
                pattern = '|'.join(map(re.escape, criteria.university_types))
                filtered_df = filtered_df[filtered_df['Üni.Türü'].str.contains(pattern, case=False, na=False)]

            if criteria.max_fee is not None:
                filtered_df['Ücret (KDV Hariç)'] = pd.to_numeric(filtered_df['Ücret (KDV Hariç)'], errors='coerce').fillna(0)
                filtered_df = filtered_df[filtered_df['Ücret (KDV Hariç)'] <= criteria.max_fee]

            if criteria.min_quota is not None:
                filtered_df['Kontenjan 2023'] = pd.to_numeric(filtered_df['Kontenjan 2023'], errors='coerce').fillna(0)
                filtered_df = filtered_df[filtered_df['Kontenjan 2023'] >= criteria.min_quota]
                 # Taban Puan 2023 filtrelemesi
            if criteria.min_score is not None:
                filtered_df = filtered_df[filtered_df['Taban Puan 2023'] >= criteria.min_score]
            if criteria.max_score is not None:
                filtered_df = filtered_df[filtered_df['Taban Puan 2023'] <= criteria.max_score]

            filtered_df = filtered_df.sort_values('Başarı Sırası 2023', ascending=True)
            return filtered_df

        except Exception as e:
            print(f"Error in applying filters: {e}")
            return df

    def _generate_response(self, filtered_df: pd.DataFrame, question: str) -> str:
        """Generate a natural language response based on the filtered results."""
        try:
            if filtered_df.empty:
                return "Üzgünüm, arama kriterlerinize uygun bir program bulamadım."

            results_summary = filtered_df.head(20).to_dict('records')

            prompt = f"""
            Kullanıcının sorusu: {question}

            En iyi 5 sonuç:
            {json.dumps(results_summary, ensure_ascii=False, indent=2)}

            Lütfen bu sonuçları analiz ederek Türkçe olarak detaylı ve yardımcı bir cevap ver:
            1. Ana bulguları özetle.
            2. Önemli eğilimleri veya kalıpları vurgula.
            3. Spesifik önerilerde bulun.
            4. Önemli dikkat edilmesi gereken noktaları belirt.
            5. Öğrenciye sonraki adımlar için tavsiyeler ver.

            Cevabı doğal ve samimi bir üslupla yaz, ancak kısa ve odaklı tut.
            """

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            print(f"Error in generating response: {e}")
            return "Sonuçları değerlendirirken bir hata oluştu. Lütfen daha sonra tekrar deneyin."

    def process_question(self, question: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Process user question and return filtered results with explanation."""
        try:
            intent_data = self._analyze_intent(question)
            criteria = self._create_filter_criteria(intent_data)
            filtered_df = self._apply_filters(df, criteria)
            response = self._generate_response(filtered_df, question)
            return filtered_df, response
            
        except Exception as e:
            print(f"Error in processing question: {e}")
            return df, "İşlem sırasında bir hata oluştu. Lütfen daha sonra tekrar deneyin."

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import Dict
import os

class UniversityRecommenderInterface:
    def __init__(self):
        """Initialize the interface with necessary configurations and session state."""
        self._initialize_session_state()
        self.recommender = self._initialize_recommender()
        self.df = self._load_data()

    def _initialize_session_state(self):
        """Initialize session state variables."""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'filtered_results' not in st.session_state:
            st.session_state.filtered_results = None
        if 'current_response' not in st.session_state:
            st.session_state.current_response = None


    def _initialize_recommender(self) -> UniversityRecommender:
        """Initialize the UniversityRecommender with Gemini API key."""
        api_key = API_KEY
        
        return UniversityRecommender(api_key=api_key)
    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess the un aiversaity data."""
        try:
            df = pd.read_csv('programs_data_with_links.csv')
            
            # Clean and convert 'Ücret (KDV Hariç)' to a numeric format
            df['Ücret (KDV Hariç)'] = (
                df['Ücret (KDV Hariç)']
                .replace(r'[\D]', '', regex=True)  # Remove any non-digit characters
                .astype(float)                     # Convert to float
            )

            
            numeric_columns = [
                'Başarı Sırası 2023', 'Taban Puan 2023', 'Kontenjan 2023',
                'Yerleşen 2023'
            ]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        except Exception as e:
            st.error(f"Veri yüklenirken hata oluştu: {str(e)}")
            return pd.DataFrame()


    def _create_visualizations(self, df: pd.DataFrame):
        """Create interactive visualizations for the filtered results."""
        if df.empty:
            return

        col1, col2 = st.columns(2)

        with col1:
            kontenjan_values = df['Kontenjan 2023'].fillna(20)
            
            fig_scatter = px.scatter(
                df,
                x='Başarı Sırası 2023',
                y='Üniversite',
                color='Puan Türü',
                size=kontenjan_values,
                hover_data={
                    'Program Adı': True,
                    'Taban Puan 2023': ':.2f',
                    'Kontenjan 2023': ':.0f'
                },
                title='Üniversite ve Başarı Sırası Dağılımı'
            )
            fig_scatter.update_layout(
                height=600,
                xaxis_title="Başarı Sırası",
                yaxis_title="Üniversite",
                showlegend=True,
                legend_title="Puan Türü"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            uni_type_counts = df['Üni.Türü'].fillna('Belirtilmemiş').value_counts()
            
            fig_pie = px.pie(
                values=uni_type_counts.values,
                names=uni_type_counts.index,
                title='Üniversite Türü Dağılımı',
                hole=0.4
            )
            fig_pie.update_layout(
                height=400,
                showlegend=True,
                legend_title="Üniversite Türü"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        try:
            box_plot_data = df.dropna(subset=['Puan Türü', 'Taban Puan 2023', 'Üni.Türü'])
            
            if not box_plot_data.empty:
                fig_box = px.box(
                    box_plot_data,
                    x='Puan Türü',
                    y='Taban Puan 2023',
                    color='Üni.Türü',
                    title='Puan Türlerine Göre Taban Puan Dağılımı'
                )
                fig_box.update_layout(
                    height=500,
                    xaxis_title="Puan Türü",
                    yaxis_title="Taban Puan",
                    showlegend=True,
                    legend_title="Üniversite Türü"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning("Taban puan dağılımı için yeterli veri bulunmamaktadır.")
        except Exception as e:
            st.warning(f"Taban puan dağılımı oluşturulurken bir hata oluştu: {str(e)}")

        st.subheader("📊 Özet İstatistikler")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Toplam Program Sayısı", len(df))
        
        with col2:
            avg_ranking = df['Başarı Sırası 2023'].mean()
            st.metric(
                "Ortalama Başarı Sırası",
                f"{avg_ranking:,.0f}" if not pd.isna(avg_ranking) else "Veri yok"
            )
        
        with col3:
            avg_points = df['Taban Puan 2023'].mean()
            st.metric(
                "Ortalama Taban Puan",
                f"{avg_points:.2f}" if not pd.isna(avg_points) else "Veri yok"
            )

    def _display_results_table(self, df: pd.DataFrame):
        """Display the filtered results in an interactive table."""
        if df.empty:
            st.warning("Arama kriterlerinize uygun sonuç bulunamadı.")
            return

        display_columns = [
            'Üniversite', 'Program Adı', 'Şehir', 'Puan Türü',
            'Başarı Sırası 2023','Başarı Sırası 2022', 'Taban Puan 2023','Taban Puan 2022','Kontenjan 2023',
            'Üni.Türü','Yıl','Ücret (KDV Hariç)','YKS Net Ort. Link','University Link','YÖP Link'
        ]

        st.dataframe(
            df[display_columns].style.format({
                'Başarı Sırası 2023': '{:,.0f}',
                'Taban Puan 2023': '{:.2f}',
                'Ücret (KDV Hariç)': '{:,.2f} ₺'
            }),
            use_container_width=True,
            height=400
        )

    def _add_to_search_history(self, question: str, results_count: int):
        """Add search query to history with timestamp."""
        st.session_state.search_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'question': question,
            'results_count': results_count
        })

    def _show_search_history(self):
        """Display search history in the sidebar."""
        if st.session_state.search_history:
            st.sidebar.header("Arama Geçmişi")
            for idx, search in enumerate(reversed(st.session_state.search_history[-5:])):
                with st.sidebar.expander(
                    f"{search['timestamp']} ({search['results_count']} sonuç)",
                    expanded=False
                ):
                    st.write(search['question'])

    def _create_filters_sidebar(self):
        """Create sidebar filters for manual filtering."""
        st.sidebar.header("Filtreleme Seçenekleri")
        
        filters = {}
        
        filters['city'] = st.sidebar.multiselect(
            "Şehir Seçin",
            options=sorted(self.df['Şehir'].unique()),
            key="city_filter"
        )
        
        filters['uni_type'] = st.sidebar.multiselect(
            "Üniversite Türü",
            options=sorted(self.df['Üni.Türü'].unique()),
            key="uni_type_filter"
        )
        
        filters['score_type'] = st.sidebar.multiselect(
            "Puan Türü",
            options=sorted(self.df['Puan Türü'].unique()),
            key="score_type_filter"
        )
        
        min_ranking = int(self.df['Başarı Sırası 2023'].min())
        max_ranking = int(self.df['Başarı Sırası 2023'].max())
        filters['ranking_range'] = st.sidebar.slider(
            "Başarı Sırası Aralığı",
            min_value=min_ranking,
            max_value=max_ranking,
            value=(min_ranking, max_ranking),
            key="ranking_filter"
        )
        
        return filters

    def _apply_manual_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply manual filters selected from sidebar."""
        filtered_df = df.copy()
        
        if filters['city']:
            filtered_df = filtered_df[filtered_df['Şehir'].isin(filters['city'])]
            
        if filters['uni_type']:
            filtered_df = filtered_df[filtered_df['Üni.Türü'].isin(filters['uni_type'])]
            
        if filters['score_type']:
            filtered_df = filtered_df[filtered_df['Puan Türü'].isin(filters['score_type'])]
            
        filtered_df = filtered_df[
            (filtered_df['Başarı Sırası 2023'] >= filters['ranking_range'][0]) &
            (filtered_df['Başarı Sırası 2023'] <= filters['ranking_range'][1])
        ]
        
        return filtered_df

    def run(self):
        """Run the Streamlit interface."""
        st.title("🎓 YKS Tercih Asistanı")
        st.write("""
        Bu asistan, YKS tercihlerinizde size yardımcı olmak için tasarlanmıştır. 
        Sorunuzu doğal bir dille yazın, size en uygun programları bulalım.
        """)

        # Create sidebar filters
        filters = self._create_filters_sidebar()

        # Main search interface
        question = st.text_input(
            "Sorunuzu yazın:",
            placeholder="Örnek: İstanbul'da başarı sırası 50000'den iyi olan bilgisayar mühendisliği bölümlerini göster"
        )

        if st.button("Ara", type="primary"):
            if not question:
                st.warning("Lütfen bir soru girin.")
                return

            with st.spinner("Sonuçlar hazırlanıyor..."):
                filtered_df, response = self.recommender.process_question(question, self.df)
                
                # Apply manual filters
                filtered_df = self._apply_manual_filters(filtered_df, filters)
                
                # Store results in session state
                st.session_state.filtered_results = filtered_df
                st.session_state.current_response = response
                
                # Add to search history
                self._add_to_search_history(question, len(filtered_df))

        # Display results if available
        if st.session_state.filtered_results is not None:
            st.header("🔍 Sonuçlar")
            
            # Display AI response
            if st.session_state.current_response:
                with st.expander("AI Değerlendirmesi", expanded=True):
                    st.write(st.session_state.current_response)
            
            # Display results table
            self._display_results_table(st.session_state.filtered_results)
            
            # Create visualizations
            with st.expander("📊 Görselleştirmeler", expanded=True):
                self._create_visualizations(st.session_state.filtered_results)

        # Show search history
        self._show_search_history()

        # Footer
        st.markdown("""
        ---
        💡 **İpucu**: Daha iyi sonuçlar için, tercih kriterlerinizi detaylı bir şekilde belirtin.
        Örneğin: "İstanbul'da devlet üniversitelerinde, başarı sırası 50000'den iyi olan bilgisayar mühendisliği bölümlerini göster"
        """)
# CareerPathFinder İlgili Kodlar
import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import re
from datetime import datetime
import base64
from io import BytesIO

# Gemini API configuration
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-002')

# Error handling function
def handle_llm_error(error, error_type):
    """LLM hatalarını yönet"""
    error_messages = {
        "api_error": "API bağlantısında bir sorun oluştu. Lütfen daha sonra tekrar deneyin.",
        "parsing_error": "Yanıt işlenirken bir hata oluştu. Lütfen tekrar deneyin.",
        "timeout_error": "İstek zaman aşımına uğradı. Lütfen daha sonra tekrar deneyin.",
    }
    st.error(error_messages.get(error_type, "Beklenmeyen bir hata oluştu."))
    st.error(f"Teknik Detay: {str(error)}")



# Enhanced question sections for Turkish high school students
SECTIONS = [
    {
        "title": "Akademik İlgi ve Başarı Alanları",
        "icon": "📚",
        "questions": [
            {
                "id": "academic_interests",
                "question": "Okulda en çok hangi derslerde başarılısınız ve bu dersleri sevmenizin nedeni nedir?",
                "placeholder": "Örn: Matematik dersinde problem çözmeyi seviyorum çünkü...",
                "tip": "Sadece notlarınızı değil, derslerde size keyif veren yönleri de düşünün.",
                "required": True
            },
            {
                "id": "learning_style",
                "question": "Yeni bir konuyu öğrenirken hangi yöntem sizin için en etkili oluyor?",
                "placeholder": "Örn: Görsel materyaller, pratik uygulamalar, grup çalışmaları...",
                "tip": "Geçmişteki başarılı öğrenme deneyimlerinizi hatırlayın.",
                "required": True
            }
        ]
    },
    {
        "title": "Sosyal Beceriler ve İletişim",
        "icon": "🤝",
        "questions": [
            {
                "id": "teamwork",
                "question": "Grup projelerinde genellikle nasıl bir rol üstlenirsiniz?",
                "placeholder": "Örn: Liderlik yapmayı severim, organizasyon konusunda iyiyim...",
                "tip": "Okul projelerindeki ve sosyal aktivitelerdeki deneyimlerinizi düşünün.",
                "required": True
            },
            {
                "id": "communication",
                "question": "İnsanlarla iletişim kurarken kendinizi nasıl hissediyorsunuz?",
                "placeholder": "Örn: Yeni insanlarla tanışmaktan keyif alırım, küçük gruplarla çalışmayı tercih ederim...",
                "tip": "Günlük sosyal etkileşimlerinizi düşünün.",
                "required": True
            }
        ]
    },
    {
        "title": "Yaratıcılık ve Problem Çözme",
        "icon": "💡",
        "questions": [
            {
                "id": "creativity",
                "question": "Son zamanlarda gerçekleştirdiğiniz en yaratıcı proje neydi?",
                "placeholder": "Örn: Okul gazetesi için özgün bir köşe tasarladım...",
                "tip": "Sadece sanatsal değil, her türlü yaratıcı çalışmayı düşünün.",
                "required": True
            },
            {
                "id": "problem_solving",
                "question": "Karşılaştığınız zorlu bir problemi nasıl çözdüğünüzü anlatır mısınız?",
                "placeholder": "Örn: Okul kulübünde yaşanan bir sorunu çözerken...",
                "tip": "Problemi nasıl analiz ettiğinizi ve çözüm sürecinizi düşünün.",
                "required": True
            }
        ]
    },
    {
        "title": "Kişisel İlgi Alanları ve Hobiler",
        "icon": "🎨",
        "questions": [
            {
                "id": "personal_interests",
                "question": "Boş zamanlarınızda en çok ne yapmaktan hoşlanırsınız?",
                "placeholder": "Örn: Kitap okumak, spor yapmak, müzik dinlemek...",
                "tip": "Hobilerinizin sizi nasıl geliştirdiğini düşünün.",
                "required": True
            },
            {
                "id": "creative_activities",
                "question": "Yaratıcı aktiviteleriniz nelerdir ve bu aktiviteleri yaparken ne hissedersiniz?",
                "placeholder": "Örn: Resim yapmak, yazı yazmak, dans etmek...",
                "tip": "Yaratıcılığınızı ifade ettiğiniz anları düşünün.",
                "required": True
            }
        ]
    },
    {
        "title": "Değerler ve Motivasyon",
        "icon": "⭐",
        "questions": [
            {
                "id": "values",
                "question": "Gelecekteki kariyerinizde sizin için en önemli değerler nelerdir?",
                "placeholder": "Örn: İnsanlara yardım etmek, yenilikçi olmak, çevreyi korumak...",
                "tip": "Size anlam ve mutluluk veren şeyleri düşünün.",
                "required": True
            },
            {
                "id": "motivation",
                "question": "Sizi en çok motive eden başarı hikayenizi paylaşır mısınız?",
                "placeholder": "Örn: Bir yarışmada elde ettiğim başarı...",
                "tip": "Kendinizle gurur duyduğunuz anları düşünün.",
                "required": True
            }
        ]
    },
    {
        "title": "Gelecek Hedefleri",
        "icon": "🎯",
        "questions": [
            {
                "id": "future_goals",
                "question": "Kendinizi 10 yıl sonra nerede görüyorsunuz?",
                "placeholder": "Örn: Kendi şirketimi kurmak istiyorum...",
                "tip": "Gerçekçi ama idealist hedeflerinizi düşünün.",
                "required": True
            },
            {
                "id": "work_life_balance",
                "question": "İş-yaşam dengesi konusundaki öncelikleriniz nelerdir?",
                "placeholder": "Örn: Esnek çalışma saatleri, seyahat imkanı...",
                "tip": "Hayat tarzınıza uygun çalışma koşullarını düşünün.",
                "required": True
            }
        ]
    }
]

def extract_json_from_response(response_text):
    """LLM yanıtından JSON formatını çıkarır"""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        json_pattern = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)
        match = json_pattern.search(response_text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        curly_pattern = re.compile(r'\{.*\}', re.DOTALL)
        match = curly_pattern.search(response_text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise ValueError("Geçerli JSON formatı bulunamadı")

def analyze_responses_with_llm(responses):
    """Gelişmiş LLM analizi"""
    try:
        prompt = f"""
        Bir Türk lise öğrencisinin aşağıdaki yanıtlarını detaylı olarak analiz et:
        {responses}

        Sadece aşağıdaki JSON formatında yanıt ver (başka açıklama ekleme) Sadece JSON döndür!:
        {{
            "kişilik_profili": {{
                "analitik": 85,
                "sosyal": 75,
                "yaraticilik": 80,
                "liderlik": 70,
                "teknik": 90,
                "uyumluluk": 85,
                "girişimcilik": 75,
                "ana_özellikler": ["özellik1", "özellik2", "özellik3"],
                "güçlü_yönler": ["güçlü_yön1", "güçlü_yön2", "güçlü_yön3"],
                "gelişim_alanları": ["alan1", "alan2", "alan3"]
            }},
            "yetkinlik_analizi": {{
                "teknik_beceriler": ["beceri1", "beceri2", "beceri3"],
                "sosyal_beceriler": ["beceri1", "beceri2", "beceri3"],
                "dil_becerileri": ["dil1", "dil2"],
                "akademik_yönelim": "detaylı akademik yönelim açıklaması",
                "öğrenme_stili": "öğrenme stili açıklaması"
            }},
            "ilgi_alanları": {{
                "birincil": ["alan1", "alan2"],
                "ikincil": ["alan1", "alan2"],
                "hobi_ve_aktiviteler": ["hobi1", "hobi2"]
            }},
            "çalışma_tercihleri": {{
                "ortam": ["tercih1", "tercih2"],
                "iş_stili": ["stil1", "stil2"],
                "takım_rolü": "tercih edilen takım rolü"
            }}
        }}
        """

        response = model.generate_content(prompt)
        print("API Yanıtı:", response.text)
        
        result = extract_json_from_response(response.text)
        print("İşlenmiş JSON:", result)
        
        return result
    except Exception as e:
        print(f"Hata detayı: {str(e)}")
        handle_llm_error(e, "api_error")
        return None

def get_career_recommendations_with_llm(profile):
    """Enhanced career recommendations with LLM"""
    try:
        prompt = f"""
        Bir Türk lise öğrencisi için aşağıdaki profili analiz et ve en az 5 meslek önerisi sun:
        {profile}

        Sadece aşağıdaki JSON formatında yanıt ver:
        {{
            "kariyer_önerileri": [
                {{
                    "meslek": "meslek_adı",
                    "uyum_puanı": 85,
                    "açıklama": "detaylı meslek açıklaması",
                    "gelişim_potansiyeli": {{
                        "kariyer_yolu": ["başlangıç", "orta", "ileri seviye pozisyonlar"],
                        "gelecek_trendleri": ["trend1", "trend2"],
                        "tahmini_büyüme": "büyüme oranı ve nedenleri",
                        "sektör_analizi": "sektörün geleceği hakkında detaylı analiz"
                    }},
                    "gerekli_beceriler": {{
                        "teknik_beceriler": ["beceri1", "beceri2"],
                        "kişisel_beceriler": ["beceri1", "beceri2"],
                        "dil_gereksinimleri": ["dil1", "dil2"],
                        "sertifikalar": ["sertifika1", "sertifika2"]
                    }},
                    "çalışma_ortamı": {{
                        "çalışma_yeri": ["ofis", "uzaktan", "hibrit"],
                        "iş_seyahati": "seyahat gereksinimleri",
                        "çalışma_saatleri": "tipik çalışma düzeni",
                        "iş_yaşam_dengesi": "denge açıklaması"
                    }},
                    "eğitim_yolu": {{
                        "önerilen_bölümler": [
                            {{
                                "bölüm_adı": "bölüm",
                                "üniversiteler": ["üni1", "üni2"],
                                "eğitim_süresi": "4 yıl",
                                "önemli_dersler": ["ders1", "ders2"],
                                "yüksek_lisans": "yüksek lisans seçenekleri",
                                "staj_imkanları": ["staj1", "staj2"]
                            }}
                        ],
                        "sertifika_programları": ["program1", "program2"],
                        "online_eğitimler": ["eğitim1", "eğitim2"]
                    }},
                    "maaş_bilgisi": {{
                        "başlangıç": "XXX-XXX TL",
                        "ortalama": "XXX-XXX TL",
                        "üst_düzey": "XXX-XXX TL",
                        "yan_haklar": ["yan_hak1", "yan_hak2"]
                    }},
                    "tavsiyeler": {{
                        "kariyer_hazırlığı": ["tavsiye1", "tavsiye2"],
                        "networking": ["öneri1", "öneri2"],
                        "kişisel_gelişim": ["öneri1", "öneri2"]
                    }}
                }}
            ],
            "ek_kaynaklar": {{
                "online_platformlar": ["platform1", "platform2"],
                "kitaplar": ["kitap1", "kitap2"],
                "youtube_kanalları": ["kanal1", "kanal2"],
                "podcasts": ["podcast1", "podcast2"],
                "mentor_programları": ["program1", "program2"]
            }},
            "gelişim_yol_haritası": {{
                "lise_dönemi": ["hedef1", "hedef2"],
                "üniversite_hazırlık": ["hedef1", "hedef2"],
                "üniversite_dönemi": ["hedef1", "hedef2"],
                "mezuniyet_sonrası": ["hedef1", "hedef2"]
            }}
        }}
        """

        response = model.generate_content(prompt)
        print("API Yanıtı:", response.text)
        
        result = extract_json_from_response(response.text)
        print("İşlenmiş JSON:", result)
        
        return result
    except Exception as e:
        print(f"Hata detayı: {str(e)}")
        handle_llm_error(e, "api_error")
        return None

def create_radar_chart(personality_data):
    """Create an enhanced radar chart for personality analysis"""
    categories = list(personality_data.keys())
    values = list(personality_data.values())
    
    fig = go.Figure()
    
    # Add main radar plot
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='#1a73e8',
        fillcolor='rgba(26,115,232,0.2)',
        name='Kişilik Profili'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=False,
                tickfont=dict(size=10, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            angularaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='rgba(255,255,255,0.2)',
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(size=12, color='white')
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    return fig

def create_salary_chart(career):
    """Create an enhanced salary progression chart"""
    def extract_average(salary_range):
        # Remove 'TL' and strip whitespace
        salary_range = salary_range.replace('TL', '').strip()
        
        # Handle 'Değişken' case
        if salary_range == 'Değişken':
            return 0
        
        # Extract the average of the range
        try:
            # Split the range and convert to numbers
            parts = salary_range.split('-')
            if len(parts) == 2:
                start = float(parts[0].replace('.', '').replace(',', '').strip())
                end = float(parts[1].replace('.', '').replace(',', '').strip())
                return (start + end) / 2
            else:
                # Handle single number (e.g., "50.000+")
                return float(parts[0].replace('+', '').replace('.', '').replace(',', '').strip())
        except:
            return 0

    salary_data = {
        'Seviye': ['Başlangıç', 'Ortalama', 'Üst Düzey'],
        'Maaş': [
            extract_average(career['maaş_bilgisi']['başlangıç']),
            extract_average(career['maaş_bilgisi']['ortalama']),
            extract_average(career['maaş_bilgisi']['üst_düzey'])
        ]
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=salary_data['Seviye'],
            y=salary_data['Maaş'],
            marker_color=['#64B5F6', '#42A5F5', '#1E88E5'],
            text=[f"{maaş:,.0f} TL" for maaş in salary_data['Maaş']],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Kariyer Maaş Gelişimi",
        title_font_color='white',
        font_color='white',
        yaxis_title="Maaş (TL)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        )
    )
    
    return fig


def display_career_recommendations():
    """Display career recommendations with single-level expanders"""
    st.header("🎯 Size Özel Kariyer Önerileri")

    if not st.session_state.recommendations or "kariyer_önerileri" not in st.session_state.recommendations:
        st.warning("Kariyer önerileri yüklenirken bir sorun oluştu. Lütfen tekrar deneyin.")
        return

    # Filtering and sorting options
    col1, col2 = st.columns([2, 2])
    with col1:
        min_match = st.slider("Minimum Uyum Yüzdesi", 0, 100, 50)
    with col2:
        sort_by = st.selectbox(
            "Sıralama Kriteri",
            ["Uyum Puanı", "Maaş Potansiyeli", "Gelişim Potansiyeli"]
        )

    filtered_careers = [
        career for career in st.session_state.recommendations["kariyer_önerileri"]
        if career["uyum_puanı"] >= min_match
    ]

    # Sort careers based on selected criterion
    if sort_by == "Uyum Puanı":
        filtered_careers.sort(key=lambda x: x["uyum_puanı"], reverse=True)
    elif sort_by == "Maaş Potansiyeli":
        filtered_careers.sort(
            key=lambda x: float(x["maaş_bilgisi"]["üst_düzey"].replace("TL", "").replace(",", "")),
            reverse=True
        )

    # Display each career recommendation
    for career in filtered_careers:
        st.markdown(f"## 🎯 {career['meslek']} - {career['uyum_puanı']}% Uyum")
        display_career_details(career)
        st.divider()

def display_career_details(career):
    """Display detailed career information without nested expanders"""
    # Main info and description
    st.subheader("📝 Meslek Açıklaması")
    st.write(career["açıklama"])
    
    # Use tabs for main sections to avoid nesting
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Gelişim ve Maaş",
        "🔑 Beceriler",
        "🎓 Eğitim",
        "💡 Tavsiyeler"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            display_development_potential(career["gelişim_potansiyeli"])
        
        with col2:
            display_salary_info(career["maaş_bilgisi"], career['meslek'])
            display_work_environment(career["çalışma_ortamı"])
    
    with tab2:
        display_skills_section(career["gerekli_beceriler"])
    
    with tab3:
        display_education_path_section(career["eğitim_yolu"])
    
    with tab4:
        display_career_advice(career["tavsiyeler"])

def display_development_potential(potential):
    """Display career development potential"""
    st.markdown("### 📈 Gelişim Potansiyeli")
    
    # Career Path Timeline
    st.markdown("#### Kariyer Yolu")
    for i, step in enumerate(potential["kariyer_yolu"], 1):
        st.markdown(f"{i}. {step}")
    
    # Future Trends
    st.markdown("#### Gelecek Trendleri")
    for trend in potential["gelecek_trendleri"]:
        st.markdown(f"- {trend}")
    
    # Growth Forecast and Sector Analysis
    st.markdown("#### Sektör Analizi")
    st.info(potential["tahmini_büyüme"])
    st.write(potential["sektör_analizi"])

def display_salary_info(salary_info, career_name):
    """Display salary information with visualization"""
    st.markdown("### 💰 Maaş Bilgisi")
    
    # Check if salary info is variable
    if salary_info['başlangıç'] == 'Değişken':
        st.info("Bu meslek için maaş bilgisi değişkenlik göstermektedir.")
    else:
        # Display salary chart
        fig = create_salary_chart({"maaş_bilgisi": salary_info})
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"salary_chart_{career_name}"
        )
    
    # Additional benefits
    st.markdown("#### Yan Haklar")
    if isinstance(salary_info["yan_haklar"], list):
        for hak in salary_info["yan_haklar"]:
            st.markdown(f"- {hak}")
    else:
        st.write(salary_info["yan_haklar"])



def display_work_environment(environment):
    """Display work environment information"""
    st.markdown("### 🏢 Çalışma Ortamı")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Work location
        st.markdown("#### Çalışma Yeri")
        for yer in environment["çalışma_yeri"]:
            st.markdown(f"- {yer}")
    
    with col2:
        # Work conditions and work-life balance
        st.markdown("#### Çalışma Koşulları")
        st.write(environment["çalışma_saatleri"])
        st.write(environment["iş_seyahati"])
        
        st.markdown("#### İş-Yaşam Dengesi")
        st.write(environment["iş_yaşam_dengesi"])


def display_skills_section(skills):
    """Display required skills with visual indicators"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💻 Teknik Beceriler")
        for skill in skills["teknik_beceriler"]:
            st.progress(0.8)
            st.write(skill)
        
        st.markdown("#### 🌐 Dil Gereksinimleri")
        for lang in skills["dil_gereksinimleri"]:
            st.write(lang)
    
    with col2:
        st.markdown("#### 👥 Kişisel Beceriler")
        for skill in skills["kişisel_beceriler"]:
            st.progress(0.7)
            st.write(skill)
        
        st.markdown("#### 📜 Gerekli Sertifikalar")
        for cert in skills["sertifikalar"]:
            st.write(cert)

def display_education_path_section(education):
    """Display education path information without expanders"""
    st.markdown("### 📚 Önerilen Bölümler")
    
    for program in education["önerilen_bölümler"]:
        st.markdown(f"#### {program['bölüm_adı']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Eğitim Süresi:** {program['eğitim_süresi']}")
            st.markdown("**Önerilen Üniversiteler:**")
            for uni in program["üniversiteler"]:
                st.markdown(f"- {uni}")
            
            if program.get("yüksek_lisans"):
                st.markdown("**Yüksek Lisans Olanakları:**")
                st.write(program["yüksek_lisans"])
        
        with col2:
            st.markdown("**Önemli Dersler:**")
            for ders in program["önemli_dersler"]:
                st.markdown(f"- {ders}")
            
            st.markdown("**Staj İmkanları:**")
            for staj in program["staj_imkanları"]:
                st.markdown(f"- {staj}")
        
        st.divider()
    
    # Additional education options
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📚 Sertifika Programları")
        for program in education["sertifika_programları"]:
            st.markdown(f"- {program}")
    
    with col2:
        st.markdown("#### 💻 Online Eğitimler")
        for course in education["online_eğitimler"]:
            st.markdown(f"- {course}")

def display_career_advice(advice):
    """Display career preparation advice"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Kariyer Hazırlığı")
        for tip in advice["kariyer_hazırlığı"]:
            st.markdown(f"- {tip}")
    
    with col2:
        st.markdown("#### 🤝 Networking")
        for tip in advice["networking"]:
            st.markdown(f"- {tip}")
        
        st.markdown("#### 📚 Kişisel Gelişim")
        for tip in advice["kişisel_gelişim"]:
            st.markdown(f"- {tip}")


def display_profile_analysis():
    """Enhanced profile analysis display"""
    if not st.session_state.profile:
        st.warning("Profil analizi henüz yapılmamış.")
        return

    st.header("📊 Kişilik ve Yetkinlik Profili")
    
    # Main profile analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Radar chart for personality traits
        personality_traits = {
            "Analitik": st.session_state.profile["kişilik_profili"]["analitik"],
            "Sosyal": st.session_state.profile["kişilik_profili"]["sosyal"],
            "Yaratıcılık": st.session_state.profile["kişilik_profili"]["yaraticilik"],
            "Liderlik": st.session_state.profile["kişilik_profili"]["liderlik"],
            "Teknik": st.session_state.profile["kişilik_profili"]["teknik"],
            "Uyumluluk": st.session_state.profile["kişilik_profili"]["uyumluluk"],
            "Girişimcilik": st.session_state.profile["kişilik_profili"]["girişimcilik"]
        }
        st.plotly_chart(
            create_radar_chart(personality_traits),
            use_container_width=True,
            key="radar_chart"  # Benzersiz key ekledik
        )
    
    with col2:
        # Key characteristics and strengths
        st.subheader("🌟 Ana Özellikler")
        for özellik in st.session_state.profile["kişilik_profili"]["ana_özellikler"]:
            st.markdown(f"- {özellik}")
        
        st.divider()
        
        st.subheader("💪 Güçlü Yönler")
        for güçlü_yön in st.session_state.profile["kişilik_profili"]["güçlü_yönler"]:
            st.markdown(f"- {güçlü_yön}")
    
    # Detailed competency analysis
    st.header("🎯 Yetkinlik Analizi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💻 Teknik Beceriler")
        for beceri in st.session_state.profile["yetkinlik_analizi"]["teknik_beceriler"]:
            with st.container():
                st.markdown(f"- {beceri}")
                st.progress(0.8)
    
    with col2:
        st.subheader("🤝 Sosyal Beceriler")
        for beceri in st.session_state.profile["yetkinlik_analizi"]["sosyal_beceriler"]:
            with st.container():
                st.markdown(f"- {beceri}")
                st.progress(0.7)
    
    with col3:
        st.subheader("🌍 Dil Becerileri")
        for dil in st.session_state.profile["yetkinlik_analizi"]["dil_becerileri"]:
            with st.container():
                st.markdown(f"- {dil}")
                st.progress(0.75)
    
    # Learning style and preferences
    st.header("📚 Öğrenme ve Çalışma Tercihleri")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📖 Öğrenme Stili")
        st.write(st.session_state.profile["yetkinlik_analizi"]["öğrenme_stili"])
        
        st.subheader("🎯 Akademik Yönelim")
        st.write(st.session_state.profile["yetkinlik_analizi"]["akademik_yönelim"])
    
    with col2:
        st.subheader("💼 Çalışma Tercihleri")
        st.markdown("#### Tercih Edilen Ortam")
        for ortam in st.session_state.profile["çalışma_tercihleri"]["ortam"]:
            st.markdown(f"- {ortam}")
        
        st.markdown("#### İş Stili")
        for stil in st.session_state.profile["çalışma_tercihleri"]["iş_stili"]:
            st.markdown(f"- {stil}")
        
        st.markdown("#### Takım Rolü")
        st.write(st.session_state.profile["çalışma_tercihleri"]["takım_rolü"])

def display_education_resources():
    """Display educational resources and recommendations"""
    if not st.session_state.recommendations or "ek_kaynaklar" not in st.session_state.recommendations:
        st.warning("Kaynak önerileri yüklenirken bir sorun oluştu.")
        return

    st.header("📚 Eğitim ve Gelişim Kaynakları")
    
    # Create tabs for different resource types
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎓 Online Platformlar",
        "📚 Kitaplar",
        "🎥 YouTube",
        "🎧 Podcasts",
        "👥 Mentörlük"
    ])
    
    with tab1:
        st.subheader("💻 Önerilen Online Platformlar")
        resources = st.session_state.recommendations["ek_kaynaklar"]["online_platformlar"]
        for platform in resources:
            st.markdown(f"- {platform}")
    
    with tab2:
        st.subheader("📚 Önerilen Kitaplar")
        books = st.session_state.recommendations["ek_kaynaklar"]["kitaplar"]
        for book in books:
            st.markdown(f"- {book}")
    
    with tab3:
        st.subheader("🎥 YouTube Kanalları")
        channels = st.session_state.recommendations["ek_kaynaklar"]["youtube_kanalları"]
        for channel in channels:
            st.markdown(f"- {channel}")
    
    with tab4:
        st.subheader("🎧 Podcast Önerileri")
        podcasts = st.session_state.recommendations["ek_kaynaklar"]["podcasts"]
        for podcast in podcasts:
            st.markdown(f"- {podcast}")
    
    with tab5:
        st.subheader("👥 Mentörlük Programları")
        programs = st.session_state.recommendations["ek_kaynaklar"]["mentor_programları"]
        for program in programs:
            st.markdown(f"- {program}")

def display_development_roadmap():
    """Display personal development roadmap"""
    if not st.session_state.recommendations or "gelişim_yol_haritası" not in st.session_state.recommendations:
        st.warning("Gelişim yol haritası yüklenirken bir sorun oluştu.")
        return

    st.header("🗺️ Kişisel Gelişim Yol Haritası")
    
    roadmap = st.session_state.recommendations["gelişim_yol_haritası"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Lise Dönemi")
        for hedef in roadmap["lise_dönemi"]:
            st.markdown(f"- {hedef}")
        
        st.subheader("🎯 Üniversite Hazırlık")
        for hedef in roadmap["üniversite_hazırlık"]:
            st.markdown(f"- {hedef}")
    
    with col2:
        st.subheader("🎓 Üniversite Dönemi")
        for hedef in roadmap["üniversite_dönemi"]:
            st.markdown(f"- {hedef}")
        
        st.subheader("💼 Mezuniyet Sonrası")
        for hedef in roadmap["mezuniyet_sonrası"]:
            st.markdown(f"- {hedef}")

def generate_pdf_report():
    """Generate PDF report of analysis and recommendations"""
    if not st.session_state.profile or not st.session_state.recommendations:
        return

    try:
        # Create a PDF buffer (implementation would go here)
        pdf_buffer = BytesIO()
        
        # Add download button
        st.download_button(
            label="📄 PDF Raporu İndir",
            data=pdf_buffer,
            file_name="kariyer_raporu.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error("PDF raporu oluşturulurken bir hata oluştu.")
        print(f"PDF hata detayı: {str(e)}")
def initialize_session_state():
    """Initialize or reset session state"""
    if 'current_section' not in st.session_state:
        st.session_state.current_section = 0
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'profile' not in st.session_state:
        st.session_state.profile = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    if 'show_tips' not in st.session_state:
        st.session_state.show_tips = True

def validate_responses(questions):
    """Validate user responses"""
    for question in questions:
        if question["required"]:
            response = st.session_state.responses.get(question["id"], "").strip()
            if response is None or len(response.strip()) < 10:
                return False
            if not response:
                return False
            if len(response) < 10:  # Minimum length check
                return False
    return True

def process_responses():
    """Process user responses and generate analysis"""
    with st.spinner("🔍 Detaylı profil analizi yapılıyor..."):
        try:
            st.session_state.profile = analyze_responses_with_llm(st.session_state.responses)
            if not st.session_state.profile:
                st.error("Profil analizi oluşturulamadı. Lütfen tekrar deneyin.")
                return False
        except Exception as e:
            handle_llm_error(e, "profile_analysis_error")
            return False

    with st.spinner("🎯 Kariyer önerileri hazırlanıyor..."):
        try:
            st.session_state.recommendations = get_career_recommendations_with_llm(
                st.session_state.profile
            )
            if not st.session_state.recommendations:
                st.error("Kariyer önerileri oluşturulamadı. Lütfen tekrar deneyin.")
                return False
        except Exception as e:
            handle_llm_error(e, "recommendations_error")
            return False

    return True

def display_questions_section():
    """Enhanced question section display"""
    current = SECTIONS[st.session_state.current_section]
    
    # Progress tracking
    progress = (st.session_state.current_section + 1) / len(SECTIONS)
    st.progress(progress)
    st.markdown(f"*Bölüm {st.session_state.current_section + 1} / {len(SECTIONS)}*")

    # Question display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class='animated'>
            <h2>{current['icon']} {current['title']}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        for question in current["questions"]:
            st.markdown(f"""
            <div class='animated'>
                <h4>{question['question']}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Get existing response or empty string
            existing_response = st.session_state.responses.get(question["id"], "")
            
            # Response input with character counter
            response = st.text_area(
                "Cevabınız:",
                value=existing_response,
                placeholder=question["placeholder"],
                height=150,
                key=question["id"],
                help="En az 10 karakter giriniz."
            )
            
            # Character count display (handle None case)
            if response is not None:
                char_count = len(response)
                st.caption(f"Karakter sayısı: {char_count}/500")
                st.session_state.responses[question["id"]] = response
            else:
                st.caption("Karakter sayısı: 0/500")
                st.session_state.responses[question["id"]] = ""

    with col2:
        if st.session_state.show_tips:
            st.markdown(f"""
            <div class="tip-box animated">
                <h4>💡 Rehber İpuçları</h4>
                <p>{current['questions'][0]['tip']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card animated">
            <h4>📊 İlerleme Durumu</h4>
            <p>Tamamlanan: {st.session_state.current_section + 1}/{len(SECTIONS)}</p>
            <p>Kalan Bölüm: {len(SECTIONS) - (st.session_state.current_section + 1)}</p>
        </div>
        """, unsafe_allow_html=True)

    # Navigation buttons
    col_space1, col_prev, col_next, col_space2 = st.columns([1, 1, 1, 1])
    
    with col_prev:
        if st.session_state.current_section > 0:
            if st.button("⬅️ Önceki", use_container_width=True):
                st.session_state.current_section -= 1
                st.rerun()
    
    with col_next:
        if st.session_state.current_section < len(SECTIONS) - 1:
            next_button_label = "İlerle ➡️"
        else:
            next_button_label = "Analizi Tamamla ✨"
            
        if st.button(next_button_label, use_container_width=True):
            if validate_responses(current["questions"]):
                st.session_state.current_section += 1
                st.rerun()
            else:
                st.error("Lütfen tüm soruları eksiksiz ve yeterli uzunlukta yanıtlayın.")

def display_results():
    """Display analysis results with tabs"""
    # Create navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Kariyer Önerileri",
        "📊 Profil Analizi",
        "🎓 Eğitim Yolu",
        "📚 Kaynaklar"
    ])

    with tab1:
        display_career_recommendations()

    with tab2:
        display_profile_analysis()

    with tab3:
        display_development_roadmap()

    with tab4:
        display_education_resources()

    # PDF Report Generation
    st.sidebar.markdown("### 📄 Rapor Oluştur")
    if st.sidebar.button("PDF Raporu İndir"):
        generate_pdf_report()

import sqlite3
from datetime import datetime
import streamlit as st
import base64
from io import BytesIO

# Forum Tablolarını Başlatma
def init_forum_tables():
    """Forum tablolarını başlat ve tüm gerekli tabloları oluştur"""
    conn = sqlite3.connect('motikoc.db')
    c = conn.cursor()
    
    try:
        # Forum kategorileri
        c.execute('''CREATE TABLE IF NOT EXISTS forum_categories
                     (id INTEGER PRIMARY KEY,
                      name TEXT,
                      description TEXT,
                      icon TEXT,
                      order_index INTEGER)''')
        
        # Sorular tablosu
        c.execute('''CREATE TABLE IF NOT EXISTS forum_questions
                     (id INTEGER PRIMARY KEY,
                      user_id INTEGER,
                      category_id INTEGER,
                      title TEXT,
                      content TEXT,
                      tags TEXT,
                      created_at TEXT,
                      updated_at TEXT,
                      view_count INTEGER DEFAULT 0,
                      is_solved BOOLEAN DEFAULT FALSE,
                      FOREIGN KEY(user_id) REFERENCES users(id),
                      FOREIGN KEY(category_id) REFERENCES forum_categories(id))''')
        
        # Cevaplar tablosu
        c.execute('''CREATE TABLE IF NOT EXISTS forum_answers
                     (id INTEGER PRIMARY KEY,
                      question_id INTEGER,
                      user_id INTEGER,
                      content TEXT,
                      created_at TEXT,
                      updated_at TEXT,
                      is_accepted BOOLEAN DEFAULT FALSE,
                      upvotes INTEGER DEFAULT 0,
                      FOREIGN KEY(question_id) REFERENCES forum_questions(id),
                      FOREIGN KEY(user_id) REFERENCES users(id))''')
        
        # Oylar tablosu
        c.execute('''CREATE TABLE IF NOT EXISTS forum_votes
                     (id INTEGER PRIMARY KEY,
                      user_id INTEGER,
                      content_type TEXT,
                      content_id INTEGER,
                      vote_type INTEGER,
                      created_at TEXT,
                      FOREIGN KEY(user_id) REFERENCES users(id))''')
        
        # Bildirimler tablosu
        c.execute('''CREATE TABLE IF NOT EXISTS forum_notifications
                     (id INTEGER PRIMARY KEY,
                      user_id INTEGER,
                      content TEXT,
                      link TEXT,
                      created_at TEXT,
                      is_read BOOLEAN DEFAULT FALSE,
                      FOREIGN KEY(user_id) REFERENCES users(id))''')
        
        # Forum kategorilerini temizle ve yeniden ekle
        c.execute('DELETE FROM forum_categories')
        
        # Varsayılan kategoriler
        default_categories = [
            ('TYT Matematik', 'TYT matematik konuları ve soru çözümleri', '📐', 1),
            ('TYT Türkçe', 'TYT Türkçe ve dil bilgisi konuları', '📚', 2),
            ('TYT Fen Bilimleri', 'TYT Fizik, Kimya ve Biyoloji', '🔬', 3),
            ('TYT Sosyal Bilimler', 'TYT Tarih, Coğrafya ve Felsefe', '🌍', 4),
            ('AYT Matematik', 'AYT matematik konuları ve soru çözümleri', '🧮', 5),
            ('AYT Fizik', 'AYT fizik konuları ve problemler', '⚡', 6),
            ('AYT Kimya', 'AYT kimya konuları ve deneyler', '⚗️', 7),
            ('AYT Biyoloji', 'AYT biyoloji konuları', '🧬', 8),
            ('Genel YKS', 'YKS sınavı hakkında genel konular', '📋', 9),
            ('Motivasyon', 'Motivasyon ve çalışma teknikleri', '💪', 10)
        ]
        
        c.executemany('''
            INSERT INTO forum_categories (name, description, icon, order_index)
            VALUES (?, ?, ?, ?)
        ''', default_categories)
        
        conn.commit()
        
    except Exception as e:
        print(f"Tablo oluşturma hatası: {str(e)}")
        conn.rollback()
    
    finally:
        conn.close()

# Forum Ana Sayfası
def show_forum():
    """Ana forum sayfası"""
    st.title("🎯 YKS Soru-Cevap Forumu")
    
    # Session state kontrolü
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "main"
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    
    # Sekmeleri göster
    tab1, tab2, tab3, tab4 = st.tabs([
        "Tüm Sorular",
        "Soru Sor",
        "Kategoriler",
        "Arama"
    ])
    
    # Eğer bir soru detayı görüntüleniyorsa
    if st.session_state.current_page == "question_detail" and st.session_state.current_question:
        show_question_detail(st.session_state.current_question)
    else:
        # Normal sayfa akışı
        with tab1:
            show_all_questions()
        
        with tab2:
            show_ask_question_form()
            
        with tab3:
            if st.session_state.current_page == "category_questions":
                show_category_questions(st.session_state.selected_category)
            else:
                show_categories()
        
        with tab4:
            show_search_questions()
    
    # Geri dönüş butonu
    if st.session_state.current_page != "main":
        if st.button("⬅️ Ana Sayfaya Dön", key="back_to_main_button"):
            st.session_state.current_page = "main"
            st.session_state.selected_category = None
            st.session_state.current_question = None
            st.rerun()

# Tüm Soruları Gösterme Fonksiyonu
def show_all_questions():
    """Tüm soruları göster"""
    st.subheader("📚 Son Sorular")
    
    # Filtreleme seçenekleri
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        sort_by = st.selectbox(
            "Sıralama",
            ["En Yeni", "En Çok Cevaplanan", "En Çok Görüntülenen"],
            key="sort_by_all_questions"
        )
    
    with col2:
        filter_by = st.multiselect(
            "Filtrele",
            ["Çözülmemiş", "TYT", "AYT", "Benim Sorularım"],
            key="filter_by_all_questions"
        )
    
    with col3:
        show_all = st.checkbox("Tümünü Göster", value=False, key="show_all_questions_checkbox")
    
    try:
        # Veritabanı bağlantısı
        conn = sqlite3.connect('motikoc.db')
        
        # SQL sorgusu
        query = '''
            SELECT 
                q.id,
                q.title,
                q.content,
                q.created_at,
                q.view_count,
                q.is_solved,
                u.username,
                c.name as category_name,
                c.icon as category_icon,
                (SELECT COUNT(*) FROM forum_answers WHERE question_id = q.id) as answer_count,
                (SELECT COUNT(*) FROM forum_votes WHERE content_type = 'question' AND content_id = q.id AND vote_type = 1) as upvotes,
                (SELECT COUNT(*) FROM forum_votes WHERE content_type = 'question' AND content_id = q.id AND vote_type = -1) as downvotes
            FROM forum_questions q
            JOIN users u ON q.user_id = u.id
            JOIN forum_categories c ON q.category_id = c.id
            WHERE 1=1
        '''
        
        params = []
        
        # Filtreleri uygula
        if "Çözülmemiş" in filter_by:
            query += " AND q.is_solved = 0"
        if "TYT" in filter_by:
            query += " AND c.name LIKE ?"
            params.append("TYT%")
        if "AYT" in filter_by:
            query += " AND c.name LIKE ?"
            params.append("AYT%")
        if "Benim Sorularım" in filter_by:
            query += " AND q.user_id = ?"
            params.append(st.session_state.user_id)
        
        # Sıralama
        if sort_by == "En Yeni":
            query += " ORDER BY q.created_at DESC"
        elif sort_by == "En Çok Cevaplanan":
            query += " ORDER BY answer_count DESC"
        else:
            query += " ORDER BY q.view_count DESC"
        
        # Limit
        if not show_all:
            query += " LIMIT 20"
        
        # Soruları getir
        questions = conn.execute(query, params).fetchall()
        
        if questions:
            for q in questions:
                question_id = q[0]  # Soru ID'si
                
                # Oy durumu kontrolü
                user_vote = conn.execute('''
                    SELECT vote_type FROM forum_votes 
                    WHERE user_id = ? AND content_type = 'question' AND content_id = ?
                ''', (st.session_state.user_id, question_id)).fetchone()
                
                if user_vote:
                    user_vote = user_vote[0]
                else:
                    user_vote = 0  # Hiç oy vermemiş
                
                with st.container():
                    col1, col2, col3 = st.columns([5, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div style='
                            padding: 15px;
                            border-radius: 10px;
                            background-color: rgba(255, 255, 255, 0.05);
                            margin-bottom: 10px;
                        '>
                            <div style='display: flex; justify-content: space-between;'>
                                <h3>{q[1]}</h3>
                                <span>{q[8]} {q[7]}</span>
                            </div>
                            <div style='
                                color: #B0B0B0;
                                font-size: 14px;
                                display: flex;
                                justify-content: space-between;
                                margin-top: 10px;
                            '>
                                <span>👤 {q[6]} | 💬 {q[9]} cevap | 👁️ {q[4]} görüntülenme</span>
                                <span>📅 {q[3]}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Upvote butonu
                        if user_vote == 1:
                            upvote_label = "👍 Beğenildi"
                        else:
                            upvote_label = "👍 Beğen"
                        if st.button(upvote_label, key=f"upvote_question_{question_id}"):
                            vote_question(question_id, 1)
                            st.rerun()
                    
                    with col3:
                        # Downvote butonu
                        if user_vote == -1:
                            downvote_label = "👎 Beğenilmedi"
                        else:
                            downvote_label = "👎 Beğenme"
                        if st.button(downvote_label, key=f"downvote_question_{question_id}"):
                            vote_question(question_id, -1)
                            st.rerun()
                    
                    with col1:
                        # Oy sayısını göster
                        st.markdown(f"**👍 {q[10]}  👎 {q[11]}**")
                    
                    # Detay butonu
                    if st.button("Detayları Gör", key=f"view_question_{question_id}"):
                        # Session state'i güncelle
                        st.session_state.current_question = question_id
                        st.session_state.current_page = "question_detail"
                        st.rerun()
        
        else:
            st.info("Henüz soru bulunmuyor veya seçilen kriterlere uygun soru yok.")
        
    except Exception as e:
        st.error(f"Sorular yüklenirken bir hata oluştu: {str(e)}")
    
    finally:
        conn.close()
def show_ask_question_form():
    """Soru ekleme formu"""
    st.subheader("❓ Yeni Soru Sor")
    
    # Kategori seçimi
    conn = sqlite3.connect('motikoc.db')
    categories = conn.execute('SELECT id, name, icon FROM forum_categories ORDER BY order_index').fetchall()
    category_options = {f"{cat[2]} {cat[1]}": cat[0] for cat in categories}
    
    with st.form("ask_question_form"):
        # Kategori seçimi
        selected_category = st.selectbox(
            "Kategori seçin",
            options=list(category_options.keys()),
            key="ask_question_category_select"
        )
        
        # Soru başlığı
        title = st.text_input(
            "Soru başlığı",
            max_chars=100,
            placeholder="Sorunuzu kısa ve net bir şekilde özetleyin",
            key="ask_question_title_input"
        )
        
        # Soru detayları
        content = st.text_area(
            "Soru detayları",
            height=200,
            placeholder="Sorunuzu detaylı bir şekilde açıklayın",
            key="ask_question_content_textarea"
        )
        
        # Etiketler
        tags = st.text_input(
            "Etiketler",
            placeholder="Örnek: #türev #limit #integral (boşlukla ayırın)",
            key="ask_question_tags_input"
        )
        
        # Görsel yükleme
        image = st.file_uploader("Görsel ekleyin (opsiyonel)", type=['png', 'jpg', 'jpeg'], key="ask_question_image_uploader")
        
        submitted = st.form_submit_button("Soruyu Gönder")  # 'key' kaldırıldı
        
        if submitted:
            if not title or not content:
                st.error("Başlık ve soru detayları zorunludur!")
            else:
                try:
                    # Görseli işle
                    if image:
                        image_base64 = base64.b64encode(image.read()).decode()
                        content += f"\n<img src='data:image/{image.type.split('/')[-1]};base64,{image_base64}'>"
                    
                    # Zaman damgası
                    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Soruyu kaydet
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO forum_questions 
                        (user_id, category_id, title, content, tags, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        st.session_state.user_id,
                        category_options[selected_category],
                        title,
                        content,
                        tags,
                        now,
                        now
                    ))
                    
                    # Son eklenen sorunun ID'sini al
                    question_id = c.lastrowid
                    
                    conn.commit()
                    
                    # XP kazandır
                    update_user_xp(st.session_state.user_id, calculate_xp_for_activity('forum_question'))
                    
                    # Başarı mesajını göster
                    st.success("✅ Sorunuz başarıyla eklendi!")
                    time.sleep(3)  # Kullanıcının mesajı görmesi için kısa bekle
                    
                    # Soru detay sayfasına yönlendir
                    st.session_state.current_question = question_id
                    st.session_state.current_page = "question_detail"
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Soru eklenirken bir hata oluştu: {str(e)}")
                finally:
                    conn.close()

# Arama Fonksiyonu
def show_search_questions():
    """Forum içinde arama yapma fonksiyonu"""
    st.subheader("🔍 Forum İçinde Ara")
    
    # Arama parametreleri
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Arama yapmak için kelime girin",
            placeholder="Örnek: türev soru çözümü",
            key="search_query_input"
        )
    
    with col2:
        search_type = st.selectbox(
            "Arama Tipi",
            ["Tümü", "Başlıklar", "İçerik", "Etiketler"],
            key="search_type_main_search"
        )
    
    # Gelişmiş arama filtreleri
    with st.expander("Gelişmiş Arama Seçenekleri"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Kategori filtresi
            conn = sqlite3.connect('motikoc.db')
            categories = conn.execute(
                'SELECT id, name, icon FROM forum_categories ORDER BY name'
            ).fetchall()
            
            category_options = {"Tüm Kategoriler": None}
            category_options.update({
                f"{cat[2]} {cat[1]}": cat[0] for cat in categories
            })
            
            selected_category = st.selectbox(
                "Kategori",
                options=list(category_options.keys()),
                key="advanced_search_category_select"
            )
            
            # Tarih filtresi
            date_filter = st.selectbox(
                "Tarih",
                ["Tüm Zamanlar", "Bugün", "Bu Hafta", "Bu Ay", "Son 3 Ay"],
                key="advanced_search_date_filter_search"
            )
        
        with col2:
            # Durum filtresi
            status_filter = st.multiselect(
                "Durum",
                ["Çözülmüş", "Çözülmemiş", "En Çok Oylanan", "En Çok Cevaplanan"],
                key="advanced_search_status_filter_search"
            )
            
            # Kullanıcı filtresi
            user_filter = st.selectbox(
                "Kullanıcı",
                ["Tüm Kullanıcılar", "Benim Sorularım", "Cevapladıklarım"],
                key="advanced_search_user_filter_search"
            )
        
        with col3:
            # Sıralama
            sort_by = st.selectbox(
                "Sıralama",
                ["Tarih (Yeni-Eski)", "Tarih (Eski-Yeni)", 
                 "En Çok Oylanan", "En Çok Görüntülenen"],
                key="advanced_search_sort_by_search"
            )
            
            # Sayfa başına sonuç
            results_per_page = st.select_slider(
                "Sayfa Başına Sonuç",
                options=[10, 20, 50, 100],
                value=20,
                key="advanced_search_results_per_page_search"
            )
    
    # Arama yap butonu
    search_clicked = st.button("🔍 Ara", key="search_button_main_search")
    
    if search_query or search_clicked:
        try:
            # SQL sorgusu oluştur
            query = '''
                SELECT DISTINCT
                    q.id,
                    q.title,
                    q.content,
                    q.tags,
                    q.created_at,
                    q.is_solved,
                    q.view_count,
                    u.username,
                    c.name as category_name,
                    c.icon as category_icon,
                    (SELECT COUNT(*) FROM forum_answers WHERE question_id = q.id) as answer_count,
                    (SELECT COUNT(*) FROM forum_votes 
                     WHERE content_type = 'question' AND content_id = q.id AND vote_type = 1) as upvotes,
                    (SELECT COUNT(*) FROM forum_votes 
                     WHERE content_type = 'question' AND content_id = q.id AND vote_type = -1) as downvotes
                FROM forum_questions q
                JOIN users u ON q.user_id = u.id
                JOIN forum_categories c ON q.category_id = c.id
                WHERE 1=1
            '''
            
            params = []
            
            # Arama tipine göre WHERE koşulları
            if search_query:
                if search_type == "Başlıklar":
                    query += " AND q.title LIKE ?"
                    params.append(f"%{search_query}%")
                elif search_type == "İçerik":
                    query += " AND q.content LIKE ?"
                    params.append(f"%{search_query}%")
                elif search_type == "Etiketler":
                    query += " AND q.tags LIKE ?"
                    params.append(f"%{search_query}%")
                else:  # Tümü
                    query += " AND (q.title LIKE ? OR q.content LIKE ? OR q.tags LIKE ?)"
                    params.extend([f"%{search_query}%"] * 3)
            
            # Kategori filtresi
            if selected_category != "Tüm Kategoriler":
                query += " AND q.category_id = ?"
                params.append(category_options[selected_category])
            
            # Tarih filtresi
            if date_filter != "Tüm Zamanlar":
                if date_filter == "Bugün":
                    query += " AND DATE(q.created_at) = DATE('now')"
                elif date_filter == "Bu Hafta":
                    query += " AND DATE(q.created_at) >= DATE('now', '-7 days')"
                elif date_filter == "Bu Ay":
                    query += " AND DATE(q.created_at) >= DATE('now', '-1 month')"
                elif date_filter == "Son 3 Ay":
                    query += " AND DATE(q.created_at) >= DATE('now', '-3 months')"
            
            # Durum filtresi
            if "Çözülmüş" in status_filter:
                query += " AND q.is_solved = TRUE"
            if "Çözülmemiş" in status_filter:
                query += " AND q.is_solved = FALSE"
            if "En Çok Oylanan" in status_filter:
                query += " AND upvotes - downvotes >= 5"
            if "En Çok Cevaplanan" in status_filter:
                query += " AND answer_count >= 3"
            
            # Kullanıcı filtresi
            if user_filter == "Benim Sorularım":
                query += " AND q.user_id = ?"
                params.append(st.session_state.user_id)
            elif user_filter == "Cevapladıklarım":
                query += """ AND EXISTS (
                    SELECT 1 FROM forum_answers a 
                    WHERE a.question_id = q.id AND a.user_id = ?
                )"""
                params.append(st.session_state.user_id)
            
            # Sıralama
            if sort_by == "Tarih (Yeni-Eski)":
                query += " ORDER BY q.created_at DESC"
            elif sort_by == "Tarih (Eski-Yeni)":
                query += " ORDER BY q.created_at ASC"
            elif sort_by == "En Çok Oylanan":
                query += " ORDER BY (upvotes - downvotes) DESC"
            else:  # En Çok Görüntülenen
                query += " ORDER BY q.view_count DESC"
            
            # Limit
            query += f" LIMIT {results_per_page}"
            
            # Sorguyu çalıştır
            conn = sqlite3.connect('motikoc.db')
            results = conn.execute(query, params).fetchall()
            
            # Sonuçları göster
            st.markdown(f"### 🔎 Arama Sonuçları ({len(results)} sonuç)")
            
            if results:
                for result in results:
                    question_id = result[0]
                    # Oy durumu kontrolü
                    user_vote = conn.execute('''
                        SELECT vote_type FROM forum_votes 
                        WHERE user_id = ? AND content_type = 'question' AND content_id = ?
                    ''', (st.session_state.user_id, question_id)).fetchone()
                    
                    if user_vote:
                        user_vote = user_vote[0]
                    else:
                        user_vote = 0  # Hiç oy vermemiş
                    
                    with st.container():
                        col1, col2, col3 = st.columns([5, 1, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div style='
                                padding: 15px;
                                border-radius: 10px;
                                background-color: rgba(255, 255, 255, 0.05);
                                margin-bottom: 10px;
                            '>
                                <div style='display: flex; justify-content: space-between;'>
                                    <h3>{result[1]}</h3>
                                    <span>{result[9]} {result[8]}</span>
                                </div>
                                <p style='
                                    color: #B0B0B0;
                                    font-size: 14px;
                                    margin: 5px 0;
                                '>{result[2][:200]}...</p>
                                <div style='
                                    display: flex;
                                    justify-content: space-between;
                                    color: #B0B0B0;
                                    font-size: 14px;
                                    margin-top: 10px;
                                '>
                                    <span>
                                        👤 {result[7]} | 
                                        💬 {result[10]} cevap | 
                                        👁️ {result[6]} görüntülenme | 
                                        👍 {result[10]} oy | 👎 {result[11]} oy
                                    </span>
                                    <span>📅 {result[4]}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Upvote butonu
                            if user_vote == 1:
                                upvote_label = "👍 Beğenildi"
                            else:
                                upvote_label = "👍 Beğen"
                            if st.button(upvote_label, key=f"search_upvote_question_{question_id}"):
                                vote_question(question_id, 1)
                                st.rerun()
                        
                        with col3:
                            # Downvote butonu
                            if user_vote == -1:
                                downvote_label = "👎 Beğenilmedi"
                            else:
                                downvote_label = "👎 Beğenme"
                            if st.button(downvote_label, key=f"search_downvote_question_{question_id}"):
                                vote_question(question_id, -1)
                                st.rerun()
                        
                        with col1:
                            # Oy sayısını göster
                            st.markdown(f"**👍 {result[10]}  👎 {result[11]}**")
                        
                        # Detay butonu
                        if st.button("Detayları Gör", key=f"search_view_question_{question_id}"):
                            st.session_state.current_question = question_id
                            st.session_state.current_page = "question_detail"
                            st.rerun()
                    
                    # Etiketler
                    if result[3]:  # tags
                        st.markdown("**Etiketler:**")
                        for tag in result[3].split():
                            st.markdown(f"`{tag}`", unsafe_allow_html=True)
            
            else:
                st.info("Arama kriterlerinize uygun sonuç bulunamadı.")
            
            conn.close()
            
        except Exception as e:
            st.error(f"Arama sırasında bir hata oluştu: {str(e)}")
    else:
        st.info("Arama yapmak için yukarıdaki alana bir kelime yazın ve 'Ara' butonuna tıklayın.")

# Kategorileri Gösterme Fonksiyonu
def show_categories():
    """Forum kategorilerini benzersiz olarak göster"""
    st.subheader("📚 Kategoriler")
    
    try:
        # Veritabanı bağlantısı
        conn = sqlite3.connect('motikoc.db')
        c = conn.cursor()
        
        # Benzersiz kategorileri çek
        c.execute('''
            SELECT DISTINCT id, name, description, icon 
            FROM forum_categories 
            ORDER BY order_index
        ''')
        categories = c.fetchall()
        
        # Grid için kolonlar oluştur
        col1, col2 = st.columns(2)
        
        # Her kategoriyi uygun kolona yerleştir
        for i, category in enumerate(categories):
            # Kategori ID'si
            cat_id = category[0]
            
            # Bu kategorideki soru sayısını hesapla
            c.execute('''
                SELECT COUNT(DISTINCT id) 
                FROM forum_questions 
                WHERE category_id = ?
            ''', (cat_id,))
            question_count = c.fetchone()[0]
            
            # Kategoriyi göster
            with col1 if i % 2 == 0 else col2:
                with st.container():
                    st.markdown(f"""
                    <div style='
                        padding: 15px;
                        border-radius: 10px;
                        background-color: rgba(255, 255, 255, 0.05);
                        margin-bottom: 15px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    '>
                        <div style='
                            display: flex;
                            align-items: center;
                            margin-bottom: 10px;
                        '>
                            <span style='font-size: 24px; margin-right: 10px;'>{category[3]}</span>
                            <h3 style='margin: 0;'>{category[1]}</h3>
                        </div>
                        <p style='
                            color: #B0B0B0;
                            margin-bottom: 10px;
                            font-size: 14px;
                        '>{category[2]}</p>
                        <div style='
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                        '>
                            <span style='color: #8B949E;'>💬 {question_count} soru</span>
                            <div>
                    """, unsafe_allow_html=True)
                    
                    # "Soruları Gör" butonu
                    if st.button("Soruları Gör", key=f"view_category_{cat_id}"):
                        st.session_state.selected_category = cat_id
                        st.session_state.current_page = "category_questions"
                        st.rerun()
                    
                    st.markdown("</div></div></div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Kategoriler yüklenirken bir hata oluştu: {str(e)}")
    
    finally:
        # Veritabanı bağlantısını kapat
        conn.close()

# Seçilen Kategorideki Soruları Gösterme Fonksiyonu
def show_category_questions(category_id):
    """Seçilen kategorideki soruları göster"""
    conn = sqlite3.connect('motikoc.db')
    
    try:
        # Kategori bilgilerini al
        category = conn.execute('''
            SELECT name, icon, description 
            FROM forum_categories 
            WHERE id = ?
        ''', (category_id,)).fetchone()
        
        if not category:
            st.error("Kategori bulunamadı.")
            return
        
        # Kategori başlığı ve açıklaması
        st.markdown(f"""
        <div style='
            padding: 20px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.05);
            margin-bottom: 20px;
        '>
            <h2>{category[1]} {category[0]}</h2>
            <p style='color: #B0B0B0;'>{category[2]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Kategori filtreleri
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            sort_option = st.selectbox(
                "Sıralama",
                ["En Yeni", "En Çok Cevaplanan", "En Çok Görüntülenen"],
                key=f"sort_option_category_{category_id}"
            )
        
        with col2:
            only_unsolved = st.checkbox("Sadece Çözülmemiş", key=f"only_unsolved_category_{category_id}")
        
        with col3:
            # Kategori içi oylama
            pass  # Ekstra özellikler eklemek isterseniz buraya yazabilirsiniz
        
        # Sorgu oluştur
        query = '''
            SELECT 
                q.id,
                q.title,
                q.content,
                q.created_at,
                q.view_count,
                q.is_solved,
                u.username,
                c.name as category_name,
                c.icon as category_icon,
                (SELECT COUNT(*) FROM forum_answers WHERE question_id = q.id) as answer_count,
                (SELECT COUNT(*) FROM forum_votes WHERE content_type = 'question' AND content_id = q.id AND vote_type = 1) as upvotes,
                (SELECT COUNT(*) FROM forum_votes WHERE content_type = 'question' AND content_id = q.id AND vote_type = -1) as downvotes
            FROM forum_questions q
            JOIN users u ON q.user_id = u.id
            JOIN forum_categories c ON q.category_id = c.id
            WHERE q.category_id = ?
        '''
        
        params = [category_id]
        
        if only_unsolved:
            query += " AND q.is_solved = FALSE"
        
        if sort_option == "En Yeni":
            query += " ORDER BY q.created_at DESC"
        elif sort_option == "En Çok Cevaplanan":
            query += " ORDER BY answer_count DESC"
        else:
            query += " ORDER BY q.view_count DESC"
        
        questions = conn.execute(query, params).fetchall()
        
        # Soruları listele
        if questions:
            for question in questions:
                question_id = question[0]
                
                # Oy durumu kontrolü
                user_vote = conn.execute('''
                    SELECT vote_type FROM forum_votes 
                    WHERE user_id = ? AND content_type = 'question' AND content_id = ?
                ''', (st.session_state.user_id, question_id)).fetchone()
                
                if user_vote:
                    user_vote = user_vote[0]
                else:
                    user_vote = 0  # Hiç oy vermemiş
                
                with st.container():
                    col1, col2, col3 = st.columns([5, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div style='
                            padding: 15px;
                            border-radius: 10px;
                            background-color: rgba(255, 255, 255, 0.05);
                            margin-bottom: 10px;
                        '>
                            <h3>{question[1]}</h3>
                            <div style='
                                color: #B0B0B0;
                                font-size: 14px;
                                display: flex;
                                justify-content: space-between;
                                margin-top: 10px;
                            '>
                                <span>👤 {question[6]} | 💬 {question[9]} cevap | 👁️ {question[4]} görüntülenme</span>
                                <span>📅 {question[3]}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Upvote butonu
                        if user_vote == 1:
                            upvote_label = "👍 Beğenildi"
                        else:
                            upvote_label = "👍 Beğen"
                        if st.button(upvote_label, key=f"category_upvote_question_{question_id}"):
                            vote_question(question_id, 1)
                            st.rerun()
                    
                    with col3:
                        # Downvote butonu
                        if user_vote == -1:
                            downvote_label = "👎 Beğenilmedi"
                        else:
                            downvote_label = "👎 Beğenme"
                        if st.button(downvote_label, key=f"category_downvote_question_{question_id}"):
                            vote_question(question_id, -1)
                            st.rerun()
                    
                    with col1:
                        # Oy sayısını göster
                        st.markdown(f"**👍 {question[10]}  👎 {question[11]}**")
                    
                    # Detay butonu
                    if st.button("Detayları Gör", key=f"category_view_question_{question_id}"):
                        st.session_state.current_question = question_id
                        st.session_state.current_page = "question_detail"
                        st.rerun()
        
        else:
            st.info("Bu kategoride henüz soru yok. İlk soruyu siz sorun!")
        
        # Yeni Soru Ekle butonu
        st.markdown("---")
        if st.button("➕ Bu Kategoride Yeni Soru Sor", key=f"add_new_question_category_{category_id}"):
            st.session_state.current_page = "ask_question"
            st.session_state.selected_category = category_id
            st.rerun()
    
    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
    
    finally:
        conn.close()

# Soru Detay Sayfası
def show_question_detail(question_id):
    """Soru detay sayfası"""
    conn = sqlite3.connect('motikoc.db')
    
    try:
        # Görüntülenme sayısını artır
        conn.execute('''
            UPDATE forum_questions 
            SET view_count = view_count + 1 
            WHERE id = ?
        ''', (question_id,))
        conn.commit()
        
        # Soru detaylarını getir
        question = conn.execute('''
            SELECT q.id, q.user_id, q.category_id, q.title, q.content, q.tags, 
                   q.created_at, q.updated_at, q.view_count, q.is_solved,
                   u.username, c.name as category_name, c.icon as category_icon
            FROM forum_questions q
            JOIN users u ON q.user_id = u.id
            JOIN forum_categories c ON q.category_id = c.id
            WHERE q.id = ?
        ''', (question_id,)).fetchone()
        
        if not question:
            st.error("Soru bulunamadı!")
            return
        
        # Soru başlığı ve meta bilgiler
        st.markdown(f"""
        <div style='
            padding: 20px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.05);
            margin-bottom: 20px;
        '>
            <h2>{question[3]}</h2>
            <div style='
                display: flex;
                justify-content: space-between;
                color: #B0B0B0;
                margin-top: 10px;
            '>
                <span>{question[10]} {question[11]} | 👤 {question[10]}</span>
                <span>📅 {question[6]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Soru içeriği
        st.markdown(f"""
        <div style='
            padding: 20px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.03);
            margin-bottom: 20px;
        '>
            {question[4]}
        </div>
        """, unsafe_allow_html=True)
        
        # Etiketler
        if question[5]:  # tags
            st.markdown("**Etiketler:**")
            for tag in question[5].split():
                st.markdown(f"`{tag}`", unsafe_allow_html=True)
        
        # Oy durumu kontrolü
        user_vote = conn.execute('''
            SELECT vote_type FROM forum_votes 
            WHERE user_id = ? AND content_type = 'question' AND content_id = ?
        ''', (st.session_state.user_id, question_id)).fetchone()
        
        if user_vote:
            user_vote = user_vote[0]
        else:
            user_vote = 0  # Hiç oy vermemiş
        
        # Oy butonları ve sayıları
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**👍 {conn.execute('SELECT COUNT(*) FROM forum_votes WHERE content_type = \'question\' AND content_id = ? AND vote_type = 1', (question_id,)).fetchone()[0]}  👎 {conn.execute('SELECT COUNT(*) FROM forum_votes WHERE content_type = \'question\' AND content_id = ? AND vote_type = -1', (question_id,)).fetchone()[0]}**")
        
        with col2:
            if user_vote == 1:
                upvote_label = "👍 Beğenildi"
            else:
                upvote_label = "👍 Beğen"
            if st.button(upvote_label, key=f"detail_upvote_question_{question_id}"):
                vote_question(question_id, 1)
                st.rerun()
        
        with col3:
            if user_vote == -1:
                downvote_label = "👎 Beğenilmedi"
            else:
                downvote_label = "👎 Beğenme"
            if st.button(downvote_label, key=f"detail_downvote_question_{question_id}"):
                vote_question(question_id, -1)
                st.rerun()
        
        # Cevapları göster
        show_answers(question_id)
        
        # Cevap formu
        st.markdown("### 💭 Cevap Yaz")
        with st.form("answer_form"):
            answer_content = st.text_area(
                "Cevabınız",
                height=150,
                placeholder="Bu soruya cevabınızı yazın...",
                key="answer_content_textarea_detail"
            )
            
            submitted = st.form_submit_button("Cevabı Gönder")
            
            if submitted and answer_content:
                try:
                    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Cevabı kaydet
                    conn.execute('''
                        INSERT INTO forum_answers 
                        (question_id, user_id, content, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (question_id, st.session_state.user_id, answer_content, now, now))
                    
                    conn.commit()
                    
                    # XP kazandır
                    update_user_xp(st.session_state.user_id, calculate_xp_for_activity('forum_answer'))
                    
                    st.success("✅ Cevabınız başarıyla gönderildi!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Cevap gönderilirken bir hata oluştu: {str(e)}")
    
    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
    
    finally:
        conn.close()

# Cevapları Gösterme Fonksiyonu
def show_answers(question_id):
    """Soru cevaplarını göster"""
    conn = sqlite3.connect('motikoc.db')
    
    try:
        # Cevapları getir
        answers = conn.execute('''
            SELECT a.id, a.question_id, a.user_id, a.content, a.created_at, 
                   a.updated_at, a.is_accepted, a.upvotes, u.username,
                   (SELECT COUNT(*) FROM forum_votes WHERE content_type = 'answer' AND content_id = a.id AND vote_type = 1) as upvotes_count,
                   (SELECT COUNT(*) FROM forum_votes WHERE content_type = 'answer' AND content_id = a.id AND vote_type = -1) as downvotes_count
            FROM forum_answers a
            JOIN users u ON a.user_id = u.id
            WHERE a.question_id = ?
            ORDER BY a.is_accepted DESC, upvotes DESC, a.created_at ASC
        ''', (question_id,)).fetchall()
        
        st.markdown(f"### 💬 Cevaplar ({len(answers)})")
        
        if answers:
            for answer in answers:
                answer_id = answer[0]
                answer_user_id = answer[2]
                
                # Oy durumu kontrolü
                user_vote = conn.execute('''
                    SELECT vote_type FROM forum_votes 
                    WHERE user_id = ? AND content_type = 'answer' AND content_id = ?
                ''', (st.session_state.user_id, answer_id)).fetchone()
                
                if user_vote:
                    user_vote = user_vote[0]
                else:
                    user_vote = 0  # Hiç oy vermemiş
                
                with st.container():
                    # Kabul edilmiş cevap için özel stil
                    if answer[6]:  # is_accepted
                        container_style = "background-color: rgba(0, 255, 0, 0.1);"
                    else:
                        container_style = "background-color: rgba(255, 255, 255, 0.05);"
                    
                    st.markdown(f"""
                    <div style='padding: 15px; border-radius: 10px; margin: 10px 0; {container_style}'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span>👤 {answer[8]} | 📅 {answer[4]}</span>
                            <span>👍 {answer[9]} | 👎 {answer[10]}</span>
                        </div>
                        <div style='margin: 10px 0;'>{answer[3]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Oy butonları ve sayıları
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**👍 {answer[9]}  👎 {answer[10]}**")
                    
                    with col2:
                        # Upvote butonu
                        if user_vote == 1:
                            upvote_label = "👍 Beğenildi"
                        else:
                            upvote_label = "👍 Beğen"
                        if st.button(upvote_label, key=f"upvote_answer_{answer_id}"):
                            vote_answer(answer_id, 1)
                            st.rerun()
                    
                    with col3:
                        # Downvote butonu
                        if user_vote == -1:
                            downvote_label = "👎 Beğenilmedi"
                        else:
                            downvote_label = "👎 Beğenme"
                        if st.button(downvote_label, key=f"downvote_answer_{answer_id}"):
                            vote_answer(answer_id, -1)
                            st.rerun()
                    
                    # Etkileşim butonları
                    col1, col2, col3 = st.columns([1, 1, 2])
                    
                    with col1:
                        if st.button("👍 Beğen", key=f"answer_upvote_{answer_id}"):
                            vote_answer(answer_id, 1)
                            st.rerun()
                    
                    with col2:
                        if st.button("👎 Beğenme", key=f"answer_downvote_{answer_id}"):
                            vote_answer(answer_id, -1)
                            st.rerun()
                    
                    with col3:
                        # Sadece soru sahibi cevabı kabul edebilir
                        question_owner = conn.execute(
                            'SELECT user_id FROM forum_questions WHERE id = ?',
                            (question_id,)
                        ).fetchone()
                        
                        if (question_owner and 
                            question_owner[0] == st.session_state.user_id and 
                            not answer[6]):  # is_accepted
                            if st.button("✅ Doğru Cevap Olarak İşaretle", 
                                       key=f"accept_answer_{answer_id}"):
                                accept_answer(question_id, answer_id)
                                # Cevap sahibine XP ve bildirim
                                update_user_xp(answer_user_id, 
                                             calculate_xp_for_activity('answer_accepted'))
                                send_notification(
                                    answer_user_id,
                                    "Cevabınız doğru cevap olarak işaretlendi! 🎉",
                                    str(question_id)
                                )
                                st.rerun()
        else:
            st.info("Henüz cevap yok. İlk cevabı siz verin!")
    
    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
    
    finally:
        conn.close()

# Cevap Oy Fonksiyonu
def vote_answer(answer_id, vote_type):
    """Cevabı oyla"""
    conn = sqlite3.connect('motikoc.db')
    c = conn.cursor()
    
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Önceki oyu kontrol et
        existing_vote = c.execute('''
            SELECT id, vote_type 
            FROM forum_votes 
            WHERE user_id = ? AND content_type = 'answer' AND content_id = ?
        ''', (st.session_state.user_id, answer_id)).fetchone()
        
        if existing_vote:
            if existing_vote[1] == vote_type:
                # Aynı oy tipi - oyu kaldır
                c.execute('DELETE FROM forum_votes WHERE id = ?', (existing_vote[0],))
            else:
                # Farklı oy tipi - oyu güncelle
                c.execute('''
                    UPDATE forum_votes 
                    SET vote_type = ?, created_at = ?
                    WHERE id = ?
                ''', (vote_type, now, existing_vote[0]))
        else:
            # Yeni oy ekle
            c.execute('''
                INSERT INTO forum_votes 
                (user_id, content_type, content_id, vote_type, created_at)
                VALUES (?, 'answer', ?, ?, ?)
            ''', (st.session_state.user_id, answer_id, vote_type, now))
        
        conn.commit()
        
    except Exception as e:
        st.error(f"Oylama sırasında bir hata oluştu: {str(e)}")
    finally:
        conn.close()

# Soru Oylama Fonksiyonu
def vote_question(question_id, vote_type):
    """Soruyu oyla"""
    conn = sqlite3.connect('motikoc.db')
    c = conn.cursor()
    
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Önceki oyu kontrol et
        existing_vote = c.execute('''
            SELECT id, vote_type 
            FROM forum_votes 
            WHERE user_id = ? AND content_type = 'question' AND content_id = ?
        ''', (st.session_state.user_id, question_id)).fetchone()
        
        if existing_vote:
            if existing_vote[1] == vote_type:
                # Aynı oy tipi - oyu kaldır
                c.execute('DELETE FROM forum_votes WHERE id = ?', (existing_vote[0],))
            else:
                # Farklı oy tipi - oyu güncelle
                c.execute('''
                    UPDATE forum_votes 
                    SET vote_type = ?, created_at = ?
                    WHERE id = ?
                ''', (vote_type, now, existing_vote[0]))
        else:
            # Yeni oy ekle
            c.execute('''
                INSERT INTO forum_votes 
                (user_id, content_type, content_id, vote_type, created_at)
                VALUES (?, 'question', ?, ?, ?)
            ''', (st.session_state.user_id, question_id, vote_type, now))
        
        conn.commit()
        
    except Exception as e:
        st.error(f"Oylama sırasında bir hata oluştu: {str(e)}")
    finally:
        conn.close()

# Cevabı Kabul Etme Fonksiyonu
def accept_answer(question_id, answer_id):
    """Cevabı doğru cevap olarak işaretle"""
    conn = sqlite3.connect('motikoc.db')
    c = conn.cursor()
    
    try:
        # Diğer kabul edilmiş cevapları resetle
        c.execute('''
            UPDATE forum_answers 
            SET is_accepted = FALSE 
            WHERE question_id = ?
        ''', (question_id,))
        
        # Yeni cevabı kabul et
        c.execute('''
            UPDATE forum_answers 
            SET is_accepted = TRUE 
            WHERE id = ?
        ''', (answer_id,))
        
        # Soruyu çözüldü olarak işaretle
        c.execute('''
            UPDATE forum_questions 
            SET is_solved = TRUE 
            WHERE id = ?
        ''', (question_id,))
        
        conn.commit()
        
    except Exception as e:
        st.error(f"Cevap kabul edilirken bir hata oluştu: {str(e)}")
    finally:
        conn.close()

# Bildirim Gönderme Fonksiyonu
def send_notification(user_id, content, link=None):
    """Kullanıcıya bildirim gönder"""
    conn = sqlite3.connect('motikoc.db')
    c = conn.cursor()
    
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        c.execute('''
            INSERT INTO forum_notifications 
            (user_id, content, link, created_at)
            VALUES (?, ?, ?, ?)
        ''', (user_id, content, link, now))
        
        conn.commit()
        
    except Exception as e:
        print(f"Bildirim gönderilirken hata oluştu: {str(e)}")
    finally:
        conn.close()

# Kullanıcı Bildirimlerini Gösterme Fonksiyonu
def show_user_notifications():
    """Kullanıcı bildirimlerini göster"""
    conn = sqlite3.connect('motikoc.db')
    
    try:
        notifications = conn.execute('''
            SELECT * FROM forum_notifications
            WHERE user_id = ? AND is_read = FALSE
            ORDER BY created_at DESC
        ''', (st.session_state.user_id,)).fetchall()
        
        if notifications:
            st.sidebar.markdown("### 🔔 Bildirimler")
            
            for notif in notifications:
                with st.sidebar.container():
                    st.markdown(f"""
                    <div style='padding: 10px; border-radius: 5px; 
                              background-color: rgba(255, 255, 255, 0.05);'>
                        {notif[2]}
                        <br/>
                        <small>{notif[4]}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.sidebar.columns([1, 1])
                    
                    with col1:
                        if st.button("Okundu", key=f"read_notification_{notif[0]}"):
                            mark_notification_read(notif[0])
                            st.rerun()
                    
                    with col2:
                        if notif[3]:  # link varsa
                            if st.button("Git", key=f"goto_notification_{notif[0]}"):
                                # Soruya git
                                st.session_state.current_question = int(notif[3])
                                mark_notification_read(notif[0])
                                st.rerun()
    
    except Exception as e:
        st.error(f"Bildirimler yüklenirken bir hata oluştu: {str(e)}")
    
    finally:
        conn.close()

# Bildirimi Okundu Olarak İşaretleme Fonksiyonu
def mark_notification_read(notification_id):
    """Bildirimi okundu olarak işaretle"""
    conn = sqlite3.connect('motikoc.db')
    c = conn.cursor()
    
    try:
        c.execute('''
            UPDATE forum_notifications
            SET is_read = TRUE
            WHERE id = ? AND user_id = ?
        ''', (notification_id, st.session_state.user_id))
        
        conn.commit()
        
    except Exception as e:
        print(f"Bildirim güncellenirken hata oluştu: {str(e)}")
    finally:
        conn.close()

# Ana Menü Güncelleme Fonksiyonu
def update_main_menu():
    selected = option_menu(
        "MotiKoç Menü",
        ["Ana Sayfa", "YKS Paneli", "Çalışma Takvimi", 
         "Performans Analizi", "Sosyal Özellikler", "YKS Forumu",  # Forum menüsü eklendi
         "Sesli Rehber", "YKS Tercih Asistanı", "Kariyer Yol Haritası", "Ayarlar"],
        icons=['house', 'mortarboard', 'calendar3', 
               'graph-up', 'people', 'chat-dots',  # Forum için icon eklendi
               'volume-up', 'search', 'map', 'gear'],
        menu_icon="book",
        default_index=0,
        key="main_menu"
    )
    return selected

# Yeni Sayfa Fonksiyonları
def show_university_finder():
    interface = UniversityRecommenderInterface()
    interface.run()

def show_career_pathfinder():
    """Main application flow"""
    # Page configuration
    st.title("🎯 Kariyer Yol Bulucu Pro+")
    st.markdown("### 🤖 Yapay Zeka Destekli Profesyonel Kariyer Rehberi")

    # Initialize session state
    initialize_session_state()

    # Sidebar navigation and controls
    with st.sidebar:
        st.markdown("### 📍 Navigasyon")
        if st.session_state.current_section < len(SECTIONS):
            st.progress((st.session_state.current_section) / len(SECTIONS))
            st.write(f"Bölüm {st.session_state.current_section + 1}/{len(SECTIONS)}")
        
        # Settings
        st.markdown("### ⚙️ Ayarlar")
        st.session_state.show_tips = st.checkbox(
            "İpuçlarını Göster",
            value=st.session_state.show_tips
        )
        
        # Reset button
        if st.button("🔄 Yeniden Başla"):
            st.session_state.clear()
            st.rerun()

    # Main content area
    try:
        if st.session_state.current_section < len(SECTIONS):
            display_questions_section()
        else:
            if st.session_state.profile is None:
                if process_responses():
                    display_results()
            else:
                display_results()
    except Exception as e:
        st.error(f"Beklenmeyen bir hata oluştu: {str(e)}")
        if st.button("🔄 Tekrar Dene"):
            st.session_state.clear()
            st.rerun()

# Ana Uygulama Fonksiyonu
def main():
    # Initialize
    yks_features = add_yks_specific_features()
    yks_features['init_tables']()
    init_forum_tables()  # Forum tablolarını başlat
    
    # Initialize session state
    init_session_state()

    if st.session_state.user_id is None:
        show_login_page()
    else:
        selected = update_main_menu()

        if selected == "Ana Sayfa":
            show_home_page()
        elif selected == "YKS Paneli":
            yks_features['show_dashboard']()
        elif selected == "Çalışma Takvimi":
            show_study_calendar()
        elif selected == "Performans Analizi":
            show_performance_analytics()
        elif selected == "Sosyal Özellikler":
            show_social_features()
        elif selected == "YKS Forumu":  # Yeni forum seçeneği
            show_forum()
        elif selected == "Sesli Rehber":
            show_voice_guidance()
        elif selected == "YKS Tercih Asistanı":
            show_university_finder()
        elif selected == "Kariyer Yol Haritası":
            show_career_pathfinder()
        elif selected == "Ayarlar":
            show_settings()

        st.markdown("---")
        if st.button("Çıkış Yap"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Session State Initialization
def init_session_state():
    """Initialize all session state variables"""
    # Mevcut session state değişkenleri
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'current_plan' not in st.session_state:
        st.session_state.current_plan = None
    
    # Yeni eklenen session state değişkenleri
    if 'current_section' not in st.session_state:
        st.session_state.current_section = 0
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'profile' not in st.session_state:
        st.session_state.profile = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

# Uygulama Başlatma
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Uygulama çalıştırılırken bir hata oluştu.")
        st.error(f"Hata detayı: {str(e)}")
        if st.button("🔄 Uygulamayı Yeniden Başlat"):
            st.session_state.clear()
            st.rerun()