# config/settings.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Settings
DATABASE_NAME = 'motikoc.db'
DB_TIMEOUT = 30
DB_PRAGMAS = {
    'journal_mode': 'WAL',
    'cache_size': -1 * 64000,  # 64MB
    'foreign_keys': 1,
    'ignore_check_constraints': 0,
    'synchronous': 1
}

# API Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-pro-002"

# Application Settings
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# Streamlit Settings
STREAMLIT_CONFIG = {
    'page_title': "MotiKoç - YKS Hazırlık Asistanı",
    'page_icon': "🎯",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Cache Settings
CACHE_TTL = 3600  # 1 hour
CACHE_TYPE = "filesystem"
CACHE_DIR = ".streamlit/cache"

# Security Settings
PASSWORD_MIN_LENGTH = 8
SESSION_EXPIRY = 3600  # 1 hour
MAX_LOGIN_ATTEMPTS = 5
LOGIN_COOLDOWN = 300  # 5 minutes

# File Upload Settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB

# Pagination Settings
ITEMS_PER_PAGE = 20
MAX_PAGES = 100

# Performance Settings
MAX_CONCURRENT_USERS = 100
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

# Forum Settings
FORUM_CONFIG = {
    'max_title_length': 100,
    'min_content_length': 10,
    'max_tags': 5,
    'cooldown_between_posts': 300,  # 5 minutes
    'max_daily_posts': 10
}

# Gamification Settings
GAMIFICATION_CONFIG = {
    'xp_per_study_minute': 1,
    'xp_per_question': 10,
    'xp_per_answer': 15,
    'daily_goal_bonus': 50,
    'streak_bonus': 100,
    'level_multiplier': 1000,  # XP needed per level = level * multiplier
}

# Notification Settings
NOTIFICATION_CONFIG = {
    'enabled': True,
    'check_interval': 300,  # 5 minutes
    'max_notifications': 50,
    'cleanup_after': 7  # days
}

# Study Calendar Settings
CALENDAR_CONFIG = {
    'min_session_duration': 15,  # minutes
    'max_session_duration': 180,  # minutes
    'break_duration': 5,  # minutes
    'long_break_duration': 15,  # minutes
    'sessions_until_long_break': 4
}

# Custom Theme Settings
THEME = {
    'primary_color': '#4F46E5',
    'secondary_color': '#7C3AED',
    'background_color': '#1A1A1A',
    'secondary_background': '#2D2D2D',
    'text_color': '#FFFFFF',
    'accent_color': '#10B981',
    'error_color': '#EF4444',
    'warning_color': '#F59E0B'
}

# Feature Flags
FEATURES = {
    'forum_enabled': True,
    'voice_guidance_enabled': True,
    'career_pathfinder_enabled': True,
    'university_finder_enabled': True,
    'social_features_enabled': True,
    'gamification_enabled': True,
    'notifications_enabled': True
}