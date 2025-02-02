# config/constants.py
from datetime import datetime

# Application Constants
APP_NAME = "MotiKoç - YKS Hazırlık Asistanı"
APP_VERSION = "1.0.0"
APP_ICON = "🎯"

# YKS Exam Constants
YKS_EXAM_DATE = datetime(2025, 6, 15)
YKS_EXAM_TYPES = ["TYT", "AYT"]

# Study Related Constants
STUDY_SUBJECTS = {
    'TYT': [
        'Türkçe',
        'Matematik',
        'Fizik',
        'Kimya',
        'Biyoloji',
        'Tarih',
        'Coğrafya',
        'Felsefe',
        'Din Kültürü'
    ],
    'AYT': {
        'Sayısal': ['Matematik', 'Fizik', 'Kimya', 'Biyoloji'],
        'Eşit Ağırlık': ['Matematik', 'Edebiyat', 'Tarih', 'Coğrafya'],
        'Sözel': ['Edebiyat', 'Tarih', 'Coğrafya', 'Felsefe'],
        'Dil': ['İngilizce']
    }
}

# Badge Related Constants
BADGE_TYPES = {
    'study_sessions': 'Çalışma Seansı',
    'daily_hours': 'Günlük Çalışma',
    'streak_days': 'Kesintisiz Çalışma',
    'mock_exams': 'Deneme Sınavı',
    'social_interaction': 'Sosyal Etkileşim',
}

# XP Related Constants
XP_REWARDS = {
    'study_session': 50,
    'study_minute': 1,
    'performance_bonus': 20,
    'streak_bonus': 100,
    'badge_earned': 200,
    'achievement_shared': 150,
    'forum_question': 75,
    'forum_answer': 100,
    'answer_accepted': 200,
    'upvote_received': 10,
    'answer_upvoted': 5
}

# Navigation Constants
MENU_ITEMS = [
    {"name": "Ana Sayfa", "icon": "house"},
    {"name": "YKS Paneli", "icon": "mortarboard"},
    {"name": "Çalışma Takvimi", "icon": "calendar3"},
    {"name": "Performans Analizi", "icon": "graph-up"},
    {"name": "Sosyal Özellikler", "icon": "people"},
    {"name": "YKS Forumu", "icon": "chat-dots"},
    {"name": "Sesli Rehber", "icon": "volume-up"},
    {"name": "YKS Tercih Asistanı", "icon": "search"},
    {"name": "Kariyer Yol Haritası", "icon": "map"},
    {"name": "Ayarlar", "icon": "gear"}
]

# Forum Categories
FORUM_CATEGORIES = [
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

# Error Messages
ERROR_MESSAGES = {
    'db_connection': 'Veritabanı bağlantısı kurulamadı.',
    'login_failed': 'Giriş başarısız. Kullanıcı adı veya şifre hatalı.',
    'registration_failed': 'Kayıt işlemi başarısız.',
    'invalid_input': 'Geçersiz giriş. Lütfen tüm alanları kontrol edin.',
    'server_error': 'Sunucu hatası. Lütfen daha sonra tekrar deneyin.',
}

# Success Messages
SUCCESS_MESSAGES = {
    'login_success': 'Giriş başarılı!',
    'registration_success': 'Kayıt başarılı! Giriş yapabilirsiniz.',
    'study_session_added': 'Çalışma seansı başarıyla eklendi!',
    'badge_earned': 'Yeni rozet kazandınız!',
}

# config/constants.py

# Add this to your existing constants
DEFAULT_BADGES = [
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


# Default subjects for YKS preparation
DEFAULT_SUBJECTS = [
    ('Türkçe', 'TYT'),
    ('Matematik', 'TYT'),
    ('Fizik', 'AYT'),
    ('Kimya', 'AYT'),
    ('Biyoloji', 'AYT'),
    ('Tarih', 'TYT'),
    ('Coğrafya', 'TYT'),
    ('Felsefe', 'TYT')
]


# YKS subject categories
YKS_CATEGORIES = {
    'TYT': [
        'Türkçe',
        'Temel Matematik',
        'Fen Bilimleri',
        'Sosyal Bilimler'
    ],
    'AYT': {
        'Sayısal': [
            'Matematik',
            'Fizik',
            'Kimya',
            'Biyoloji'
        ],
        'Eşit Ağırlık': [
            'Matematik',
            'Edebiyat',
            'Tarih',
            'Coğrafya'
        ],
        'Sözel': [
            'Edebiyat',
            'Tarih',
            'Coğrafya',
            'Felsefe'
        ]
    }
}

# Study session types
STUDY_TYPES = [
    'Konu Çalışması',
    'Soru Çözümü',
    'Tekrar',
    'Deneme Sınavı',
    'Ödev'
]

# Performance rating scale
PERFORMANCE_RATINGS = {
    1: 'Çok Kötü',
    2: 'Kötü',
    3: 'Orta',
    4: 'İyi',
    5: 'Çok İyi'
}

# Mood types
MOOD_TYPES = [
    'Çok İyi',
    'İyi',
    'Normal',
    'Kötü',
    'Çok Kötü'
]

# Stress levels
STRESS_LEVELS = {
    1: 'Çok Düşük',
    2: 'Düşük',
    3: 'Orta',
    4: 'Yüksek',
    5: 'Çok Yüksek'
}

# Task types
TASK_TYPES = [
    'study',
    'social',
    'motivation',
    'practice',
    'review'
]

# Competition types
COMPETITION_TYPES = [
    'Deneme Sınavı',
    'Soru Çözme',
    'Çalışma Süresi',
    'Haftalık Hedef'
]

# Study group types
STUDY_GROUP_TYPES = [
    'TYT Hazırlık',
    'AYT Sayısal',
    'AYT Eşit Ağırlık',
    'AYT Sözel',
    'Genel Çalışma'
]

# Question difficulty levels
DIFFICULTY_LEVELS = {
    1: 'Çok Kolay',
    2: 'Kolay',
    3: 'Orta',
    4: 'Zor',
    5: 'Çok Zor'
}



YKS_CATEGORIES = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'Turkish', 'History', 'Geography', 'English']
STUDY_TYPES = ['Sayısal', 'Eşit Ağırlık', 'Sözel', 'Dil']
PERFORMANCE_RATINGS = [1, 2, 3, 4, 5]
MOOD_TYPES = ['Mutlu', 'Hüzünlü', 'Stresli', 'Motivasyonlu']
STRESS_LEVELS = [1, 2, 3, 4, 5]
TASK_TYPES = ['study', 'practice', 'review', 'social']
COMPETITION_TYPES = ['quiz', 'challenge']
STUDY_GROUP_TYPES = ['Private', 'Public']
DIFFICULTY_LEVELS = [1, 2, 3, 4, 5]





# config/constants.py

from datetime import datetime

# Application Constants
APP_NAME = "MotiKoç - YKS Hazırlık Asistanı"
APP_VERSION = "1.0.0"
APP_ICON = "🎯"

# YKS Exam Constants
YKS_EXAM_DATE = datetime(2025, 6, 15)
YKS_EXAM_TYPES = ["TYT", "AYT"]

# Study Related Constants
STUDY_SUBJECTS = {
    'TYT': [
        'Türkçe',
        'Matematik',
        'Fizik',
        'Kimya',
        'Biyoloji',
        'Tarih',
        'Coğrafya',
        'Felsefe',
        'Din Kültürü'
    ],
    'AYT': {
        'Sayısal': ['Matematik', 'Fizik', 'Kimya', 'Biyoloji'],
        'Eşit Ağırlık': ['Matematik', 'Edebiyat', 'Tarih', 'Coğrafya'],
        'Sözel': ['Edebiyat', 'Tarih', 'Coğrafya', 'Felsefe'],
        'Dil': ['İngilizce']
    }
}

# YKS Subject Categories (Dictionary)
YKS_CATEGORIES = {
    'TYT': [
        'Türkçe',
        'Temel Matematik',
        'Fen Bilimleri',
        'Sosyal Bilimler'
    ],
    'AYT': {
        'Sayısal': [
            'Matematik',
            'Fizik',
            'Kimya',
            'Biyoloji'
        ],
        'Eşit Ağırlık': [
            'Matematik',
            'Edebiyat',
            'Tarih',
            'Coğrafya'
        ],
        'Sözel': [
            'Edebiyat',
            'Tarih',
            'Coğrafya',
            'Felsefe'
        ]
    }
}

# YKS Categories List (Separate Variable)
YKS_CATEGORIES_LIST = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'Turkish', 'History', 'Geography', 'English']

# Badge Related Constants
BADGE_TYPES = {
    'study_sessions': 'Çalışma Seansı',
    'daily_hours': 'Günlük Çalışma',
    'streak_days': 'Kesintisiz Çalışma',
    'mock_exams': 'Deneme Sınavı',
    'social_interaction': 'Sosyal Etkileşim',
}

# XP Related Constants
XP_REWARDS = {
    'study_session': 50,
    'study_minute': 1,
    'performance_bonus': 20,
    'streak_bonus': 100,
    'badge_earned': 200,
    'achievement_shared': 150,
    'forum_question': 75,
    'forum_answer': 100,
    'answer_accepted': 200,
    'upvote_received': 10,
    'answer_upvoted': 5
}

# Navigation Constants
MENU_ITEMS = [
    {"name": "Ana Sayfa", "icon": "house"},
    {"name": "YKS Paneli", "icon": "mortarboard"},
    {"name": "Çalışma Takvimi", "icon": "calendar3"},
    {"name": "Performans Analizi", "icon": "graph-up"},
    {"name": "Sosyal Özellikler", "icon": "people"},
    {"name": "YKS Forumu", "icon": "chat-dots"},
    {"name": "Sesli Rehber", "icon": "volume-up"},
    {"name": "YKS Tercih Asistanı", "icon": "search"},
    {"name": "Kariyer Yol Haritası", "icon": "map"},
    {"name": "Ayarlar", "icon": "gear"}
]

# Forum Categories
FORUM_CATEGORIES = [
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

# Error Messages
ERROR_MESSAGES = {
    'db_connection': 'Veritabanı bağlantısı kurulamadı.',
    'login_failed': 'Giriş başarısız. Kullanıcı adı veya şifre hatalı.',
    'registration_failed': 'Kayıt işlemi başarısız.',
    'invalid_input': 'Geçersiz giriş. Lütfen tüm alanları kontrol edin.',
    'server_error': 'Sunucu hatası. Lütfen daha sonra tekrar deneyin.',
}

# Success Messages
SUCCESS_MESSAGES = {
    'login_success': 'Giriş başarılı!',
    'registration_success': 'Kayıt başarılı! Giriş yapabilirsiniz.',
    'study_session_added': 'Çalışma seansı başarıyla eklendi!',
    'badge_earned': 'Yeni rozet kazandınız!',
}

# Default Badges (Defined Once)
DEFAULT_BADGES = [
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

# Study Session Types
STUDY_TYPES = [
    'Konu Çalışması',
    'Soru Çözümü',
    'Tekrar',
    'Deneme Sınavı',
    'Ödev'
]

# Performance Rating Scale (Keep as Dict for Detailed Labels)
PERFORMANCE_RATINGS_DICT = {
    1: 'Çok Kötü',
    2: 'Kötü',
    3: 'Orta',
    4: 'İyi',
    5: 'Çok İyi'
}

# Performance Ratings List (for Type Hints or Simpler Usage)
PERFORMANCE_RATINGS_LIST = [1, 2, 3, 4, 5]

# Mood Types
MOOD_TYPES_DICT = [
    'Çok İyi',
    'İyi',
    'Normal',
    'Kötü',
    'Çok Kötü'
]

# Simplified Mood Types List
MOOD_TYPES_LIST = ['Mutlu', 'Hüzünlü', 'Stresli', 'Motivasyonlu']

# Stress Levels
STRESS_LEVELS_DICT = {
    1: 'Çok Düşük',
    2: 'Düşük',
    3: 'Orta',
    4: 'Yüksek',
    5: 'Çok Yüksek'
}

# Stress Levels List
STRESS_LEVELS_LIST = [1, 2, 3, 4, 5]

# Task Types
TASK_TYPES_DICT = [
    'study',
    'social',
    'motivation',
    'practice',
    'review'
]

# Competition Types
COMPETITION_TYPES_DICT = [
    'Deneme Sınavı',
    'Soru Çözme',
    'Çalışma Süresi',
    'Haftalık Hedef'
]

# Simplified Competition Types List
COMPETITION_TYPES_LIST = ['quiz', 'challenge']

# Study Group Types
STUDY_GROUP_TYPES_DICT = [
    'TYT Hazırlık',
    'AYT Sayısal',
    'AYT Eşit Ağırlık',
    'AYT Sözel',
    'Genel Çalışma'
]

# Simplified Study Group Types List
STUDY_GROUP_TYPES_LIST = ['Private', 'Public']

# Question Difficulty Levels
DIFFICULTY_LEVELS_DICT = {
    1: 'Çok Kolay',
    2: 'Kolay',
    3: 'Orta',
    4: 'Zor',
    5: 'Çok Zor'
}

# Difficulty Levels List
DIFFICULTY_LEVELS_LIST = [1, 2, 3, 4, 5]
