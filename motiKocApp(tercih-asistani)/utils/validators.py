# utils/validators.py
import re
from typing import Optional, Dict, Any, List
from datetime import datetime
import hashlib

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password(password: str) -> Dict[str, Any]:
    """
    Validate password strength.
    Returns dict with success status and validation messages.
    """
    min_length = 8
    validation = {
        'valid': True,
        'messages': [],
        'score': 0  # Password strength score (0-5)
    }
    
    # Length check
    if len(password) < min_length:
        validation['messages'].append(
            f'Şifre en az {min_length} karakter olmalıdır.'
        )
        validation['valid'] = False
    else:
        validation['score'] += 1
    
    # Uppercase check
    if not re.search(r'[A-Z]', password):
        validation['messages'].append(
            'Şifre en az bir büyük harf içermelidir.'
        )
        validation['valid'] = False
    else:
        validation['score'] += 1
    
    # Lowercase check
    if not re.search(r'[a-z]', password):
        validation['messages'].append(
            'Şifre en az bir küçük harf içermelidir.'
        )
        validation['valid'] = False
    else:
        validation['score'] += 1
    
    # Number check
    if not re.search(r'\d', password):
        validation['messages'].append(
            'Şifre en az bir rakam içermelidir.'
        )
        validation['valid'] = False
    else:
        validation['score'] += 1
    
    # Special character check
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        validation['messages'].append(
            'Şifre en az bir özel karakter içermelidir.'
        )
        validation['valid'] = False
    else:
        validation['score'] += 1
    
    return validation

def validate_username(username: str) -> Dict[str, Any]:
    """Validate username format."""
    validation = {
        'valid': True,
        'messages': []
    }
    
    # Length check
    if len(username) < 3 or len(username) > 20:
        validation['messages'].append(
            'Kullanıcı adı 3-20 karakter arasında olmalıdır.'
        )
        validation['valid'] = False
    
    # Character check
    if not re.match(r'^[a-zA-Z0-9_.-]+$', username):
        validation['messages'].append(
            'Kullanıcı adı sadece harf, rakam ve _.- karakterlerini içerebilir.'
        )
        validation['valid'] = False
    
    return validation

def validate_study_log(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate study log data."""
    validation = {
        'valid': True,
        'messages': []
    }
    
    # Required fields check
    required_fields = ['subject', 'duration', 'date']
    for field in required_fields:
        if field not in log_data or not log_data[field]:
            validation['messages'].append(
                f'{field} alanı zorunludur.'
            )
            validation['valid'] = False
    
    # Duration check
    if 'duration' in log_data:
        duration = log_data['duration']
        if not isinstance(duration, int) or duration < 1:
            validation['messages'].append(
                'Süre pozitif bir sayı olmalıdır.'
            )
            validation['valid'] = False
    
    # Date format check
    if 'date' in log_data:
        try:
            datetime.strptime(log_data['date'], '%Y-%m-%d')
        except ValueError:
            validation['messages'].append(
                'Geçersiz tarih formatı. (YYYY-MM-DD)'
            )
            validation['valid'] = False
    
    return validation

def validate_mock_exam(exam_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate mock exam data."""
    validation = {
        'valid': True,
        'messages': []
    }
    
    required_fields = [
        'exam_type',
        'total_time',
        'subject_results'
    ]
    
    # Required fields check
    for field in required_fields:
        if field not in exam_data or not exam_data[field]:
            validation['messages'].append(
                f'{field} alanı zorunludur.'
            )
            validation['valid'] = False
            # utils/validators.py (continued)

    # Exam type check
    if 'exam_type' in exam_data:
        valid_types = ['TYT', 'AYT']
        if exam_data['exam_type'] not in valid_types:
            validation['messages'].append(
                'Geçersiz sınav türü. (TYT veya AYT olmalıdır)'
            )
            validation['valid'] = False
    
    # Time check
    if 'total_time' in exam_data:
        total_time = exam_data['total_time']
        if not isinstance(total_time, int) or total_time < 1:
            validation['messages'].append(
                'Geçersiz süre değeri.'
            )
            validation['valid'] = False
    
    # Subject results check
    if 'subject_results' in exam_data:
        results = exam_data['subject_results']
        if not isinstance(results, dict):
            validation['messages'].append(
                'Ders sonuçları geçersiz format.'
            )
            validation['valid'] = False
        else:
            for subject, data in results.items():
                if not all(key in data for key in ['correct', 'incorrect', 'empty']):
                    validation['messages'].append(
                        f'{subject} için eksik sonuç bilgisi.'
                    )
                    validation['valid'] = False
    
    return validation

def validate_profile_data(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate user profile data."""
    validation = {
        'valid': True,
        'messages': []
    }
    
    # Required fields check
    required_fields = {
        'name': 'İsim',
        'email': 'E-posta',
        'grade': 'Sınıf',
        'study_type': 'Çalışma Türü'
    }
    
    for field, label in required_fields.items():
        if field not in profile_data or not profile_data[field]:
            validation['messages'].append(
                f'{label} alanı zorunludur.'
            )
            validation['valid'] = False
    
    # Email validation
    if 'email' in profile_data:
        if not validate_email(profile_data['email']):
            validation['messages'].append(
                'Geçersiz e-posta adresi.'
            )
            validation['valid'] = False
    
    # Grade validation
    if 'grade' in profile_data:
        valid_grades = ['9', '10', '11', '12', 'Mezun']
        if profile_data['grade'] not in valid_grades:
            validation['messages'].append(
                'Geçersiz sınıf bilgisi.'
            )
            validation['valid'] = False
    
    # Study type validation
    if 'study_type' in profile_data:
        valid_types = ['Sayısal', 'Eşit Ağırlık', 'Sözel', 'Dil']
        if profile_data['study_type'] not in valid_types:
            validation['messages'].append(
                'Geçersiz çalışma türü.'
            )
            validation['valid'] = False
    
    return validation

def validate_goal_data(goal_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate goal data."""
    validation = {
        'valid': True,
        'messages': []
    }
    
    # Required fields check
    required_fields = ['title', 'deadline']
    for field in required_fields:
        if field not in goal_data or not goal_data[field]:
            validation['messages'].append(
                f'{field} alanı zorunludur.'
            )
            validation['valid'] = False
    
    # Deadline validation
    if 'deadline' in goal_data:
        try:
            deadline = datetime.strptime(goal_data['deadline'], '%Y-%m-%d')
            if deadline.date() < datetime.now().date():
                validation['messages'].append(
                    'Hedef tarihi geçmiş bir tarih olamaz.'
                )
                validation['valid'] = False
        except ValueError:
            validation['messages'].append(
                'Geçersiz tarih formatı. (YYYY-MM-DD)'
            )
            validation['valid'] = False
    
    return validation

def validate_file_upload(file_data: Any, 
                        allowed_extensions: List[str], 
                        max_size_mb: int = 5) -> Dict[str, Any]:
    """Validate file upload."""
    validation = {
        'valid': True,
        'messages': []
    }
    
    if not file_data:
        validation['messages'].append('Dosya seçilmedi.')
        validation['valid'] = False
        return validation
    
    # File extension check
    file_extension = file_data.name.split('.')[-1].lower()
    if file_extension not in allowed_extensions:
        validation['messages'].append(
            f'Geçersiz dosya türü. İzin verilen türler: {", ".join(allowed_extensions)}'
        )
        validation['valid'] = False
    
    # File size check
    if hasattr(file_data, 'size'):
        file_size_mb = file_data.size / (1024 * 1024)  # Convert to MB
        if file_size_mb > max_size_mb:
            validation['messages'].append(
                f'Dosya boyutu çok büyük. Maksimum: {max_size_mb}MB'
            )
            validation['valid'] = False
    
    return validation

def generate_hash(text: str) -> str:
    """Generate SHA-256 hash of text."""
    return hashlib.sha256(text.encode()).hexdigest()

def validate_date_range(start_date: str, end_date: str) -> Dict[str, Any]:
    """Validate date range."""
    validation = {
        'valid': True,
        'messages': []
    }
    
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if end < start:
            validation['messages'].append(
                'Bitiş tarihi başlangıç tarihinden önce olamaz.'
            )
            validation['valid'] = False
        
        if start.date() < datetime.now().date():
            validation['messages'].append(
                'Başlangıç tarihi geçmiş bir tarih olamaz.'
            )
            validation['valid'] = False
            
    except ValueError:
        validation['messages'].append(
            'Geçersiz tarih formatı. (YYYY-MM-DD)'
        )
        validation['valid'] = False
    
    return validation

def validate_notification_data(notification_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate notification data."""
    validation = {
        'valid': True,
        'messages': []
    }
    
    required_fields = ['title', 'content']
    for field in required_fields:
        if field not in notification_data or not notification_data[field]:
            validation['messages'].append(
                f'{field} alanı zorunludur.'
            )
            validation['valid'] = False
    
    if 'title' in notification_data:
        if len(notification_data['title']) > 100:
            validation['messages'].append(
                'Başlık çok uzun. (Maksimum 100 karakter)'
            )
            validation['valid'] = False
    
    if 'content' in notification_data:
        if len(notification_data['content']) > 500:
            validation['messages'].append(
                'İçerik çok uzun. (Maksimum 500 karakter)'
            )
            validation['valid'] = False
    
    return validation

def validate_settings_data(settings_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate user settings data."""
    validation = {
        'valid': True,
        'messages': []
    }
    
    valid_themes = ['light', 'dark', 'system']
    if 'theme' in settings_data and settings_data['theme'] not in valid_themes:
        validation['messages'].append(
            'Geçersiz tema seçimi.'
        )
        validation['valid'] = False
    
    if 'notification_preferences' in settings_data:
        prefs = settings_data['notification_preferences']
        if not isinstance(prefs, dict):
            validation['messages'].append(
                'Geçersiz bildirim tercihleri formatı.'
            )
            validation['valid'] = False
    
    if 'study_reminder_time' in settings_data:
        try:
            datetime.strptime(settings_data['study_reminder_time'], '%H:%M')
        except ValueError:
            validation['messages'].append(
                'Geçersiz hatırlatıcı saat formatı. (HH:MM)'
            )
            validation['valid'] = False
    
    return validation