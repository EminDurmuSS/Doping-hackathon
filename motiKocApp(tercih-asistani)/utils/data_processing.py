# utils/data_processing.py
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Union
import re
from datetime import datetime, timedelta

def clean_and_parse_json(json_string: str) -> Optional[Dict[str, Any]]:
    """Clean and parse AI-generated JSON string with error handling."""
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

def process_study_logs(logs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Process study logs into a pandas DataFrame with calculated metrics."""
    try:
        df = pd.DataFrame(logs)
        if df.empty:
            return pd.DataFrame()
        
        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate additional metrics
        df['duration_hours'] = df['duration'] / 60
        df['week_number'] = df['date'].dt.isocalendar().week
        df['weekday'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month
        
        # Add efficiency score based on duration and performance
        df['efficiency_score'] = (df['duration'] * df['performance_rating']) / 100
        
        return df
    except Exception as e:
        print(f"Error processing study logs: {str(e)}")
        return pd.DataFrame()

def calculate_study_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive study statistics from DataFrame."""
    if df.empty:
        return {
            'total_hours': 0,
            'average_daily_hours': 0,
            'total_days': 0,
            'average_performance': 0,
            'subject_distribution': {},
            'weekly_trends': {}
        }
    
    try:
        stats = {
            'total_hours': df['duration_hours'].sum(),
            'average_daily_hours': df.groupby('date')['duration_hours'].sum().mean(),
            'total_days': df['date'].nunique(),
            'average_performance': df['performance_rating'].mean(),
            'subject_distribution': df.groupby('subject')['duration_hours'].sum().to_dict(),
            'weekly_trends': df.groupby('week_number')['duration_hours'].sum().to_dict()
        }
        
        return stats
    except Exception as e:
        print(f"Error calculating statistics: {str(e)}")
        return {}

def process_mock_exam_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process mock exam results and calculate trends."""
    try:
        df = pd.DataFrame(results)
        if df.empty:
            return {}
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate trends
        trends = {
            'overall_trend': calculate_trend(df['total_score']),
            'subject_trends': {
                subject: calculate_trend(df[f'{subject.lower()}_score'])
                for subject in df.columns if subject.endswith('_score')
            },
            'improvement_areas': identify_improvement_areas(df),
            'strong_subjects': identify_strong_subjects(df)
        }
        
        return trends
    except Exception as e:
        print(f"Error processing mock exam results: {str(e)}")
        return {}

def calculate_trend(series: pd.Series) -> Dict[str, float]:
    """Calculate trend metrics for a series of scores."""
    try:
        if len(series) < 2:
            return {'trend': 0, 'improvement': 0}
        
        trend = series.diff().mean()
        improvement = ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100
        
        return {
            'trend': float(trend),
            'improvement': float(improvement)
        }
    except Exception:
        return {'trend': 0, 'improvement': 0}

def identify_improvement_areas(df: pd.DataFrame) -> List[str]:
    """Identify subjects needing improvement based on recent scores."""
    try:
        recent_scores = df.tail(3)
        problem_areas = []
        
        for column in df.columns:
            if column.endswith('_score'):
                subject = column.replace('_score', '')
                avg_score = recent_scores[column].mean()
                if avg_score < 60:  # Threshold for improvement needed
                    problem_areas.append(subject)
        
        return problem_areas
    except Exception:
        return []

def identify_strong_subjects(df: pd.DataFrame) -> List[str]:
    """Identify subjects with consistently high scores."""
    try:
        recent_scores = df.tail(3)
        strong_subjects = []
        
        for column in df.columns:
            if column.endswith('_score'):
                subject = column.replace('_score', '')
                avg_score = recent_scores[column].mean()
                if avg_score >= 80:  # Threshold for strong performance
                    strong_subjects.append(subject)
        
        return strong_subjects
    except Exception:
        return []

def format_duration(minutes: int) -> str:
    """Format duration in minutes to human-readable string."""
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    if hours and remaining_minutes:
        return f"{hours} saat {remaining_minutes} dakika"
    elif hours:
        return f"{hours} saat"
    else:
        return f"{remaining_minutes} dakika"

def clean_text(text: str) -> str:
    """Clean and normalize text input."""
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters except Turkish characters
    text = re.sub(r'[^a-zA-ZğĞıİöÖüÜşŞçÇ0-9\s]', '', text)
    return text