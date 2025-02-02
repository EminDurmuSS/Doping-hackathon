# features/career/recommendations.py

import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime
from services.ai_service import AIService
import json

class CareerRecommendations:
    def __init__(self, user_id: int, db_conn):
        self.user_id = user_id
        self.db_conn = db_conn
        self.ai_service = AIService()

    def show(self):
        st.title("💼 Kariyer Önerileri")
        
        # Get recommended careers based on user profile
        recommendations = self._get_career_recommendations()
        
        if recommendations:
            self._show_recommendations(recommendations)
        else:
            st.info("Kariyer önerileri için lütfen önce Kariyer Yol Haritası bölümünü tamamlayın.")

    def _show_recommendations(self, recommendations: List[Dict[str, Any]]):
        """Display career recommendations"""
        st.markdown("### 🎯 Size Özel Kariyer Önerileri")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            min_match = st.slider(
                "Minimum Uyum Yüzdesi",
                min_value=0,
                max_value=100,
                value=70
            )
        
        with col2:
            categories = list(set(rec['category'] for rec in recommendations))
            selected_category = st.selectbox(
                "Kategori",
                ["Tümü"] + categories
            )
        
        # Filter recommendations
        filtered_recs = [
            rec for rec in recommendations
            if rec['match_percentage'] >= min_match and
            (selected_category == "Tümü" or rec['category'] == selected_category)
        ]
        
        # Display recommendations
        for rec in filtered_recs:
            with st.expander(f"💼 {rec['title']} - {rec['match_percentage']}% Uyum"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Kategori:** {rec['category']}")
                    st.markdown(f"**Açıklama:** {rec['description']}")
                    
                    st.markdown("**Gerekli Beceriler:**")
                    for skill in rec['required_skills']:
                        st.markdown(f"- {skill}")
                
                with col2:
                    st.markdown("**Eğitim:**")
                    for edu in rec['education']:
                        st.markdown(f"- {edu}")
                    
                    st.markdown("**Ortalama Maaş:**")
                    st.write(f"₺{rec['salary_range']['min']:,} - ₺{rec['salary_range']['max']:,}")
                
                # Job market info
                st.markdown("#### 📊 Sektör Bilgisi")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Büyüme Potansiyeli",
                        f"{rec['growth_potential']}%"
                    )
                    st.write(f"İş İlanı Sayısı: {rec['job_postings']:,}")
                
                with col2:
                    st.metric(
                        "Talep",
                        rec['demand_level']
                    )
                    st.write(f"Rekabet Seviyesi: {rec['competition_level']}")
                
                # Career path
                st.markdown("#### 📈 Kariyer Yolu")
                for i, step in enumerate(rec['career_path'], 1):
                    st.markdown(f"{i}. {step}")
                
                # Success factors
                st.markdown("#### 🌟 Başarı Faktörleri")
                for factor in rec['success_factors']:
                    st.markdown(f"- {factor}")

    def _get_career_recommendations(self) -> Optional[List[Dict[str, Any]]]:
        """Get career recommendations for user"""
        try:
            # Get user's profile and responses
            cursor = self.db_conn.cursor()
            
            # Get latest career assessment
            cursor.execute('''
                SELECT responses
                FROM career_responses
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (self.user_id,))
            
            assessment = cursor.fetchone()
            
            if not assessment or not assessment['responses']:
                return None
            
            # Get user profile
            cursor.execute('''
                SELECT grade, study_type, target_university, 
                       target_department, target_rank
                FROM users
                WHERE id = ?
            ''', (self.user_id,))
            
            profile = cursor.fetchone()
            
            if not profile:
                return None
            
            # Get study performance
            cursor.execute('''
                SELECT 
                    subject,
                    AVG(performance_rating) as avg_performance
                FROM study_logs
                WHERE user_id = ?
                GROUP BY subject
            ''', (self.user_id,))
            
            performance_rows = cursor.fetchall()
            performance = {
                row['subject']: row['avg_performance']
                for row in performance_rows
            }
            
            # Generate recommendations using AI service
            recommendations = self.ai_service.generate_career_recommendations({
                'assessment': json.loads(assessment['responses']),
                'profile': {
                    'grade': profile['grade'],
                    'study_type': profile['study_type'],
                    'target_university': profile['target_university'],
                    'target_department': profile['target_department'],
                    'target_rank': profile['target_rank']
                },
                'performance': performance
            })
            
            return recommendations
                
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return None
