import streamlit as st
from typing import Dict, List, Any, Optional, cast
from datetime import datetime
import logging
import plotly.graph_objects as go
import json
from core.database import DatabaseManager, DatabaseError
from services.ai_service import AIService

# Configure logging
logger = logging.getLogger(__name__)

class CareerPathFinder:
    def __init__(self, user_id: int):
        """Initialize CareerPathFinder with user_id"""
        self.user_id = user_id
        self.db = DatabaseManager()
        self.ai_service = AIService()

    def show(self):
        """Display career pathfinder interface"""
        st.title("🎯 Kariyer Yol Haritası")
        
        try:
            # Get user's profile and responses
            profile_data = self._get_user_profile()
            responses = self._get_user_responses()
            
            if not responses:
                self._show_questionnaire()
            else:
                self._show_career_analysis(profile_data, responses)
        except Exception as e:
            logger.error(f"Error displaying career pathfinder: {e}")
            st.error("Bir hata oluştu. Lütfen daha sonra tekrar deneyin.")

    def _show_questionnaire(self):
        """Display career questionnaire with error handling"""
        st.markdown("### 📝 Kariyer Keşif Anketi")
        
        try:
            with st.form("career_questionnaire"):
                # Academic Interests
                st.markdown("#### 📚 Akademik İlgi Alanları")
                
                favorite_subjects = st.multiselect(
                    "En sevdiğiniz dersler nelerdir?",
                    self._get_subjects()
                )
                
                academic_strength = st.text_area(
                    "Akademik olarak kendinizi hangi alanlarda güçlü görüyorsunuz?"
                )
                
                # Skills and Abilities
                st.markdown("#### 💪 Beceriler ve Yetenekler")
                
                skills = st.multiselect(
                    "Hangi becerilere sahipsiniz?",
                    [
                        "Problem Çözme",
                        "Analitik Düşünme",
                        "İletişim",
                        "Liderlik",
                        "Yaratıcılık",
                        "Takım Çalışması",
                        "Organizasyon",
                        "Teknoloji Kullanımı"
                    ]
                )
                
                strengths = st.text_area(
                    "En güçlü yönleriniz nelerdir?"
                )
                
                # Interests and Values
                st.markdown("#### 🎯 İlgi Alanları ve Değerler")
                
                interests = st.multiselect(
                    "İlgi alanlarınız nelerdir?",
                    [
                        "Bilim ve Teknoloji",
                        "Sanat ve Tasarım",
                        "İş ve Yönetim",
                        "Sağlık",
                        "Eğitim",
                        "Sosyal Bilimler",
                        "Mühendislik",
                        "Hukuk"
                    ]
                )
                
                work_values = st.multiselect(
                    "İş hayatında sizin için önemli olan değerler nelerdir?",
                    [
                        "İş-Yaşam Dengesi",
                        "Yüksek Gelir",
                        "Kariyer Gelişimi",
                        "Topluma Fayda",
                        "Yaratıcılık",
                        "Bağımsızlık",
                        "Güvenlik",
                        "Prestij"
                    ]
                )
                
                # Future Goals
                st.markdown("#### 🌟 Gelecek Hedefleri")
                
                career_goals = st.text_area(
                    "Kariyer hedefleriniz nelerdir?"
                )
                
                work_environment = st.multiselect(
                    "Nasıl bir çalışma ortamı tercih edersiniz?",
                    [
                        "Ofis Ortamı",
                        "Uzaktan Çalışma",
                        "Sahada Çalışma",
                        "Esnek Çalışma",
                        "Takım Çalışması",
                        "Bireysel Çalışma"
                    ]
                )
                
                submitted = st.form_submit_button("Analiz Et")
                
                if submitted:
                    self._handle_questionnaire_submission(
                        favorite_subjects, academic_strength, skills,
                        strengths, interests, work_values, career_goals,
                        work_environment
                    )

        except Exception as e:
            logger.error(f"Error displaying questionnaire: {e}")
            st.error("Anket yüklenirken bir hata oluştu.")

    def _handle_questionnaire_submission(
        self, 
        favorite_subjects: List[str],
        academic_strength: str,
        skills: List[str],
        strengths: str,
        interests: List[str],
        work_values: List[str],
        career_goals: str,
        work_environment: List[str]
    ):
        """Handle questionnaire submission with validation"""
        try:
            # Create responses dictionary
            responses = {
                'favorite_subjects': favorite_subjects,
                'academic_strength': academic_strength,
                'skills': skills,
                'strengths': strengths,
                'interests': interests,
                'work_values': work_values,
                'career_goals': career_goals,
                'work_environment': work_environment
            }

            # Validate required fields
            if not all([favorite_subjects, skills, interests, work_values]):
                st.error("Lütfen tüm gerekli alanları doldurun.")
                return

            if self._save_responses(responses):
                st.success("Yanıtlarınız kaydedildi! Analiz hazırlanıyor...")
                st.rerun()
            else:
                st.error("Yanıtlar kaydedilirken bir hata oluştu!")

        except Exception as e:
            logger.error(f"Error handling questionnaire submission: {e}")
            st.error("Yanıtlar kaydedilirken bir hata oluştu.")

    def _show_career_analysis(self, profile_data: Dict[str, Any], responses: Dict[str, Any]):
        """Display career analysis results with proper error handling"""
        st.markdown("### 📊 Kariyer Analizi")
        
        try:
            # Get AI analysis
            analysis = self.ai_service.analyze_career_responses(responses)
            
            if not analysis:
                st.error("Analiz oluşturulurken bir hata oluştu!")
                return
            
            # Display results
            self._display_personality_profile(analysis)
            self._display_career_recommendations(analysis)
            self._display_education_recommendations(analysis)
            self._display_next_steps(analysis)
            
            # Update responses button
            if st.button("Yanıtları Güncelle"):
                if self._clear_responses():
                    st.success("Yanıtlarınız silindi. Lütfen yanıtlarınızı güncelleyin.")
                    st.rerun()
                else:
                    st.error("Yanıtlar silinirken bir hata oluştu!")

        except Exception as e:
            logger.error(f"Error displaying career analysis: {e}")
            st.error("Analiz gösterilirken bir hata oluştu.")

    def _display_personality_profile(self, analysis: Dict[str, Any]):
        """Display personality profile section"""
        st.markdown("#### 👤 Kişilik Profili")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Personality traits radar chart
            fig = self._create_personality_chart(analysis.get('personality_traits', {}))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Key characteristics
            st.markdown("**Güçlü Yönler**")
            for strength in analysis.get('strengths', []):
                st.markdown(f"- {strength}")
            
            st.markdown("**Gelişim Alanları**")
            for area in analysis.get('development_areas', []):
                st.markdown(f"- {area}")

    def _display_career_recommendations(self, analysis: Dict[str, Any]):
        """Display career recommendations section"""
        st.markdown("#### 💼 Kariyer Önerileri")
        
        for career in analysis.get('recommended_careers', []):
            with st.expander(f"🎯 {career.get('title', 'Belirsiz Meslek')}"):
                st.markdown(f"**Uyum Oranı:** {career.get('match_percentage', 0)}%")
                st.markdown(f"**Açıklama:** {career.get('description', 'Açıklama yok.')}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Gerekli Beceriler:**")
                    for skill in career.get('required_skills', []):
                        st.markdown(f"- {skill}")
                
                with col2:
                    st.markdown("**Eğitim Gereksinimleri:**")
                    for req in career.get('education_requirements', []):
                        st.markdown(f"- {req}")
                
                st.markdown("**Kariyer Yolu:**")
                for step in career.get('career_path', []):
                    st.markdown(f"- {step}")

    def _display_education_recommendations(self, analysis: Dict[str, Any]):
        """Display education recommendations section"""
        st.markdown("#### 🎓 Eğitim Önerileri")
        
        for edu in analysis.get('education_recommendations', []):
            with st.expander(f"📚 {edu.get('program', 'Program Yok')}"):
                st.markdown(f"**Açıklama:** {edu.get('description', 'Açıklama yok.')}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Önerilen Üniversiteler:**")
                    for uni in edu.get('universities', []):
                        st.markdown(f"- {uni}")
                
                with col2:
                    st.markdown("**Önemli Dersler:**")
                    for course in edu.get('key_courses', []):
                        st.markdown(f"- {course}")

    def _display_next_steps(self, analysis: Dict[str, Any]):
        """Display next steps section"""
        st.markdown("#### 📋 Sonraki Adımlar")
        
        for i, step in enumerate(analysis.get('next_steps', []), 1):
            st.markdown(f"{i}. {step}")

    def _get_user_profile(self) -> Dict[str, Any]:
        """Get user's profile data from database"""
        try:
            result = self.db.execute_query('''
                SELECT grade, study_type, target_university, 
                       target_department, target_rank
                FROM users
                WHERE id = ?
            ''', (self.user_id,))
            
            return cast(Dict[str, Any], result) if result else {}

        except DatabaseError as e:
            logger.error(f"Error fetching user profile: {e}")
            return {}

    def _get_user_responses(self) -> Optional[Dict[str, Any]]:
        """Get user's career questionnaire responses"""
        try:
            results = self.db.execute_query('''
                SELECT responses
                FROM career_responses
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (self.user_id,))
            
            if results and isinstance(results, list) and len(results) > 0:
                # Access the first row properly
                row = results[0]
                if 'responses' in row:
                    return json.loads(row['responses'])
            return None

        except DatabaseError as e:
            logger.error(f"Error fetching responses: {e}")
            return None


    def _save_responses(self, responses: Dict[str, Any]) -> bool:
        """Save career questionnaire responses"""
        try:
            self.db.insert_record('career_responses', {
                'user_id': self.user_id,
                'responses': json.dumps(responses, ensure_ascii=False),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return True

        except DatabaseError as e:
            logger.error(f"Error saving responses: {e}")
            return False

    def _clear_responses(self) -> bool:
        """Clear user's career questionnaire responses"""
        try:
            self.db.execute_query('''
                DELETE FROM career_responses
                WHERE user_id = ?
            ''', (self.user_id,))
            return True

        except DatabaseError as e:
            logger.error(f"Error clearing responses: {e}")
            return False

    def _get_subjects(self) -> List[str]:
        """Get available subjects from database"""
        try:
            results = self.db.execute_query(
                'SELECT name FROM subjects',
                fetch_all=True
            )
            return [row['name'] for row in results] if results else []

        except DatabaseError as e:
            logger.error(f"Error fetching subjects: {e}")
            return []

    def _create_personality_chart(self, traits: Dict[str, float]) -> go.Figure:
        """Create radar chart for personality traits"""
        if not traits:
            st.warning("Kişilik özellikleri verisi bulunamadı.")
            return go.Figure()
        
        categories = list(traits.keys())
        values = list(traits.values())
        
        # Ensure the radar chart is a closed loop
        categories += [categories[0]]
        values += [values[0]]
        
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Kişilik Özellikleri'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False
        )

        return fig


def show_career_pathfinder():
    """Initialize and display career pathfinder"""
    user_id = st.session_state.get('user_id')

    if user_id is None:
        st.error("Kullanıcı oturumu açık değil.")
        return

    pathfinder = CareerPathFinder(user_id)
    pathfinder.show()
