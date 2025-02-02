# main.py
import streamlit as st
from core.state import init_session_state
from core.database import init_db
from ui.components.navigation import create_sidebar_menu  # Updated import
from ui.pages.home import HomePage  # Import the HomePage class
from ui.pages import settings
from features.forum.views import show_forum
from features.voice.guidance import show_voice_guidance
from features.university.finder import show_university_finder
from features.career.pathfinder import show_career_pathfinder
from features.social.features import show_social_features  # Adjust based on where you defined it
from features.calendar.study_calendar import show_study_calendar
from features.performance.analytics import show_performance_analytics  # Newly added import



def main():
    # Initialize
    init_db()
    init_session_state()

    if st.session_state.user_id is None:
        from ui.pages.authy import show_login_page
        show_login_page()
    else:
        selected = create_sidebar_menu()

        # Route to appropriate page based on selection
        pages = {
            "Ana Sayfa": lambda: HomePage(st.session_state.user_id).show(),  # Fixed
            "Çalışma Takvimi": show_study_calendar,
            "Performans Analizi": show_performance_analytics,
            "Sosyal Özellikler": show_social_features,
            "YKS Forumu": show_forum,
            "Sesli Rehber": show_voice_guidance,
            "YKS Tercih Asistanı": show_university_finder,
            "Kariyer Yol Haritası": show_career_pathfinder,
            "Ayarlar": settings.show_settings
        }
        
        # Execute the selected page function/method
        if selected in pages:
            pages[selected]()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Uygulama çalıştırılırken bir hata oluştu.")
        st.error(f"Hata detayı: {str(e)}")
        if st.button("🔄 Uygulamayı Yeniden Başlat"):
            st.session_state.clear()
            st.rerun()