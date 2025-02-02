# ui/components/navigation.py
import streamlit as st
from streamlit_option_menu import option_menu
from typing import List, Dict, Any, Callable, Sequence
from streamlit.delta_generator import DeltaGenerator  # Correct import

def create_sidebar_menu() -> str:
    """Create sidebar navigation menu"""
    with st.sidebar:
        selected = option_menu(
            "MotiKoç Menü",
            ["Ana Sayfa", "YKS Paneli", "Çalışma Takvimi", 
             "Performans Analizi", "Sosyal Özellikler", "YKS Forumu",
             "Sesli Rehber", "YKS Tercih Asistanı", "Kariyer Yol Haritası", 
             "Ayarlar"],
            icons=['house', 'mortarboard', 'calendar3', 
                   'graph-up', 'people', 'chat-dots',
                   'volume-up', 'search', 'map', 'gear'],
            menu_icon="book",
            default_index=0,
            key="main_menu"
        )
        return selected

def create_page_tabs(tabs: List[Dict[str, Any]]) -> Sequence[DeltaGenerator]:
    """Create page tabs and return tab objects"""
    tab_titles = [tab['title'] for tab in tabs]
    tab_icons = [tab.get('icon', '') for tab in tabs]
    
    # Create tabs with icons
    tab_labels = [f"{icon} {title}" for icon, title in zip(tab_icons, tab_titles)]
    return st.tabs(tab_labels)

def create_breadcrumb(items: List[Dict[str, str]]) -> None:
    """Create breadcrumb navigation"""
    breadcrumb = " > ".join([
        f"[{item['label']}]({item['url']})" for item in items
    ])
    st.markdown(breadcrumb)

def show_notifications() -> None:
    """Display user notifications in sidebar"""
    with st.sidebar:
        with st.expander("🔔 Bildirimler", expanded=False):
            # Get notifications from session state or database
            notifications = st.session_state.get('notifications', [])
            
            if notifications:
                for notif in notifications:
                    st.markdown(f"""
                    <div class="notification-item">
                        <p>{notif['message']}</p>
                        <small>{notif['time']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("Yeni bildirim yok")

def create_back_button(on_click: Callable[[], None]) -> None:
    """Create back navigation button"""
    if st.button("⬅️ Geri Dön"):
        on_click()
