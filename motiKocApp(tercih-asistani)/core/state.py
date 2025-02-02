# core/state.py
import streamlit as st
from typing import Any, Dict

def init_session_state():
    """Initialize all session state variables"""
    default_state = {
        'page': 'home',
        'user_id': None,
        'username': '',
        'current_plan': None,
        'current_section': 0,
        'responses': {},
        'profile': None,
        'recommendations': None,
        'search_history': [],
        'filtered_results': None,
        'current_response': None,
        'show_tips': True
    }
    
    for key, default_value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_state(key: str, default: Any = None) -> Any:
    """Get value from session state with default"""
    return st.session_state.get(key, default)

def set_state(key: str, value: Any):
    """Set value in session state"""
    st.session_state[key] = value

def update_state(updates: Dict[str, Any]):
    """Update multiple session state values"""
    for key, value in updates.items():
        set_state(key, value)

def clear_state():
    """Clear all session state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()

def reset_page_state():
    """Reset page-specific state while maintaining user session"""
    user_id = get_state('user_id')
    username = get_state('username')
    clear_state()
    update_state({
        'user_id': user_id,
        'username': username
    })

def check_state_requirements(required_keys: list) -> bool:
    """Check if all required state variables are present"""
    return all(key in st.session_state for key in required_keys)