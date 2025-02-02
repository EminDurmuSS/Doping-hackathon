# ui/styles/custom.py
def load_css():
    """Load custom CSS styles"""
    return """
        /* Dark Theme Base */
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
        
        /* Main Container */
        .main {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        /* Custom Card */
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
        
        /* Alert Cards */
        .alert-card {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            display: flex;
            align-items: center;
        }
        
        .alert-card.success {
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid #10B981;
        }
        
        .alert-card.warning {
            background-color: rgba(245, 158, 11, 0.1);
            border: 1px solid #F59E0B;
        }
        
        .alert-card.error {
            background-color: rgba(239, 68, 68, 0.1);
            border: 1px solid #EF4444;
        }
        
        .alert-card.info {
            background-color: rgba(79, 70, 229, 0.1);
            border: 1px solid #4F46E5;
        }
        
        /* Buttons */
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
        
        /* Form Controls */
        .stTextInput > div > div > input {
            background-color: var(--secondary-background);
            color: var(--text-color);
            border-radius: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 0.75rem;
            transition: all 0.3s ease;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade-in {
            animation: fadeIn 0.5s ease-out forwards;
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
        
        /* Notification Item */
        .notification-item {
            padding: 0.75rem;
            border-radius: 0.5rem;
            background-color: rgba(255, 255, 255, 0.05);
            margin-bottom: 0.5rem;
        }
        
        .notification-item small {
            color: #B0B0B0;
        }
        
        /* Progress Bars */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, 
                       var(--primary-color), 
                       var(--accent-color));
            border-radius: 1rem;
        }
        
        /* Metrics */
        .css-1wivf9j {
            background-color: var(--secondary-background);
            border-radius: 0.75rem;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    """

def apply_custom_css():
    """Apply custom CSS to the application"""
    import streamlit as st
    
    css = load_css()
    st.markdown(f"""
        <style>
            {css}
        </style>
    """, unsafe_allow_html=True)