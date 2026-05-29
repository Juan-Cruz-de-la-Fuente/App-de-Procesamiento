import streamlit as st

def apply_styles():
    st.markdown("""
<style>
    /* IMPORTS */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');

    /* GLOBAL RESET */
    html {
        scroll-behavior: smooth !important;
    }
    
    .stApp {
        background-color: #000000;
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        color: #ffffff;
        letter-spacing: 2px;
    }

    /* HIDE STREAMLIT ELEMENTS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* CUSTOM BUTTONS (SpaceX Style) */
    .stButton > button {
        background-color: transparent !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        border-radius: 0px !important; /* Sharp edges */
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        padding: 0.5rem 1rem;
        font-family: 'Orbitron', sans-serif;
        width: 100%;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }

    .stButton > button:hover {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.5);
    }
    
    /* INPUT FIELDS */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
        background-color: #111111 !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 0px !important;
    }
    
    /* CARDS & CONTAINERS */
    div.section-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        transition: transform 0.3s ease;
        margin-bottom: 2rem; /* Spacing for rows */
    }
    
    div.section-card:hover {
        border-color: #ffffff;
        transform: translateY(-5px);
    }

    /* SCROLLBAR */
    ::-webkit-scrollbar {
        width: 8px;
        background: #000;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #fff;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #000;
        border-radius: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #888;
        border-radius: 0;
        text-transform: uppercase;
        font-family: 'Orbitron', sans-serif;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #fff;
        border-bottom: 2px solid #fff;
    }
    
    /* TEXT MANUAL STYLE */
    .manual-text {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        color: #e0e0e0;
        font-size: 1rem;
        text-align: justify;
    }
    .manual-header {
        font-family: 'Orbitron', sans-serif;
        color: #4ade80; /* Green accent for manual headers */
        margin-top: 1.5rem;
    }

    /* POPOVER STYLE (FOR NAVBAR) */
    div[data-testid='stPopover'] {
        height: 100% !important;
        margin: 0 !important;
        display: flex !important;
    }
    div[data-testid='stPopover'] > div {
        height: 100% !important;
        width: 100% !important;
    }
    div[data-testid='stPopover'] > div > button, div[data-testid='stPopover'] > button {
        background-color: transparent !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        border-radius: 0px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        padding: 0.5rem 1rem !important;
        font-family: 'Orbitron', sans-serif !important;
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    div[data-testid='stPopover'] > div > button:hover, div[data-testid='stPopover'] > button:hover {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.5) !important;
    }
    
    div[data-testid='stPopoverBody'] button { 
        font-size: 0.85rem !important; 
        padding: 0.2rem 0.5rem !important; 
    }

</style>
""", unsafe_allow_html=True)
