# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import json

def calcular_variable_atmosferica(df, variable):
    import pandas as pd
    res = df.get('Presion', pd.Series([0]*len(df)))
    if variable == 'Presión Total [Actual]':
        return res
    elif variable == 'rho_inf':
        return df.get('rho_inf', 1.225).fillna(1.225)
    elif variable == 'V_inf':
        return df.get('V_inf', 0.0).fillna(0.0)
    elif variable == 'P_inf':
        return df.get('P_inf', 101325.0).fillna(101325.0)
    elif variable == 'T_inf':
        return df.get('T_inf', 15.0).fillna(15.0)
    return res
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
import io
from datetime import datetime
import pyvista as pv
import tempfile
import os
import base64
from scipy.spatial import Delaunay
from typing import Dict
import os
import zipfile
import random
import pyvista as pv
from scipy.interpolate import griddata
import auth  # Import the new authentication module
import itertools


def rotate_points(x, y, z, angle_x, angle_y, angle_z):
    """Rotates 3D points around X, Y, Z axes (angles in degrees)."""
    # Convert to radians
    rad_x = np.radians(angle_x)
    rad_y = np.radians(angle_y)
    rad_z = np.radians(angle_z)
    
    # Rotation Matrices
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(rad_x), -np.sin(rad_x)],
        [0, np.sin(rad_x), np.cos(rad_x)]
    ])
    
    ry = np.array([
        [np.cos(rad_y), 0, np.sin(rad_y)],
        [0, 1, 0],
        [-np.sin(rad_y), 0, np.cos(rad_y)]
    ])
    
    rz = np.array([
        [np.cos(rad_z), -np.sin(rad_z), 0],
        [np.sin(rad_z), np.cos(rad_z), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: R = Rz * Ry * Rx
    R = rz @ (ry @ rx)
    
    # Arrange points as (3, N)
    points = np.vstack([x, y, z])
    
    # Rotate
    rotated_points = R @ points
    
    return rotated_points[0,:], rotated_points[1,:], rotated_points[2,:]

# Configuración de la página
st.set_page_config(
    page_title="Laboratorio de Aerodinámica y Fluidos - UTN HAEDO",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- INITIALIZATION ---
if 'seccion_actual' not in st.session_state:
    st.session_state.seccion_actual = 'inicio'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

# --- SHADCN-INSPIRED CLEAN THEME & HORIZONTAL NAV ---
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

    /* HIDE STREAMLIT ELEMENT S */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* REMOVE SIDEBAR MARGIN IF PRESENT (Though we aren't using st.sidebar anymore) */
    
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
    
    /* NAV BUTTON SPECIFIC (Optional, if we want to differentiate) */
    div[data-testid="column"] .stButton > button {
        font-size: 0.9rem;
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

</style>
""", unsafe_allow_html=True)



def rotate_points(x, y, z, angle_x, angle_y, angle_z):
    """
    Rota puntos 3D alrededor de los ejes X, Y, Z.
     Ángulos en grados.
    """
    # Convertir a radianes
    rad_x = np.radians(angle_x)
    rad_y = np.radians(angle_y)
    rad_z = np.radians(angle_z)
    
    # Matrices de rotación
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rad_x), -np.sin(rad_x)],
        [0, np.sin(rad_x), np.cos(rad_x)]
    ])
    
    Ry = np.array([
        [np.cos(rad_y), 0, np.sin(rad_y)],
        [0, 1, 0],
        [-np.sin(rad_y), 0, np.cos(rad_y)]
    ])
    
    Rz = np.array([
        [np.cos(rad_z), -np.sin(rad_z), 0],
        [np.sin(rad_z), np.cos(rad_z), 0],
        [0, 0, 1]
    ])
    
    # Matriz de rotación combinada R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    
    # Apilar puntos
    points = np.vstack([x, y, z])
    
    # Rotar
    rotated_points = R @ points
    
    return rotated_points[0], rotated_points[1], rotated_points[2]


def render_navbar():
    # --- HORIZONTAL NAVIGATION BAR ---
    if st.session_state.get('logged_in'):
        with st.container():
            c_sl, c1, c2, c_smn, c3, c4, c5, c6, c_sr = st.columns([0.5, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 0.5])
            
            with c1:
                t1 = "primary" if st.session_state.seccion_actual == 'inicio' else "secondary"
                if st.button("🚀 INICIO", use_container_width=True, type=t1):
                    st.session_state.seccion_actual = 'inicio'
                    st.rerun()
            
            with c2:
                # Dropdown for Ensayo Estela
                is_estela = st.session_state.seccion_actual in ['betz_2d', 'vis_2d_nueva', 'betz_3d', 'betz_4d', 'animacion_4d']
                if is_estela:
                    st.markdown("""<style>
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
                        color: var(--primary-color) !important;
                        border: 1px solid var(--primary-color) !important;
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
                    </style>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<style>
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
                    </style>""", unsafe_allow_html=True)
                
                with st.popover("🌌 ENSAYO ESTELA", use_container_width=True):
                    st.markdown("<style>div[data-testid='stPopoverBody'] button { font-size: 0.85rem !important; padding: 0.2rem 0.5rem !important; }</style>", unsafe_allow_html=True)
                    if st.button("📈 Vis. Estela 1D", use_container_width=True, type="primary" if st.session_state.seccion_actual == 'betz_2d' else "secondary"):
                        st.session_state.seccion_actual = 'betz_2d'
                        st.rerun()
                    if st.button("📈 Vis. Estela 2D", use_container_width=True, type="primary" if st.session_state.seccion_actual == 'vis_2d_nueva' else "secondary"):
                        st.session_state.seccion_actual = 'vis_2d_nueva'
                        st.rerun()
                    if st.button("🌪️ Vis. Estela 3D", use_container_width=True, type="primary" if st.session_state.seccion_actual == 'betz_3d' else "secondary"):
                        st.session_state.seccion_actual = 'betz_3d'
                        st.rerun()
                    if st.button("🌌 Vis. Estela 4D", use_container_width=True, type="primary" if st.session_state.seccion_actual == 'betz_4d' else "secondary"):
                        st.session_state.seccion_actual = 'betz_4d'
                        st.rerun()
                    if st.button("🌀 Análisis de Vórtices", use_container_width=True, type="primary" if st.session_state.seccion_actual == 'analisis_vortices' else "secondary"):
                        st.session_state.seccion_actual = 'analisis_vortices'
                        st.rerun()
                    if st.button("🎬 Animación 4D", use_container_width=True, type="primary" if st.session_state.seccion_actual == 'animacion_4d' else "secondary"):
                        st.session_state.seccion_actual = 'animacion_4d'
                        st.rerun()
                    if st.button("🔧 Herramientas", use_container_width=True, type="primary" if st.session_state.seccion_actual == 'herramientas' else "secondary"):
                        st.session_state.seccion_actual = 'herramientas'
                        st.rerun()
                    if st.button("📊 INTERPOLACIÖN", use_container_width=True, type="primary" if st.session_state.seccion_actual == 'interpolacion_aoa' else "secondary"):
                        st.session_state.seccion_actual = 'interpolacion_aoa'
                        st.rerun()

            with c_smn:
                with st.popover("🧪 ENSAYO SMN", use_container_width=True):
                    if st.button("📈 Vis. SMN 2D", use_container_width=True, key="legacy_btn_smn_2d"):
                        st.session_state.seccion_actual = 'smn_2d'
                        st.rerun()
                    if st.button("🌪️ Vis. SMN 3D", use_container_width=True, key="legacy_btn_smn_3d"):
                        st.session_state.seccion_actual = 'smn_3d'
                        st.rerun()
                    if st.button("🌌 Vis. SMN 4D", use_container_width=True, key="legacy_btn_smn_4d"):
                        st.session_state.seccion_actual = 'smn_4d'
                        st.rerun()

            with c3:
                t3 = "primary" if st.session_state.seccion_actual == 'ensayo_betz' else "secondary"
                if st.button("🧪 ENSAYO DE BETZ", use_container_width=True, type=t3):
                     st.session_state.seccion_actual = 'ensayo_betz'
                     st.rerun()

            with c4:
                t4 = "primary" if st.session_state.seccion_actual == 'modelos' else "secondary"
                if st.button("📦 MODELOS", use_container_width=True, type=t4):
                     st.session_state.seccion_actual = 'modelos'
                     st.rerun()

            with c5:
                t5 = "primary" if st.session_state.seccion_actual == 'configuracion' else "secondary"
                if st.button("⚙️ CONFIG", use_container_width=True, type=t5):
                    st.session_state.seccion_actual = 'configuracion'
                    st.rerun()

            with c6:
                if st.button(f"👤 PERFIL / SALIR", use_container_width=True):
                    st.session_state.logged_in = False
                    st.session_state.username = None
                    st.rerun()

            st.markdown("<hr style='border-top: 1px solid #333; margin-top: 10px;'>", unsafe_allow_html=True)

# Render Nav ONLY if NOT on Inicio page (on Inicio it comes after Hero)
if st.session_state.get('logged_in') and st.session_state.seccion_actual != 'inicio':
    render_navbar()


# --- AUTHENTICATION CHECK ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

def login_page():
    folder_portada = "Imagenes de portada"
    img_b64_list = []
    if os.path.exists(folder_portada):
        valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
        import random
        archivos = [f for f in os.listdir(folder_portada) if os.path.splitext(f)[1].lower() in valid_exts]
        random.shuffle(archivos)
        for f in archivos[:15]: 
            try:
                with open(os.path.join(folder_portada, f), 'rb') as img_f:
                    img_b64_list.append(base64.b64encode(img_f.read()).decode())
            except:
                pass
                
    if not img_b64_list:
        fallback_url = 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop'
        carousel_css = ""
        carousel_html = f'<div style="position: absolute; top:0; left:0; right:0; bottom:0; background-size: cover; background-position: center; background-image: linear-gradient(to bottom, rgba(0,0,0,0.2), rgba(0,0,0,0.8)), url(\'{fallback_url}\'); z-index: 0; opacity: 1;"></div>'
    else:
        num_imgs = len(img_b64_list)
        if num_imgs == 1:
            carousel_css = ""
            carousel_html = f'<div style="position: absolute; top:0; left:0; right:0; bottom:0; background-size: cover; background-position: center; background-image: linear-gradient(to bottom, rgba(0,0,0,0.2), rgba(0,0,0,0.8)), url(\'data:image/jpeg;base64,{img_b64_list[0]}\'); z-index: 0; opacity: 1;"></div>'
        else:
            time_per_slide = 5
            total_time = num_imgs * time_per_slide
            percent_visible = 100.0 / num_imgs
            fade_percent = percent_visible * 0.15
            
            carousel_css = "<style>\n"
            carousel_html = ""
            for i, b64 in enumerate(img_b64_list):
                p_start = (i * percent_visible)
                p_in = p_start + fade_percent
                p_out = ((i+1) * percent_visible) - fade_percent
                p_end = ((i+1) * percent_visible)
                
                carousel_css += f"""
.login-bg-{i} {{
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background-image: linear-gradient(to bottom, rgba(0,0,0,0.2), rgba(0,0,0,0.8)), url('data:image/jpeg;base64,{b64}');
    background-size: cover; background-position: center;
    animation: fadeLogin{i} {total_time}s infinite;
    opacity: 0;
    z-index: 0;
}}
@keyframes fadeLogin{i} {{
    0%, {max(0, p_start-0.01):.2f}% {{ opacity: 0; }}
    {p_start:.2f}% {{ opacity: 0; }}
    {p_in:.2f}% {{ opacity: 1; }}
    {p_out:.2f}% {{ opacity: 1; }}
    {p_end:.2f}% {{ opacity: 0; }}
    100% {{ opacity: 0; }}
}}
"""
                carousel_html += f'<div class="login-bg-{i}"></div>\n'
                
            carousel_css += "</style>\n"

    # Login Hero Image
    st.markdown(f"{carousel_css}", unsafe_allow_html=True)
    st.markdown(f"""
<div style="position: relative; width: 100%; min-height: 80vh; padding: 4rem 1rem; border-radius: 0px; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 1px solid #333; overflow: hidden; margin-top: -1rem; margin-bottom: 2rem;">
{carousel_html}
<div style="position: relative; z-index: 10; display: flex; flex-direction: column; align-items: center; width: 100%; max-width: 800px; padding: 2.5rem; background-color: transparent; border: none; box-shadow: none;">
<h1 style="font-family: 'Orbitron', sans-serif; font-size: 4.5rem; font-weight: 900; letter-spacing: 2px; margin-bottom: 0.5rem; text-shadow: 0 10px 30px rgba(0,0,0,0.5); color: white; text-align: center;">BETZ APP</h1>
<p style="font-family: 'Inter', sans-serif; font-size: 1.2rem; letter-spacing: 6px; text-transform: uppercase; color: rgba(255,255,255,0.8); margin-top: 0.5rem; text-shadow: 0 4px 15px rgba(0,0,0,0.8); text-align: center; margin-bottom: 2rem;">Sistema de Procesamiento de Datos de Túnel de Viento</p>
</div>
</div>
""", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.markdown("###  Iniciar Sesión")
        username = st.text_input("Usuario", placeholder="admin", key="login_user")
        password = st.text_input("Contraseña", type="password", placeholder="••••", key="login_pass")
        
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        
        if st.button("Iniciar Sesión", type="primary", use_container_width=True):
             # VALIDACIÓN 100% DESDE LA BASE DE DATOS DRIVE
            if auth.verify_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                # Inicializar estructura de carpetas en Drive para este usuario
                try:
                    import drive_api as _dapi
                    _dapi.init_user_folders(username)
                except Exception as _e:
                    pass  # No bloquear el login si Drive falla
                st.rerun()
            else:
                st.error("Credenciales incorrectas")
        
        st.info("💡 Contacte al administrador para obtener credenciales de acceso.")

if not st.session_state.logged_in:
    login_page()
    st.stop()  # Stop execution here if not logged in

# Inicializar estado de la sesión (resto de inicializaciones)
if 'seccion_actual' not in st.session_state:
    st.session_state.seccion_actual = 'inicio'
if 'archivos_cargados' not in st.session_state:
    st.session_state.archivos_cargados = []
if 'datos_procesados' not in st.session_state:
    st.session_state.datos_procesados = {}
if 'configuracion_inicial' not in st.session_state:
    st.session_state.configuracion_inicial = {}
if 'sub_archivos_generados' not in st.session_state:
    st.session_state.sub_archivos_generados = {}
if 'datos_3d_filtrados' not in st.session_state:
    st.session_state.datos_3d_filtrados = {}
if 'sub_archivos_3d' not in st.session_state:
    st.session_state.sub_archivos_3d = {}
if 'configuracion_3d' not in st.session_state:
    st.session_state.configuracion_3d = {}
if 'sub_archivos_3d_generados' not in st.session_state:
    st.session_state.sub_archivos_3d_generados = {}
if 'diferencias_guardadas' not in st.session_state:
    st.session_state.diferencias_guardadas = {}



def extraer_tiempo_y_coordenadas_YZ(nombre_archivo):
    """
    Extraer tiempo y coordenadas para el sistema estandarizado Y-Z.
    
    Mapeo:
    - Archivo 'X' -> Traverser -> App 'Pos_Y_Traverser'
    - Archivo 'Y' -> Altura Base -> App 'Pos_Z_Base' (Si existe, sino 0)
    - Tiempo -> App 'Tiempo_s'
    """
    tiempo = None
    y_traverser = None 
    z_base = None

    # Normalizar nombre sin extensión
    nombre = os.path.basename(str(nombre_archivo))
    nombre_sin_ext = re.sub(r'\.\w+$', '', nombre)

    # 1) Intentar extraer tiempo
    partes = nombre_sin_ext.split('_')
    if partes and partes[-1].isdigit():
        try:
            tiempo = int(partes[-1])
        except:
            tiempo = None

    if tiempo is None:
        tiempo_match = re.search(r"[Tt](\d+)\s*[sS]?$", nombre_sin_ext)
        if tiempo_match:
            tiempo = int(tiempo_match.group(1))

    # 2) Extraer X del archivo (que es nuestra Pos_Y_Traverser)
    x_match = re.search(r"[Xx][_\-=]?(-?\d+)", nombre_sin_ext)
    if x_match:
        try:
            y_traverser = int(x_match.group(1))
        except:
            pass
    
    # Fallback X
    if y_traverser is None:
        m = re.search(r"[Xx]\s*(\d+)", nombre_sin_ext)
        if m:
            y_traverser = int(m.group(1))

    # 3) Extraer Y del archivo (que es nuestra Pos_Z_Base)
    y_match = re.search(r"[Yy][_\-=]?(-?\d+)", nombre_sin_ext)
    if y_match:
        try:
            z_base = int(y_match.group(1))
        except:
            pass
            
    # Fallback Y
    if z_base is None:
        m = re.search(r"[Yy]\s*(\d+)", nombre_sin_ext)
        if m:
            z_base = int(m.group(1))

    return tiempo, y_traverser, z_base

def normalizar_nombre_sensor(sensor_text):
    """Normaliza una entrada de cabecera de sensor a 'Presion-Sensor N' (N entero global).
       Acepta formatos como:
         - 'Presion-Sensor_0_1' -> 'Presion-Sensor 1'
         - 'Presion-Sensor_1_12' -> 'Presion-Sensor 24'
         - 'Presion-Sensor 5' -> 'Presion-Sensor 5'
       Devuelve None si no puede normalizar.
    """
    if pd.isna(sensor_text):
        return None
    s = str(sensor_text).strip()
    if not s:
        return None

    # Caso 'Presion-Sensor_offset_index' (ej 'Presion-Sensor_1_3')
    m = re.search(r'(?i)presion[-_ ]*sensor[_\-]?(\d+)[_\-](\d+)', s)
    if m:
        offset = int(m.group(1))
        idx = int(m.group(2))
        sensor_global = offset * 12 + idx
        return f"Presion-Sensor {sensor_global}"

    # Caso 'Presion-Sensor 12' o 'Presion-Sensor_12'
    m2 = re.search(r'(?i)presion[-_ ]*sensor[_\-\s]*(\d+)', s)
    if m2:
        sensor_global = int(m2.group(1))
        return f"Presion-Sensor {sensor_global}"

    # Caso donde venga solo un número al final
    nums = re.findall(r'(\d+)', s)
    if nums:
        # si hay dos números, considerar que puede ser offset,index
        if len(nums) >= 2:
            offset = int(nums[-2])
            idx = int(nums[-1])
            if 0 <= offset <= 9 and 1 <= idx <= 12:
                sensor_global = offset * 12 + idx
                return f"Presion-Sensor {sensor_global}"
        # si hay uno solo, usarlo como número de sensor
        sensor_global = int(nums[-1])
        return f"Presion-Sensor {sensor_global}"

    # Si no se reconoce, devolver la cadena original (por si acaso)
    return s


def obtener_numero_sensor_desde_columna(col_name):
    """Devuelve el número entero del sensor si el nombre de columna tiene 'Presion-Sensor N' (o similar), sino None."""
    if pd.isna(col_name):
        return None
    s = str(col_name)
    m = re.search(r'(?i)presion[-_ ]*sensor[_\-\s]*(\d+)', s)
    if m:
        return int(m.group(1))
    # si no coincide, intentar extraer último número
    nums = re.findall(r'(\d+)', s)
    if nums:
        return int(nums[-1])
    return None


def calcular_altura_absoluta_z(sensor_num, z_base_ref, posicion_inicial, distancia_entre_tomas, n_sensores, orden="asc"):
    """
    Calcula la altura absoluta Z de un sensor dado, basándose en una altura de referencia (Z Base).
    """
    if sensor_num is None:
        return None
    
    toma_index = int(sensor_num)  # ahora global: 1..21

    # Nota: la lógica original sumaba (toma_index - 1) * distancia a la referencia.
    # Asumimos que z_base_ref es la altura del primer sensor (o del 12, segun contexto, pero aqui parece ser base).
    # Sin embargo, el parametro 'posicion_inicial' se llama 'distancia_toma_12' en config.
    # Si z_base_ref es la lectura del 'y_traverser' (que era Z), entonces:
    
    if orden == "asc":
        z_total = z_base_ref + (toma_index - 1) * distancia_entre_tomas
    else:
        z_total = z_base_ref + (n_sensores - toma_index) * distancia_entre_tomas

    return z_total

def extraer_nombre_base_archivo(nombre_archivo):
    """Extraer nombre base del archivo (sin extensión y sin 'incertidumbre_')"""
    nombre_base = nombre_archivo.replace('.csv', '').replace('incertidumbre_', '').replace('_', ' ')
    # Capitalizar primera letra de cada palabra
    return ' '.join(word.capitalize() for word in nombre_base.split())

def procesar_promedios(archivo_csv, orden="asc", archivo_infinito=None):
    """Procesar archivo de incertidumbre y detectar automáticamente la cantidad de sensores."""
    try:
        df_raw = pd.read_csv(archivo_csv, sep=";", header=None, dtype=str)  # leer como texto para robustez

        # Buscar la palabra "importante" para determinar dónde terminar
        index_final = df_raw[df_raw.apply(lambda row: row.astype(str).str.contains("importante", case=False).any(), axis=1)].index
        if not index_final.empty:
            df_raw = df_raw.iloc[:index_final[0]]

        resultados = []

        # Procesar bloques de 10 filas (misma lógica base)
        for i in range(0, df_raw.shape[0], 10):
            bloque = df_raw.iloc[i:i+10]
            if bloque.empty or len(bloque) < 3:
                continue

            archivo = bloque.iloc[0, 0]
            raw_sensores = bloque.iloc[0, 1:].tolist()
            muestras = bloque.iloc[1, 1] if 1 < bloque.shape[1] else None

            # Normalizar: si la cabecera vino entera en una sola celda separada por ';'
            sensores_lista = []
            for entry in raw_sensores:
                if pd.isna(entry):
                    continue
                s = str(entry).strip()
                if ';' in s:
                    partes = [p.strip() for p in s.split(';') if p.strip()]
                    sensores_lista.extend(partes)
                else:
                    sensores_lista.append(s)

            # Lo mismo para valores: si hay una celda con ;, expandir
            valores_lista = []
            for entry in bloque.iloc[2, 1:].tolist(): # Asumimos que los valores están en la fila 2 (índice 2)
                if pd.isna(entry):
                    continue
                s = str(entry).strip()
                if ';' in s:
                    partes = [p.strip() for p in s.split(';')]
                    valores_lista.extend(partes)
                else:
                    valores_lista.append(s)

            # Si por alguna razón no se alinean en longitud, ajustar
            n = max(len(sensores_lista), len(valores_lista))
            sensores_lista = (sensores_lista + [None] * n)[:n]
            valores_lista = (valores_lista + [None] * n)[:n]

            fila = {
                "Archivo": archivo,
                "Numero de muestras": muestras,
            }

            # Mapear cada sensor raw -> nombre normalizado y poner su valor
            for sensor_raw, valor_raw in zip(sensores_lista, valores_lista):
                nombre_sensor_norm = normalizar_nombre_sensor(sensor_raw)
                if nombre_sensor_norm is None:
                    continue
                # limpiar y convertir valor (coma -> punto)
                valor = valor_raw
                if isinstance(valor, str):
                    valor = valor.replace(',', '.').strip()
                try:
                    valor_num = float(valor) if (valor is not None and str(valor) != '') else np.nan
                except:
                    valor_num = np.nan

                fila[nombre_sensor_norm] = valor_num

            resultados.append(fila)

        df_resultado = pd.DataFrame(resultados)

        # Extraer tiempo y coordenadas desde nombre de archivo (columna "Archivo")
        if "Archivo" in df_resultado.columns:
            coordenadas_tiempo = df_resultado["Archivo"].apply(extraer_tiempo_y_coordenadas_YZ)
            df_resultado["Tiempo_s"] = [coord[0] for coord in coordenadas_tiempo]
            df_resultado["Pos_Y_Traverser"] = [coord[1] for coord in coordenadas_tiempo]
            df_resultado["Pos_Z_Base"] = [coord[2] for coord in coordenadas_tiempo]

            # Inyectar valores en el infinito
            def _extract_ts(n):
                m = re.search(r'(\d{10,14})', str(n))
                return m.group(1) if m else None
            df_resultado["Timestamp"] = df_resultado["Archivo"].apply(_extract_ts)

            # Soporte para archivo_infinito como upload o ruta local
            if archivo_infinito is not None:
                inf_file = archivo_infinito
                use_path = False
            else:
                inf_file = "Valores en el infinito.txt"
                use_path = True
            df_resultado["rho_inf"] = 1.225
            df_resultado["V_inf"] = 0.0
            df_resultado["P_inf"] = 101325.0

            file_exists = (os.path.exists(inf_file) if use_path else True)
            if file_exists:
                try:
                    df_inf = pd.read_csv(inf_file, sep=";", engine="python", skip_blank_lines=True)
                    df_inf.columns = [str(c).strip() for c in df_inf.columns]
                    if len(df_inf.columns) > 2:
                        first_col = df_inf.columns[0]
                        # Limpiar timestamp: tomar solo la parte entera antes de la coma
                        df_inf["ts_clean"] = df_inf[first_col].astype(str).str.split(',').str[0].str.strip()
                        # Formato real: DDMMYYHHMMSS (ej: 260424144919 = 26/04/2024 14:49:19)
                        df_inf["dt_val"] = pd.to_datetime(df_inf["ts_clean"], format='%d%m%y%H%M%S', errors='coerce')
                        # Fallback: intentar formato alternativo YYMMDDHHMMSS
                        mask_failed = df_inf["dt_val"].isna()
                        if mask_failed.any():
                            df_inf.loc[mask_failed, "dt_val"] = pd.to_datetime(
                                df_inf.loc[mask_failed, "ts_clean"], format='%y%m%d%H%M%S', errors='coerce'
                            )
                        df_inf = df_inf.dropna(subset=["dt_val"])

                        def get_inf_values(ts_str):
                            try:
                                if ts_str is None or str(ts_str) == 'None': return 1.225, 0.0, 101325.0, 15.0
                                ts_clean = str(ts_str).split(',')[0].strip()
                                # Intentar formato DDMMYYHHMMSS primero
                                target_dt = pd.to_datetime(ts_clean, format='%d%m%y%H%M%S', errors='coerce')
                                if pd.isna(target_dt):
                                    target_dt = pd.to_datetime(ts_clean, format='%y%m%d%H%M%S', errors='coerce')
                                if pd.isna(target_dt): return 1.225, 0.0, 101325.0, 15.0
                                diffs = (df_inf["dt_val"] - target_dt).abs()
                                idx = diffs.idxmin()
                                row = df_inf.loc[idx]
                                T = float(str(row.get("temp_baro", "15")).replace(",", "."))
                                P_hpa = float(str(row.get("pres_baro", "1013.25")).replace(",", "."))
                                HR = float(str(row.get("hrel", "50")).replace(",", "."))
                                P_pa = P_hpa * 100.0
                                T_kelvin = T + 273.15
                                P_v_sat = 6.1078 * (10 ** ((7.5 * T)/(237.3 + T)))
                                P_v = HR / 100.0 * P_v_sat
                                P_d = P_hpa - P_v
                                rho = (P_d * 100) / (287.058 * T_kelvin) + (P_v * 100) / (461.495 * T_kelvin)
                                v_inf = float(str(row.get("velocidad", "0.0")).replace(",", "."))
                                return rho, v_inf, P_pa, T
                            except:
                                return 1.225, 0.0, 101325.0, 15.0

                        recs = df_resultado["Timestamp"].apply(get_inf_values)
                        df_resultado["rho_inf"] = [r[0] for r in recs]
                        df_resultado["V_inf"] = [r[1] for r in recs]
                        df_resultado["P_inf"] = [r[2] for r in recs]
                        df_resultado["T_inf"] = [r[3] for r in recs]
                except Exception as e:
                    print("Error leyendo infinito:", e)
        else:
            df_resultado["Tiempo_s"] = None
            df_resultado["Pos_Y_Traverser"] = None
            df_resultado["Pos_Z_Base"] = None

        # 🔎 Detectar cantidad de sensores automáticamente
        sensores_cols = [c for c in df_resultado.columns if re.search(r'Presion[-_ ]*Sensor', str(c), re.IGNORECASE)]
        if sensores_cols:
            n_sensores = max([obtener_numero_sensor_desde_columna(c) for c in sensores_cols if obtener_numero_sensor_desde_columna(c) is not None], default=0)
        else:
            n_sensores = 0

        # Guardar en atributos del DataFrame para usar después
        df_resultado.attrs["n_sensores"] = n_sensores

        return df_resultado

    except Exception as e:
        st.error(f"Error al procesar archivo: {str(e)}")
        return None



def crear_archivos_individuales_por_tiempo_y_posicion(df_resultado, nombre_archivo_fuente):
    """
    Crea sub-archivos usando las coordenadas estandarizadas.
    """
    sub_archivos = {}
    nombre_base = extraer_nombre_base_archivo(nombre_archivo_fuente)
    nombre_original = os.path.splitext(os.path.basename(nombre_archivo_fuente))[0]

    # Iterar sobre Pos_Y_Traverser (antes X_coord)
    y_vals = df_resultado["Pos_Y_Traverser"].dropna().unique()
    for y_valor in sorted(y_vals):
        df_y = df_resultado[df_resultado["Pos_Y_Traverser"] == y_valor]

        # Iterar sobre Tiempo_s (antes Tiempo (s))
        t_vals = df_y["Tiempo_s"].dropna().unique()
        for tiempo in sorted(t_vals):
            df_yt = df_y[df_y["Tiempo_s"] == tiempo]

            clave_sub_archivo = f"{nombre_original}_X{int(y_valor) if pd.notna(y_valor) else 0}_T{tiempo}s"
            
            # Contar posiciones Z únicas
            num_z = len(df_yt['Pos_Z_Base'].unique()) if 'Pos_Z_Base' in df_yt.columns else 1

            sub_archivos[clave_sub_archivo] = {
                'archivo_fuente': nombre_base,
                'archivo_origen': nombre_original,
                'tiempo': tiempo,
                'pos_y_traverser': y_valor,
                'datos': df_yt,
                'nombre_archivo': f"{nombre_original}_X{int(y_valor) if pd.notna(y_valor) else 0}_T{tiempo}s.csv",
                'num_posiciones_z': num_z
            }

    return sub_archivos

def calcular_posiciones_sensores(distancia_toma_12, distancia_entre_tomas, n_sensores, orden="asc"):
    """
    Calcula las posiciones físicas de todos los sensores en función de:
    - distancia_toma_12: posición de la toma física número 12 (en mm)
    - distancia_entre_tomas: separación entre sensores consecutivos (en mm)
    - n_sensores: cantidad total de sensores detectados en el archivo
    - orden: "asc" o "des" (según cómo están montados los sensores)
    Devuelve un diccionario con la posición y número físico de cada sensor.
    """
    posiciones = {}
    for sensor_num in range(1, n_sensores + 1):
        if orden == "asc":
            y_position = (sensor_num - 1) * distancia_entre_tomas
        else:
            y_position = (n_sensores - sensor_num) * distancia_entre_tomas


        posiciones[f"Presion-Sensor {sensor_num}"] = {
            'x': 0,
            'y': y_position,
            'sensor_fisico': sensor_num
        }
    return posiciones


def crear_grafico_betz_concatenado(sub_archivos_seleccionados, posiciones_sensores, configuracion):
    fig = go.Figure()

    posicion_inicial = configuracion['distancia_toma_12']
    distancia_entre_tomas = configuracion['distancia_entre_tomas']
    orden = configuracion.get('orden', 'asc')
    colores_por_tiempo = {10: '#08596C', 20: '#E74C3C', 30: '#F39C12', 40: '#27AE60', 50: '#8E44AD', 60: '#3498DB'}

    datos_agrupados = {}
    for clave, sub_archivo in sub_archivos_seleccionados.items():
        grupo = (sub_archivo['archivo_fuente'], sub_archivo['tiempo'])
        datos_agrupados.setdefault(grupo, []).append(sub_archivo)

    for grupo, sub_archivos_del_grupo in datos_agrupados.items():
        archivo_fuente, tiempo = grupo
        color = colores_por_tiempo.get(tiempo, '#333333')

        z_grupo, presion_grupo = [], []

        for sub_archivo in sub_archivos_del_grupo:
            datos_tiempo = sub_archivo['datos']
            sensor_cols = [c for c in datos_tiempo.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]

            for _, fila in datos_tiempo.iterrows():
                y_traverser = fila.get('Pos_Y_Traverser', 0) if pd.notna(fila.get('Pos_Y_Traverser', np.nan)) else 0
                z_base_ref = fila.get('Pos_Z_Base', 0) if pd.notna(fila.get('Pos_Z_Base', np.nan)) else 0

                for col in sensor_cols:
                    sensor_num = obtener_numero_sensor_desde_columna(col)
                    if sensor_num is None:
                        continue
                    # Usamos z_base_ref como referencia
                    z_total = calcular_altura_absoluta_z(sensor_num, z_base_ref, posicion_inicial, distancia_entre_tomas, configuracion.get('n_sensores', len(sensor_cols)), orden)
                    presion = fila.get(col, None)
                    if pd.isna(presion):
                        continue
                    try:
                        presion = float(str(presion).replace(',', '.'))
                        presion_grupo.append(presion)
                        z_grupo.append(z_total)
                    except ValueError:
                        continue


def extraer_datos_para_grafico(sub_archivo, configuracion, variable='Presion Total'):
    """Extraer datos de presión y altura de un sub-archivo para gráficos (múltiples posiciones).
       Ahora soporta sensores numerados dinámicamente y mapeo de variables atmosféricas.
    """
    datos_tiempo = sub_archivo['datos']
    distancia_entre_tomas = configuracion['distancia_entre_tomas']
    posicion_inicial = configuracion.get('distancia_toma_12', 0)
    orden = configuracion.get('orden', 'asc')

    z_datos, presion_datos = [], []

    sensor_cols = [c for c in datos_tiempo.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]
    n_sensores = max([obtener_numero_sensor_desde_columna(c) for c in sensor_cols], default=0)

    for _, fila in datos_tiempo.iterrows():
        # Antes X_coord era Y. Ahora Pos_Y_Traverser es Y explícito.
        y_traverser = fila.get('Pos_Y_Traverser', 0)
        z_base_ref = fila.get('Pos_Z_Base', 0)

        for col in sensor_cols:
            sensor_num = obtener_numero_sensor_desde_columna(col)
            if sensor_num is None:
                continue
            z_total = calcular_altura_absoluta_z(sensor_num, z_base_ref, posicion_inicial, distancia_entre_tomas, n_sensores, orden)
            presion = fila.get(col, None)
            if pd.isna(presion):
                continue
            try:
                presion_val = float(str(presion).replace(',', '.'))
                # Calcular la variable seleccionada
                valor_final = presion_val # Por defecto Presion Total
                
                if variable == 'P_t / Rho_inf':
                    rho = float(fila.get('rho_inf', 1.225)) if pd.notna(fila.get('rho_inf', 1.225)) else 1.225
                    valor_final = presion_val / rho if rho != 0 else 0
                elif variable == 'Velocidad Infinito':
                    valor_final = float(fila.get('V_inf', 0.0)) if pd.notna(fila.get('V_inf', 0.0)) else 0.0
                elif variable == 'Presion Infinito':
                    valor_final = float(fila.get('P_inf', 101325.0)) if pd.notna(fila.get('P_inf', 101325.0)) else 101325.0

                z_datos.append(z_total)
                presion_datos.append(valor_final)
            except (ValueError, TypeError):
                continue

    # Ordenar y devolver
    if z_datos and presion_datos:
        # Ordenar por altura (z_datos)
        datos_ordenados = sorted(zip(z_datos, presion_datos))
        z_ordenado, presion_ordenada = zip(*datos_ordenados)
        return list(z_ordenado), list(presion_ordenada)

    # 🔑 SIEMPRE devolver dos listas
    return [], []


def calcular_area_bajo_curva(z_datos, presion_datos):
    """Calcular área bajo la curva usando regla del trapecio"""
    if len(z_datos) < 2 or len(presion_datos) < 2:
        return 0
    
    area = 0
    for i in range(len(z_datos) - 1):
        # Regla del trapecio
        h = z_datos[i + 1] - z_datos[i]
        area += h * (presion_datos[i] + presion_datos[i + 1]) / 2
    
    return abs(area)

def crear_superficie_diferencia_delaunay_3d(datos_a, datos_b, nombre_a, nombre_b, configuracion_3d, mostrar_puntos=True):
    """
    Crea una superficie 3D de diferencias con Delaunay y mejoras visuales.
    """
    try:
        posicion_inicial = configuracion_3d['distancia_toma_12']
        distancia_entre_tomas = configuracion_3d['distancia_entre_tomas']
        orden = configuracion_3d['orden']

        def extraer_puntos(datos):
            puntos = {}
            sensor_cols = [c for c in datos.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]
            for _, fila in datos.iterrows():
                y_traverser = fila.get('Pos_Y_Traverser', None) 
                z_base_ref = fila.get('Pos_Z_Base', None)
                if pd.isna(y_traverser) or pd.isna(z_base_ref):
                    continue

                for col in sensor_cols:
                    sensor_num = obtener_numero_sensor_desde_columna(col)
                    if sensor_num is None:
                        continue
                    altura_sensor_real = calcular_altura_absoluta_z(
                        sensor_num, z_base_ref, posicion_inicial, distancia_entre_tomas, orden
                    )
                    presion = fila.get(col, None)
                    if pd.isna(presion) or presion is None:
                        continue
                    try:
                        if isinstance(presion, str):
                            presion = float(presion.replace(',', '.'))
                        puntos[(y_traverser, altura_sensor_real)] = float(presion)
                    except (ValueError, TypeError):
                        continue
            return puntos

        puntos_a = extraer_puntos(datos_a)
        puntos_b = extraer_puntos(datos_b)

        puntos_comunes = set(puntos_a.keys()) & set(puntos_b.keys())
        if len(puntos_comunes) < 4:
            st.error("No hay suficientes puntos comunes para generar la resta de superficies.")
            return None

        puntos_y, puntos_z_altura, puntos_z_diff = [], [], []
        for (y, z) in puntos_comunes:
            diff = puntos_a[(y, z)] - puntos_b[(y, z)]
            puntos_y.append(y)
            puntos_z_altura.append(z)
            puntos_z_diff.append(diff)

        puntos_2d = np.vstack([puntos_y, puntos_z_altura]).T
        tri = Delaunay(puntos_2d)

        fig = go.Figure()

        # Superficie
        fig.add_trace(go.Mesh3d(
            x=puntos_y,
            y=puntos_z_altura,
            z=puntos_z_diff,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            intensity=puntos_z_diff,
            colorscale='Turbo',
            colorbar_title='Δ Presión [Pa]',
            name=f"Diferencia {nombre_a} - {nombre_b}",
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5, roughness=0.5, fresnel=0.2),
            lightposition=dict(x=100, y=200, z=100),
            hovertemplate='<b>Δ Presión</b>: %{intensity:.3f} Pa<br>Pos Y: %{x:.1f} mm<br>Altura Z: %{y:.1f} mm<extra></extra>'
        ))

        # Wireframe
        wire_y, wire_z, wire_presion = [], [], []
        for simplex in tri.simplices:
            for idx_pair in [(0,1), (1,2), (2,0)]:
                for idx in idx_pair:
                    wire_y.append(puntos_y[simplex[idx]])
                    wire_z.append(puntos_z_altura[simplex[idx]])
                    wire_presion.append(puntos_z_diff[simplex[idx]])
                wire_y.append(None)
                wire_z.append(None)
                wire_presion.append(None)

        fig.add_trace(go.Scatter3d(
            x=wire_y,
            y=wire_z,
            z=wire_presion,
            mode='lines',
            line=dict(color='black', width=1),
            name='Malla',
            hoverinfo='skip'
        ))

        # Puntos medidos
        if mostrar_puntos:
            fig.add_trace(go.Scatter3d(
                x=puntos_y,
                y=puntos_z_altura,
                z=puntos_z_diff,
                mode='markers',
                marker=dict(size=3, color='red'),
                name='Puntos medidos',
                hovertemplate='<b>Punto medido</b><br>Δ Presión: %{z:.3f} Pa<br>Pos Y: %{x:.1f} mm<br>Altura Z: %{y:.1f} mm<extra></extra>'
            ))

        fig.update_layout(
            title=f"Diferencia de Superficies 3D Mejorada - {nombre_a} - {nombre_b}",
            scene=dict(
                xaxis_title="Posición Y Traverser [mm]",
                yaxis_title="Altura Física Z [mm]",
                zaxis_title="Delta Presión [Pa]",
                aspectmode='data',
                aspectratio=dict(x=1, y=1, z=0.3),
                camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))
            ),
            width=1600,
            height=900,
            margin=dict(l=0, r=0, b=0, t=50)
        )

        return fig

    except Exception as e:
        st.error(f"Error creando la superficie de diferencia 3D mejorada: {str(e)}")
        return None




def crear_superficie_diferencia(datos_a, datos_b, nombre_a, nombre_b):
    """
    Resta dos superficies 3D: para cada (X,Y) común calcula la media de
    todas las columnas 'Presion-Sensor N' presentes en esa fila y resta.
    """
    coords_a = set(tuple(row) for row in datos_a[['Pos_Y_Traverser', 'Pos_Z_Base']].dropna().to_numpy())
    coords_b = set(tuple(row) for row in datos_b[['Pos_Y_Traverser', 'Pos_Z_Base']].dropna().to_numpy())
    coords_comunes = sorted(list(coords_a & coords_b))

    if len(coords_comunes) < 4:
        st.warning("No hay suficientes puntos comunes para generar una superficie.")
        return None

    X_final, Y_final, Z_final = [], [], []

    for y_trav, z_base in coords_comunes:
        fila_a = datos_a[(datos_a['Pos_Y_Traverser'] == y_trav) & (datos_a['Pos_Z_Base'] == z_base)]
        fila_b = datos_b[(datos_b['Pos_Y_Traverser'] == y_trav) & (datos_b['Pos_Z_Base'] == z_base)]

        if fila_a.empty or fila_b.empty:
            continue

        # detectar columnas de sensores en cada fila y promediar
        cols_a = [c for c in fila_a.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]
        cols_b = [c for c in fila_b.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]

        presiones_a = []
        for c in cols_a:
            try:
                val = fila_a.iloc[0][c]
                if isinstance(val, str):
                    val = float(val.replace(',', '.'))
                presiones_a.append(float(val))
            except:
                continue

        presiones_b = []
        for c in cols_b:
            try:
                val = fila_b.iloc[0][c]
                if isinstance(val, str):
                    val = float(val.replace(',', '.'))
                presiones_b.append(float(val))
            except:
                continue

        if presiones_a and presiones_b:
            diferencia = np.mean(presiones_a) - np.mean(presiones_b)
            X_final.append(y_trav)
            Y_final.append(z_base)
            Z_final.append(diferencia)

    if len(X_final) < 4:
        st.warning("No se pudieron generar suficientes puntos de diferencia para la superficie.")
        return None

    # Crear la malla
    X_unique = sorted(list(set(X_final)))
    Y_unique = sorted(list(set(Y_final)))
    Z_matrix = np.full((len(Y_unique), len(X_unique)), np.nan)

    for x, y, z in zip(X_final, Y_final, Z_final):
        iy = Y_unique.index(y)
        ix = X_unique.index(x)
        Z_matrix[iy, ix] = z

    X_mesh, Y_mesh = np.meshgrid(X_unique, Y_unique)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X_mesh, y=Y_mesh, z=Z_matrix,
        colorscale='RdBu_r',
        colorbar=dict(title="Diferencia de Presión [Pa]"),
        hovertemplate='<b>Diferencia de Presión</b><br>X: %{x:.1f} mm, Y: %{y:.1f} mm<br>Diferencia: %{z:.3f} Pa<extra></extra>'
    ))

    fig.update_layout(
        title=f"Diferencia de Superficies: {nombre_a} vs {nombre_b}",
        scene=dict(
            xaxis_title="Posición X [mm]",
            yaxis_title="Posición Y [mm]",
            zaxis_title="Diferencia de Presión [Pa]"
        ),
        font=dict(color="black")
    )
    fig.update_layout(width=1600, height=900, margin=dict(l=0, r=0, t=50, b=0))
    return fig

    
def crear_grafico_diferencia_areas(sub_archivo_a, sub_archivo_b, configuracion):
    """Crear gráfico mostrando la diferencia como UNA sola área"""
    
    # Extraer datos de ambos sub-archivos
    z_a, presion_a = extraer_datos_para_grafico(sub_archivo_a, configuracion)
    z_b, presion_b = extraer_datos_para_grafico(sub_archivo_b, configuracion)
    
    if not z_a or not z_b or not presion_a or not presion_b:
        return None, 0
    
    # Crear gráfico
    fig = go.Figure()
    
    # Agregar líneas de referencia (más tenues)
    fig.add_trace(go.Scatter(
        x=presion_a, y=z_a,
        mode='lines',
        name=f"{sub_archivo_a['archivo_fuente']} T{sub_archivo_a['tiempo']}s",
        line=dict(color='#08596C', width=2, dash='dot'),
        opacity=0.6,
        hovertemplate='<b>%{fullData.name}</b><br>' +
                    'Presión: %{x:.3f} Pa<br>' +
                    'Altura: %{y:.1f} mm<br>' +
                    '<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=presion_b, y=z_b,
        mode='lines',
        name=f"{sub_archivo_b['archivo_fuente']} T{sub_archivo_b['tiempo']}s",
        line=dict(color='#E74C3C', width=2, dash='dot'),
        opacity=0.6,
        hovertemplate='<b>%{fullData.name}</b><br>' +
                    'Presión: %{x:.3f} Pa<br>' +
                    'Altura: %{y:.1f} mm<br>' +
                    '<extra></extra>'
    ))
    
    # Calcular diferencia punto a punto (interpolando si es necesario)
    # Usar el rango de alturas común
    z_min = max(min(z_a), min(z_b))
    z_max = min(max(z_a), max(z_b))
    
    # Crear puntos interpolados
    z_interp = np.linspace(z_min, z_max, 50)
    
    # Interpolar presiones
    presion_a_interp = np.interp(z_interp, z_a, presion_a)
    presion_b_interp = np.interp(z_b, presion_b)
    
    # Calcular diferencia
    diferencia_presion = presion_a_interp - presion_b_interp
    
    # Crear área de diferencia ÚNICA
    # Determinar color basado en si la diferencia es mayormente positiva o negativa
    diferencia_promedio = np.mean(diferencia_presion)
    color_diferencia = '#27AE60' if diferencia_promedio >= 0 else '#E67E22'  # Verde si A>B, naranja si B>A
    
    # Crear área desde cero hasta la diferencia
    x_area = [0] + list(diferencia_presion) + [0]
    y_area = [z_interp[0]] + list(z_interp) + [z_interp[-1]]
    
    fig.add_trace(go.Scatter(
        x=x_area, y=y_area,
        fill='toself',
        fillcolor=f'rgba({int(color_diferencia[1:3], 16)}, {int(color_diferencia[3:5], 16)}, {int(color_diferencia[5:7], 16)}, 0.4)',
        line=dict(color=color_diferencia, width=3),
        name=f'Diferencia: {sub_archivo_a["archivo_fuente"]} - {sub_archivo_b["archivo_fuente"]}',
        hovertemplate='<b>Diferencia</b><br>' +
                    'Diferencia: %{x:.3f} Pa<br>' +
                    'Altura: %{y:.1f} mm<br>' +
                    '<extra></extra>'
    ))
    
    # Calcular área total de diferencia
    area_diferencia = np.trapz(np.abs(diferencia_presion), z_interp)
    
    # Layout CON LEYENDA MEJORADA
    fig.update_layout(
        title=f"Diferencia de Perfiles: {sub_archivo_a['archivo_fuente']} - {sub_archivo_b['archivo_fuente']}",
        xaxis_title="Presión / Diferencia de Presión [Pa]",
        yaxis_title="Altura z [mm]",
        height=700, width=1000,  # Más ancho para leyenda
        showlegend=True,  # FORZAR LEYENDA VISIBLE
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12),
        title_font=dict(size=16, color="#FFFFFF"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#000005",
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=2,
            scaleanchor="y",      # AGREGADO: Configuración solicitada
            scaleratio=4          # AGREGADO: Configuración solicitada
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#080000",
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=2
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#08596C',
            borderwidth=2,
            x=1.02,
            y=1,
            font=dict(size=12, color='black')  # AGREGAR COLOR NEGRO
        )
    )
    fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',   # Fondo transparente
    paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
    font=dict(color='white'),       # Texto en blanco
    xaxis_title="Presión Total [Pa]",
    yaxis_title="Altura Z [mm]",
    height=900,
    width=1600
    )
    return fig, area_diferencia

def mostrar_configuracion_sensores(section_key):
    """Muestra los widgets de configuración de sensores y guarda el estado."""
    st.markdown("###  Configuración de Sensores y Geometría")

    config_key = f'configuracion_{section_key}'
    if config_key not in st.session_state:
        st.session_state[config_key] = {}

    # Orden de sensores
    orden_sensores = st.selectbox(
        "Orden de lectura de sensores:", ["asc", "des"],
        format_func=lambda x: "Ascendente (sensor 1 abajo, 12 arriba)" if x == "asc" else "Descendente (sensor 12 abajo, 1 arriba)",
        help="Define cómo se leen los datos de los sensores en relación a su posición física.",
        key=f'orden_sensores_{section_key}'
    )
    
    st.info(" **Pregunta:** ¿Qué sensor corresponde a la toma número 12 (la que se encuentra cerca del piso)?")
    sensor_referencia = st.selectbox(
        "Sensor de referencia (toma 12):", [f"Sensor {i}" for i in range(1, 13)],
        index=11, help="Seleccione el sensor que corresponde a la toma física número 12.",
        key=f'sensor_ref_{section_key}'
    )
    
    distancia_toma_12 = st.number_input(
        "Distancia de la toma 12 a la posición X=0, Y=0 del traverser [mm]:",
        value=-120.0, step=1.0, format="%.1f",
        help="Distancia en mm desde el punto de referencia del traverser.",
        key=f'dist_toma_{section_key}'
    )
    
    distancia_entre_tomas = st.number_input(
        "Distancia entre tomas [mm]:", value=10.0, step=0.01, format="%.2f",
        help="Distancia física entre tomas consecutivas según el plano técnico.",
        key=f'dist_entre_{section_key}'
    )
    
    if st.button(f"💾 Guardar Configuración", type="primary", key=f'save_config_{section_key}'):
        st.session_state[config_key] = {
            'orden': orden_sensores,
            'sensor_referencia': sensor_referencia,
            'distancia_toma_12': distancia_toma_12,
            'distancia_entre_tomas': distancia_entre_tomas
        }
        st.success(f"✅ Configuración para la sección {section_key.upper()} guardada.")
        st.rerun()

    return st.session_state.get(config_key, {})

def crear_superficie_delaunay_3d(datos_completos, configuracion_3d, nombre_archivo, mostrar_puntos=True, variable='Presion Total'):
    """
    Crea una superficie 3D continua con Delaunay y mejoras visuales.
    Ahora permite activar/desactivar la visualización de puntos medidos y seleccionar vector a plotear.
    """
    try:
        posicion_inicial = configuracion_3d['distancia_toma_12']
        distancia_entre_tomas = configuracion_3d['distancia_entre_tomas']
        orden = configuracion_3d['orden']

        puntos_y, puntos_z_altura, presiones_z = [], [], []  # Cambio de nombres: x->y, y->z

        # Detectar columnas de sensores
        sensor_cols = [c for c in datos_completos.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]

        for _, fila in datos_completos.iterrows():
            y_traverser = fila.get('Pos_Y_Traverser', None) 
            z_base_ref = fila.get('Pos_Z_Base', None)
            if pd.isna(y_traverser) or pd.isna(z_base_ref):
                continue

            for col in sensor_cols:
                sensor_num = obtener_numero_sensor_desde_columna(col)
                if sensor_num is None:
                    continue
                altura_sensor_real = calcular_altura_absoluta_z(
                    sensor_num, z_base_ref, posicion_inicial, distancia_entre_tomas, orden
                )
                presion = fila.get(col, None)
                if pd.isna(presion) or presion is None:
                    continue
                try:
                    if isinstance(presion, str):
                        presion = float(presion.replace(',', '.'))
                    presion_val = float(presion)
                    # Calcular la variable seleccionada
                    valor_final = presion_val # Por defecto Presion Total
                    
                    if variable == 'P_t / Rho_inf':
                        rho = float(fila.get('rho_inf', 1.225)) if pd.notna(fila.get('rho_inf', 1.225)) else 1.225
                        valor_final = presion_val / rho if rho != 0 else 0
                    elif variable == 'Velocidad Infinito':
                        valor_final = float(fila.get('V_inf', 0.0)) if pd.notna(fila.get('V_inf', 0.0)) else 0.0
                    elif variable == 'Presion Infinito':
                        valor_final = float(fila.get('P_inf', 101325.0)) if pd.notna(fila.get('P_inf', 101325.0)) else 101325.0

                    puntos_y.append(y_traverser)  # Ahora es Y explicitamente
                    puntos_z_altura.append(altura_sensor_real)  # Ahora es Z explicitamente
                    presiones_z.append(valor_final)
                except (ValueError, TypeError):
                    continue

        if len(puntos_y) < 4:
            st.error("No hay suficientes datos válidos para generar una superficie.")
            return None

        # Triangulación Delaunay
        puntos_2d = np.vstack([puntos_y, puntos_z_altura]).T
        tri = Delaunay(puntos_2d)

        fig = go.Figure()

        # Superficie principal
        fig.add_trace(go.Mesh3d(
            x=puntos_y,  # Ahora Y en eje X del gráfico
            y=puntos_z_altura,  # Ahora Z en eje Y del gráfico
            z=presiones_z,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            intensity=presiones_z,
            colorscale='Turbo',
            colorbar_title='Presión [Pa]',
            name='Superficie de presión',
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5, roughness=0.5, fresnel=0.2),
            lightposition=dict(x=100, y=200, z=100),
            hovertemplate='<b>Presión</b>: %{intensity:.3f} Pa<br>Pos Y: %{x:.1f} mm<br>Altura Z: %{y:.1f} mm<extra></extra>'
        ))

        # Wireframe
        wire_y, wire_z, wire_presion = [], [], []
        for simplex in tri.simplices:
            for idx_pair in [(0,1), (1,2), (2,0)]:
                for idx in idx_pair:
                    wire_y.append(puntos_y[simplex[idx]])
                    wire_z.append(puntos_z_altura[simplex[idx]])
                    wire_presion.append(presiones_z[simplex[idx]])
                wire_y.append(None)
                wire_z.append(None)
                wire_presion.append(None)

        fig.add_trace(go.Scatter3d(
            x=wire_y,
            y=wire_z,
            z=wire_presion,
            mode='lines',
            line=dict(color='black', width=1),
            name='Malla',
            hoverinfo='skip'
        ))

        # Puntos medidos (si mostrar_puntos=True)
        if mostrar_puntos:
            fig.add_trace(go.Scatter3d(
                x=puntos_y,
                y=puntos_z_altura,
                z=presiones_z,
                mode='markers',
                marker=dict(size=3, color='red'),
                name='Puntos medidos',
                hovertemplate='<b>Punto medido</b><br>Presión: %{z:.3f} Pa<br>Pos Y: %{x:.1f} mm<br>Altura Z: %{y:.1f} mm<extra></extra>'
            ))

        fig.update_layout(
            title=f"Superficie de Presión 3D Mejorada - {nombre_archivo}",
            scene=dict(
                xaxis_title="Posición Y Traverser [mm]",  # Cambio de etiquetas
                yaxis_title="Altura Física Z [mm]",  # Cambio de etiquetas
                zaxis_title="Presión [Pa]",
                aspectmode='data',  # Configuración manual para escala 1:1
                aspectratio=dict(x=1, y=1, z=1),  # Relación 1:1 entre Y y Z
            ),
            width=1600,
            height=900,
            margin=dict(l=0, r=0, b=0, t=50)
        )

        return fig

    except Exception as e:
        st.error(f"Error creando la superficie de malla 3D mejorada: {str(e)}")
        return None

def unir_archivos_incertidumbre(archivos_lista, nombre_salida):
    """Une múltiples archivos de incertidumbre en uno solo"""
    try:
        contenido_unido = []
        puntos_sobrepuestos = []
        coordenadas_vistas = set()
        
        for archivo in archivos_lista:
            # Leer archivo CSV
            df_raw = pd.read_csv(archivo, sep=";", header=None, dtype=str)
            
            # Buscar la palabra "importante" para determinar dónde terminar
            index_final = df_raw[df_raw.apply(lambda row: row.astype(str).str.contains("importante", case=False).any(), axis=1)].index
            if not index_final.empty:
                df_raw = df_raw.iloc[:index_final[0]]
            
            # Procesar bloques de 10 filas
            for i in range(0, df_raw.shape[0], 10):
                bloque = df_raw.iloc[i:i+10]
                if bloque.empty or len(bloque) < 3:
                    continue
                
                nombre_archivo_bloque = bloque.iloc[0, 0]
                tiempo, y_trav, z_base = extraer_tiempo_y_coordenadas_YZ(nombre_archivo_bloque)
                
                # Verificar si hay puntos sobrepuestos
                coordenada_key = (y_trav, z_base, tiempo)
                if coordenada_key in coordenadas_vistas:
                    puntos_sobrepuestos.append(coordenada_key)
                else:
                    coordenadas_vistas.add(coordenada_key)
                
                # Agregar bloque al contenido unido
                contenido_unido.extend(bloque.values.tolist())
        
        # Crear DataFrame final
        df_unido = pd.DataFrame(contenido_unido)
        
        # Convertir a CSV
        csv_output = df_unido.to_csv(sep=';', index=False, header=False)
        
        return csv_output, puntos_sobrepuestos
        
    except Exception as e:
        st.error(f"Error al unir archivos: {str(e)}")
        return None, []

def extraer_matriz_presiones_completa(archivo_incertidumbre, configuracion, archivo_infinito=None):
    """
    Devuelve un DataFrame con columnas Y, Z, Presion
    listo para exportar como VTK estructurado.
    Unificada y corregida para usar lógica Y-Z.
    """
    try:
        # Procesar archivo CSV
        datos = procesar_promedios(archivo_incertidumbre, configuracion["orden"], archivo_infinito)
        if datos is None:
            return pd.DataFrame(columns=["Y", "Z", "Presion"])

        registros = []
        sensor_cols = [c for c in datos.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]

        for _, fila in datos.iterrows():
            y_trav = fila.get("Pos_Y_Traverser")
            z_base = fila.get("Pos_Z_Base") 
            if pd.isna(y_trav) or pd.isna(z_base):
                continue

            for col in sensor_cols:
                num = obtener_numero_sensor_desde_columna(col)
                if num is None:
                    continue

                z_real = calcular_altura_absoluta_z(
                    num,
                    z_base,
                    configuracion["distancia_toma_12"],
                    configuracion["distancia_entre_tomas"],
                    len(sensor_cols),
                    configuracion["orden"]
                )

                presion = fila.get(col, None)
                if presion is None or pd.isna(presion):
                    continue

                try:
                    presion_val = float(str(presion).replace(",", "."))
                    registros.append((float(y_trav), float(z_real), presion_val))
                except ValueError:
                    continue

        # Convertir a DataFrame ORDENADO
        df = pd.DataFrame(registros, columns=["Y", "Z", "Presion"])
        df = df.sort_values(by=["Y", "Z"]).reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Error al extraer matriz de presiones: {str(e)}")
        return pd.DataFrame(columns=["Y", "Z", "Presion"])


def crear_archivo_vtk_superficie_delaunay(df_matriz, nombre_archivo_vtk):
    """
    Genera un archivo VTK con superficie triangulada (Delaunay) CORREGIDO para Salome/ParaView.
    La matriz debe tener columnas: Y, Z, Presion.
    """
    try:
        # Extraer puntos y presiones
        puntos_y = df_matriz["Y"].astype(float).values
        puntos_z = df_matriz["Z"].astype(float).values
        presiones = df_matriz["Presion"].astype(float).values
        
        # Filtrar valores NaN
        mask = ~(np.isnan(puntos_y) | np.isnan(puntos_z) | np.isnan(presiones))
        puntos_y = puntos_y[mask]
        puntos_z = puntos_z[mask]
        presiones = presiones[mask]

        if len(puntos_y) < 4:
            print("Error: No hay suficientes puntos válidos para generar una superficie triangulada.")
            return None

        # Triangulación en el plano (Y,Z)
        puntos_2d = np.column_stack([puntos_y, puntos_z])
        tri = Delaunay(puntos_2d)

        n_points = len(puntos_y)
        n_triangles = len(tri.simplices)

        # Crear contenido VTK con formato correcto
        vtk_content = "# vtk DataFile Version 3.0\n"
        vtk_content += "Superficie Delaunay - Matriz de Presion\n"
        vtk_content += "ASCII\n"
        vtk_content += "DATASET UNSTRUCTURED_GRID\n"
        vtk_content += f"POINTS {n_points} float\n"

        # CORRECCIÓN: Usar coordenadas Y, Z y presión como altura Z
        for i in range(n_points):
            vtk_content += f"{puntos_y[i]:.6f} {puntos_z[i]:.6f} {presiones[i]:.6f}\n"

        # Definir celdas triangulares
        vtk_content += f"\nCELLS {n_triangles} {4*n_triangles}\n"
        for simplex in tri.simplices:
            vtk_content += f"3 {simplex[0]} {simplex[1]} {simplex[2]}\n"

        # Tipos de celda (5 = VTK_TRIANGLE)
        vtk_content += f"\nCELL_TYPES {n_triangles}\n"
        for _ in range(n_triangles):
            vtk_content += "5\n"

        # Datos escalares en puntos
        vtk_content += f"\nPOINT_DATA {n_points}\n"
        vtk_content += "SCALARS Presion float 1\n"
        vtk_content += "LOOKUP_TABLE default\n"
        for p in presiones:
            vtk_content += f"{p:.6f}\n"

        # Coordenadas adicionales como campos escalares
        vtk_content += "\nSCALARS Coord_Y float 1\n"
        vtk_content += "LOOKUP_TABLE default\n"
        for y in puntos_y:
            vtk_content += f"{y:.6f}\n"

        vtk_content += "\nSCALARS Coord_Z float 1\n"
        vtk_content += "LOOKUP_TABLE default\n"
        for z in puntos_z:
            vtk_content += f"{z:.6f}\n"

        # Guardar archivo
        with open(nombre_archivo_vtk, "w", encoding="utf-8") as f:
            f.write(vtk_content)

        print(f"✅ Archivo VTK creado exitosamente: {nombre_archivo_vtk}")
        print(f"Superficie con {n_points} puntos y {n_triangles} triángulos")
        
        return nombre_archivo_vtk

    except Exception as e:
        print(f"  Error al crear archivo VTK de superficie Delaunay: {str(e)}")
        return None

def crear_sub_archivos_3d_por_tiempo_y_posicion(df_datos, nombre_archivo):
    """Crear sub-archivos 3D por tiempo y posición (similar a 2D)"""
    sub_archivos = {}
    
    # Obtener tiempos únicos
    tiempos_unicos = df_datos["Tiempo_s"].dropna().unique()
    
    for tiempo in tiempos_unicos:
        # Filtrar datos por tiempo
        df_tiempo = df_datos[df_datos["Tiempo_s"] == tiempo].copy()
        
        # Crear clave para el sub-archivo
        clave_sub_archivo = f"{nombre_archivo}_T{tiempo}s"

        sub_archivos[clave_sub_archivo] = {
            'archivo_fuente': nombre_archivo,
            'tiempo': tiempo,
            'datos': df_tiempo,
            'nombre_archivo': f"{nombre_archivo}_T{tiempo}s.csv"
        }
    
    return sub_archivos

def mostrar_resumen_archivos_tabla(sub_archivos_por_fuente):
    """Mostrar resumen de archivos en formato tabla organizada"""
    st.markdown("### 📊 Resumen de Sub-archivos Generados")
    
    # Crear datos para la tabla
    datos_tabla = []
    
    for archivo_fuente, tiempos_dict in sub_archivos_por_fuente.items():
        for tiempo, sub_archivos_tiempo in sub_archivos_tiempo:
            for clave, sub_archivo in sub_archivos_tiempo:
                datos_tabla.append({
                    'Archivo_Fuente': archivo_fuente,
                    'Tiempo_s': f"T{tiempo}s", 
                    'Posición_X': sub_archivo['x_traverser'],
                    'Registros': len(sub_archivo['datos']),
                    'Nombre_Archivo': sub_archivo['nombre_archivo'],
                    'Clave': clave
                })
    
    # Crear DataFrame y mostrar como tabla
    df_resumen = pd.DataFrame(datos_tabla).sort_values(['Archivo_Fuente', 'Posición_X', 'Tiempo_s'])
    
    # NUEVO: Mostrar tabla con separadores correctos para CSV
    st.dataframe(
        df_resumen[['Archivo_Fuente', 'Tiempo_s', 'Registros', 'Nombre_Archivo']], 
        use_container_width=True,
        hide_index=True
    )
    
    # NUEVO: Botón para descargar tabla como CSV bien formateado
    csv_tabla = df_resumen[['Archivo_Fuente', 'Tiempo_s', 'Registros', 'Nombre_Archivo']].to_csv(
        index=False, 
        sep=';',  # CAMBIAR a punto y coma para Excel
        encoding='utf-8-sig',  # CAMBIAR encoding para Excel
        decimal=','  # AGREGAR separador decimal para Excel
    )
    
    st.download_button(
        label="📥 Descargar Tabla Resumen (CSV)",
        data=csv_tabla,
        file_name=f"resumen_subarchivos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Estadísticas adicionales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sub-archivos", len(datos_tabla))
    with col2:
        st.metric("Archivos Fuente", len(sub_archivos_por_fuente))
    with col3:
        total_registros = sum([item['Registros'] for item in datos_tabla])
        st.metric("Total Registros", total_registros)

def cargar_matriz_para_vtk():
    if opcion_matriz == "Usar matriz existente" and matriz_disponible:
        st.info(f"Usando matriz existente: {matriz_disponible['nombre']}")
        return matriz_disponible['matriz']

    elif opcion_matriz == "Cargar nuevo archivo" and archivo_vtk_source:
        with st.spinner("Extrayendo matriz para VTK..."):
            try:
                    # Intentar leer como matriz ya procesada
                df = pd.read_csv(archivo_vtk_source)
                if {"Y", "Z", "Presion"}.issubset(df.columns):
                    return df
                else:
                    # Si no tiene las columnas, asumir que es archivo de sensores crudos
                    return extraer_matriz_presiones_completa(
                        archivo_vtk_source,
                        st.session_state.get("configuracion_inicial", {})
                    )
            except Exception as e:
                st.error(f"Error al cargar CSV: {e}")
                return None

    return None

def crear_archivo_vtk_interpolado(df_matriz, nombre_base, resolucion_factor=2, metadata=None, posicion_x=0.0):
    """
    Genera un archivo VTK ESTRUCTURADA con interpolación Cúbica (Spline).
    Permite aumentar la resolución de la grilla para suavizar la visualización.
    NOTA: Se removió la metadata para evitar errores de compatibilidad.
    """
    try:
        if df_matriz.empty or not {"Y", "Z", "Presion"}.issubset(df_matriz.columns):
            st.error("El DataFrame para VTK está vacío o no tiene las columnas requeridas (Y, Z, Presion).")
            return None

        # Obtener valores únicos
        y_vals = sorted(df_matriz['Y'].unique())
        z_vals = sorted(df_matriz['Z'].unique())
        
        # Grid original
        Y_orig, Z_orig = np.meshgrid(y_vals, z_vals, indexing='ij')
        
        # Puntos y valores conocidos
        puntos_conocidos = df_matriz[['Y', 'Z']].values
        valores_conocidos = df_matriz['Presion'].values

        # Crear nueva grilla de mayor resolución
        y_new = np.linspace(min(y_vals), max(y_vals), len(y_vals) * resolucion_factor)
        z_new = np.linspace(min(z_vals), max(z_vals), len(z_vals) * resolucion_factor)
        ny, nz = len(y_new), len(z_new)
        
        Y_grid_new, Z_grid_new = np.meshgrid(y_new, z_new, indexing='ij')
        
        # Interpolación Cúbica
        P_grid_new = griddata(
            puntos_conocidos, 
            valores_conocidos, 
            (Y_grid_new, Z_grid_new), 
            method='cubic', 
            fill_value=0.0
        )
        
        # Rellenar nans en bordes con nearest si es necesario, o dejar 0
        mask_nan = np.isnan(P_grid_new)
        if mask_nan.any():
             P_grid_new[mask_nan] = griddata(
                puntos_conocidos, 
                valores_conocidos, 
                (Y_grid_new[mask_nan], Z_grid_new[mask_nan]), 
                method='nearest'
            )

        nombre_archivo = f"{nombre_base}_interpolado_cubic.vtk"

        # Cabecera VTK
        lines = []
        lines.append("# vtk DataFile Version 3.0")
        lines.append("Campo de Presiones Interpolado (Cubic Spline)")
        lines.append("ASCII")
        lines.append("DATASET STRUCTURED_GRID")
        lines.append(f"DIMENSIONS 1 {ny} {nz}")
        lines.append(f"POINTS {ny * nz} float")

        # Escribir puntos (Planar YZ -> X=0)
        for j in range(nz):
            for i in range(ny):
                lines.append(f"{posicion_x:.6f} {y_new[i]:.6f} {z_new[j]:.6f}")

        # POINT DATA
        lines.append(f"\nPOINT_DATA {ny * nz}")
        lines.append("SCALARS Presion float 1")
        lines.append("LOOKUP_TABLE default")
        for j in range(nz):
            for i in range(ny):
                lines.append(f"{P_grid_new[i, j]:.6f}")

        # METADATA REMOVED PER USER REQUEST TO AVOID ERRORS

        with open(nombre_archivo, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        st.success(f"✅ Archivo VTK Avanzado creado: {nombre_archivo}")
        return nombre_archivo

    except Exception as e:
        st.error(f"  Error al crear el archivo VTK interpolado: {str(e)}")
        return None



def crear_vtk_superficie_3d_delaunay(df_matriz, nombre_base, posicion_x=0.0):
    """
    Genera un archivo VTK con una superficie 3D (malla no estructurada)
    usando triangulación de Delaunay.
    La base de la superficie está en el plano YZ y la presión se representa
    como la coordenada en el eje X.
    """
    try:
        if df_matriz.empty or not {"Y", "Z", "Presion"}.issubset(df_matriz.columns):
            st.error("El DataFrame para VTK 3D está vacío o no tiene las columnas requeridas.")
            return None

        # Extraer y limpiar los datos
        puntos_y = df_matriz["Y"].values
        puntos_z = df_matriz["Z"].values
        presiones = df_matriz["Presion"].values
        
        mask = ~np.isnan(puntos_y) & ~np.isnan(puntos_z) & ~np.isnan(presiones)
        puntos_y, puntos_z, presiones = puntos_y[mask], puntos_z[mask], presiones[mask]

        if len(puntos_y) < 3:
            st.error("Se necesitan al menos 3 puntos válidos para la triangulación.")
            return None

        # La triangulación se sigue haciendo en el plano YZ
        puntos_2d_plano = np.column_stack([puntos_y, puntos_z])
        tri = Delaunay(puntos_2d_plano)

        n_points = len(puntos_y)
        n_triangles = len(tri.simplices)
        nombre_archivo_vtk = f"{nombre_base}_superficie_3D_presion_en_X.vtk"

        # Contenido del archivo VTK
        vtk_content = "# vtk DataFile Version 3.0\n"
        vtk_content += "Superficie de Presion 3D (Plano YZ, Presion en X)\n"
        vtk_content += "ASCII\n"
        vtk_content += "DATASET UNSTRUCTURED_GRID\n"
        vtk_content += f"POINTS {n_points} float\n"

        for i in range(n_points):
            # ---------------------------------------------------------------- #
            # ¡CAMBIO CLAVE! AHORA SE AÑADE LA POSICIÓN X A LA PRESIÓN (Estilo 4D)
            # Formato (X, Y, Z) -> (posicion_x + Presion_valor, Y_coord, Z_coord)
            x_def = posicion_x + presiones[i]
            vtk_content += f"{x_def:.6f} {puntos_y[i]:.6f} {puntos_z[i]:.6f}\n"
            # ---------------------------------------------------------------- #

        # El resto de la función (celdas y datos) permanece igual
        vtk_content += f"\nCELLS {n_triangles} {4 * n_triangles}\n"
        for simplex in tri.simplices:
            vtk_content += f"3 {simplex[0]} {simplex[1]} {simplex[2]}\n"

        vtk_content += f"\nCELL_TYPES {n_triangles}\n"
        vtk_content += "5\n" * n_triangles

        vtk_content += f"\nPOINT_DATA {n_points}\n"
        vtk_content += "SCALARS Presion float 1\n"
        vtk_content += "LOOKUP_TABLE default\n"
        for p in presiones:
            vtk_content += f"{p:.6f}\n"
            
        with open(nombre_archivo_vtk, "w", encoding="ascii") as f:
            f.write(vtk_content)
            
        st.success(f"✅ Archivo VTK 3D (Presión en Eje X) creado: {nombre_archivo_vtk}")
        return nombre_archivo_vtk

    except Exception as e:
        st.error(f"  Error al crear el archivo VTK de superficie 3D: {str(e)}")
        return None

# ---------------------------------------------------------------------------
# FUNCIÓN: VTK PLANO DE PRESIÓN 2D
# Genera un archivo VTK ESTRUCTURADA plano (sin deformación en X).
# La presión se guarda SOLO como dato escalar (color), no como geometría.
# ---------------------------------------------------------------------------
def crear_vtk_plano_presion_2d(df_matriz, nombre_base, posicion_x=0.0):
    """
    Genera un VTK 'STRUCTURED_GRID' plano en el plano YZ (X fijo = posicion_x).
    La presión se codifica ÚNICAMENTE como escalar de color (POINT_DATA),
    sin ninguna deformación geométrica. Ideal para visualizar contornos de
    presión en ParaView con colormaps.
    """
    try:
        if df_matriz.empty or not {"Y", "Z", "Presion"}.issubset(df_matriz.columns):
            st.error("El DataFrame para VTK Plano está vacío o le faltan columnas (Y, Z, Presion).")
            return None

        y_vals = sorted(df_matriz['Y'].unique())
        z_vals = sorted(df_matriz['Z'].unique())
        ny, nz = len(y_vals), len(z_vals)

        # Mapa rápido (Y, Z) → Presion
        presion_map = {(float(row['Y']), float(row['Z'])): float(row['Presion'])
                       for _, row in df_matriz.iterrows()
                       if not (pd.isna(row['Y']) or pd.isna(row['Z']) or pd.isna(row['Presion']))}

        lines = []
        lines.append("# vtk DataFile Version 3.0")
        lines.append("Plano de Presion 2D (YZ) - Solo Color")
        lines.append("ASCII")
        lines.append("DATASET STRUCTURED_GRID")
        lines.append(f"DIMENSIONS 1 {ny} {nz}")
        lines.append(f"POINTS {ny * nz} float")

        presiones_ordenadas = []
        for z in z_vals:
            for y in y_vals:
                lines.append(f"{posicion_x:.6f} {y:.6f} {z:.6f}")
                # Presión para este punto (0.0 si no encontrado)
                presiones_ordenadas.append(presion_map.get((float(y), float(z)), 0.0))

        lines.append(f"\nPOINT_DATA {ny * nz}")
        lines.append("SCALARS Presion float 1")
        lines.append("LOOKUP_TABLE default")
        for p in presiones_ordenadas:
            lines.append(f"{p:.6f}")

        vtk_str = "\n".join(lines)
        nombre_archivo = f"{nombre_base}_plano2D_X{int(posicion_x)}.vtk"

        with open(nombre_archivo, "w", encoding="utf-8") as f:
            f.write(vtk_str)

        st.success(f"✅ VTK Plano 2D creado: {nombre_archivo}")
        return nombre_archivo, vtk_str.encode('ascii')

    except Exception as e:
        st.error(f"  Error al crear VTK Plano 2D: {str(e)}")
        return None


# Sidebar (Legacy removed)

# Contenido principal según la sección
if st.session_state.seccion_actual == 'inicio':
    # --- LECTURA DE IMÁGENES CAROUSEL ---
    folder_portada = "Imagenes de portada"
    img_b64_list = []
    if os.path.exists(folder_portada):
        valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
        import random
        archivos = [f for f in os.listdir(folder_portada) if os.path.splitext(f)[1].lower() in valid_exts]
        random.shuffle(archivos)
        for f in archivos:
            try:
                with open(os.path.join(folder_portada, f), 'rb') as img_f:
                    img_b64_list.append(base64.b64encode(img_f.read()).decode())
            except:
                pass
    
    # Si no hay imágenes, poner una predeterminada
    if not img_b64_list:
        fallback_url = 'https://images.unsplash.com/photo-1517976487492-5750f3195933?q=80&w=2070&auto=format&fit=crop'
        carousel_css = f".hero-bg-0 {{ background-image: linear-gradient(to bottom, rgba(0,0,0,0.3), rgba(0,0,0,1)), url('{fallback_url}'); opacity: 1; }}"
        carousel_html = '<div class="hero-bg hero-bg-0"></div>'
    else:
        num_imgs = len(img_b64_list)
        if num_imgs == 1:
            # Caso especial si hay 1 sola imagen: mostrar fija sin crossfade extra
            carousel_css = f"""
            .hero-bg-0 {{
                background-image: linear-gradient(to bottom, rgba(0,0,0,0.3), rgba(0,0,0,1)), url('data:image/jpeg;base64,{img_b64_list[0]}');
                opacity: 1;
            }}
            """
            carousel_html = '<div class="hero-bg hero-bg-0"></div>'
        else:
            time_per_slide = 5 # segundos que dura cada imagen en pantalla
            total_time = num_imgs * time_per_slide
            percent_visible = 100.0 / num_imgs
            fade_percent = percent_visible * 0.15 # 15% del lapso usado en transicion suave
            
            carousel_css = ""
            carousel_html = ""
            for i, b64 in enumerate(img_b64_list):
                # Calcular timestamps de opacidad
                p_start = (i * percent_visible)
                p_in = p_start + fade_percent
                p_out = ((i+1) * percent_visible) - fade_percent
                p_end = ((i+1) * percent_visible)
                
                carousel_css += f"""
                .hero-bg-{i} {{
                    background-image: linear-gradient(to bottom, rgba(0,0,0,0.3), rgba(0,0,0,1)), url('data:image/jpeg;base64,{b64}');
                    animation: fadeCarousel{i} {total_time}s infinite;
                    opacity: 0;
                }}
                @keyframes fadeCarousel{i} {{
                    0%, {max(0, p_start-0.01)}% {{ opacity: 0; }}
                    {p_start}% {{ opacity: 0; }}
                    {p_in}% {{ opacity: 1; }}
                    {p_out}% {{ opacity: 1; }}
                    {p_end}% {{ opacity: 0; }}
                    100% {{ opacity: 0; }}
                }}
                """
                carousel_html += f'<div class="hero-bg hero-bg-{i}"></div>'

    # --- HERO SECTION (SpaceX Style) ---
    st.markdown(f"""
    <style>
        .hero-container {{
            position: relative;
            width: 100%;
            padding: 4rem 1rem;
            min-height: 80vh; /* Dimensión aumentada para mayor inmersión */
            border-radius: 0px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            margin-bottom: 3rem;
            border: 1px solid #333;
            overflow: hidden;
        }}
        
        .hero-bg {{
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background-size: cover;
            background-position: center;
            z-index: 0;
            transition: opacity 1s ease-in-out;
        }}
        
        {carousel_css}
        
        .hero-content {{
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .hero-title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 4.5rem;
            font-weight: 900;
            letter-spacing: 2px;
            margin-bottom: 0.5rem;
            text-shadow: 0 10px 30px rgba(0,0,0,0.5);
            color: white;
        }}
        
        .hero-subtitle {{
            font-family: 'Inter', sans-serif;
            font-size: 1.2rem;
            letter-spacing: 6px;
            text-transform: uppercase;
            color: rgba(255,255,255,0.8);
            margin-top: 0.5rem;
            text-shadow: 0 4px 15px rgba(0,0,0,0.8);
        }}
        
        .scroll-indicator {{
            margin-top: 4rem; 
            opacity: 0.7; 
            font-size: 0.8rem;
            animation: bounce 2s infinite;
            text-shadow: 0 2px 5px rgba(0,0,0,1);
        }}
        
        @keyframes bounce {{
            0%, 20%, 50%, 80%, 100% {{transform: translateY(0);}}
            40% {{transform: translateY(-10px);}}
            60% {{transform: translateY(-5px);}}
        }}
    </style>
    
    <div class="hero-container">
        {carousel_html}
        <div class="hero-content">
            <h1 class="hero-title">LABORATORIO</h1>
            <p class="hero-subtitle">Aerodinámica Experimental</p>
            <div class="scroll-indicator">
                ▼ DESLIZA PARA NAVEGAR
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- RENDER NAVBAR AFTER HERO ---
    render_navbar()
    
    # --- 📖 SECCIÓN 1: MANUAL DE USUARIO (SCROLLING) ---
    st.markdown("""
<div class="section-card">
<h2 style="border-bottom: 2px solid #4ade80; padding-bottom: 15px; margin-bottom: 20px;">📖 MANUAL DE USUARIO</h2>
<div class="manual-text">
<h4 class="manual-header">1. INTRODUCCIÓN</h4>
<p>Bienvenido al Sistema de Procesamiento de Datos del Túnel de Viento (App BETZ). Esta aplicación ha sido diseñada para automatizar el análisis de perfiles de presión, visualización de estelas y generación de superficies 4D.</p>

<h4 class="manual-header">2. VISUALIZACIÓN DE ESTELA 1D</h4>
<p>Este módulo permite analizar perfiles de presión en un corte unidimensional. El usuario debe cargar los archivos CSV de incertidumbre. El sistema detectará automáticamente el tiempo de medición y la posición del traverser.</p>
<p><strong>Configuración:</strong> Antes de cargar datos, asegúrese de configurar la referencia geométrica (distancia de toma 12, separación entre tomas) en el 'Paso 1'.</p>

<h4 class="manual-header">3. VISUALIZACIÓN DE ESTELA 2D</h4>
<p>Próximamente disponible.</p>

<h4 class="manual-header">4. VISUALIZACIÓN DE ESTELA 3D</h4>
<p>Permite la reconstrucción de volúmenes de presión a partir de múltiples barridos. Cargue múltiples archivos CSV correspondientes a diferentes estaciones o tiempos. El visualizador 3D generará una superficie interactiva.</p>

<h4 class="manual-header">5. VISUALIZACIÓN DE ESTELA 4D</h4>
<p>Genera progresiones espaciales y temporales interactuando con las superficies tridimensionales.</p>

<h4 class="manual-header">6. HERRAMIENTAS ADICIONALES</h4>
<p>En la sección inferior encontrará utilidades para unir archivos fraccionados, extraer matrices puras y convertir formatos para software CFD externo (ParaView, Salome).</p>

<br>
<p style="font-style: italic; color: #888;">(Este es un texto de ejemplo. Por favor, reemplace este contenido con el texto completo de su archivo Word 'Manual de Usuario.docx').</p>
</div>
</div>
""", unsafe_allow_html=True)
    
    # --- 🛠️ SECCIÓN 2: ACCESO A SECCIONES (GRID) ---
    st.markdown("<h2 style='margin-top: 4rem; margin-bottom: 2rem; text-align: center;'>NUESTRO TABLERO DE TRABAJO</h2>", unsafe_allow_html=True)
    
    # === GRUPO 1: ESTELAS ===
    st.markdown("<h3 style='margin-top: 1rem; color: #aaa; border-bottom: 1px solid #333; padding-bottom: 10px;'>🌌 ENSAYOS DE ESTELA</h3>", unsafe_allow_html=True)
    
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("""
        <div class="section-card" style="border-left: 5px solid #3b82f6; height: 160px; margin-bottom: 10px;">
            <h3 style="color: #3b82f6; margin-top: 0; margin-bottom: 10px;">📊 VIS. ESTELA 1D</h3>
            <p style="color: #bbb; font-size: 0.95rem; margin-bottom: 0;">Procesa archivos CSV de incertidumbre, calcula integrales de presión y permite exportar tabulados bidimensionales.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("ACCEDER AL MÓDULO 1D", key="btn_row_1d", use_container_width=True):
             st.session_state.seccion_actual = 'betz_2d'
             st.rerun()

    with r1c2:
        st.markdown("""
        <div class="section-card" style="border-left: 5px solid #06b6d4; height: 160px; margin-bottom: 10px;">
            <h3 style="color: #06b6d4; margin-top: 0; margin-bottom: 10px;">📈 VIS. ESTELA 2D</h3>
            <p style="color: #bbb; font-size: 0.95rem; margin-bottom: 0;">Sección en pleno desarrollo para procesamiento mejorado con distribuciones 2D.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("ACCEDER AL MÓDULO 2D", key="btn_row_2d_nueva", use_container_width=True):
             st.session_state.seccion_actual = 'vis_2d_nueva'
             st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("""
        <div class="section-card" style="border-left: 5px solid #8b5cf6; height: 160px; margin-bottom: 10px;">
            <h3 style="color: #8b5cf6; margin-top: 0; margin-bottom: 10px;">🌪️ VIS. ESTELA 3D</h3>
            <p style="color: #bbb; font-size: 0.95rem; margin-bottom: 0;">Reconstrucción 3D volumétrica a partir de cortes. Genera mallas VTK para exportación a CFD.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("ACCEDER AL MÓDULO 3D", key="btn_row_3d", use_container_width=True):
             st.session_state.seccion_actual = 'betz_3d'
             st.rerun()

    with r2c2:
        st.markdown("""
        <div class="section-card" style="border-left: 5px solid #ec4899; height: 160px; margin-bottom: 10px;">
            <h3 style="color: #ec4899; margin-top: 0; margin-bottom: 10px;">🌌 VIS. ESTELA 4D</h3>
            <p style="color: #bbb; font-size: 0.95rem; margin-bottom: 0;">Cálculo de progresiones espaciales y temporales interactuando con las superficies 3D.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("ACCEDER AL MÓDULO 4D", key="btn_row_4d", use_container_width=True):
             st.session_state.seccion_actual = 'betz_4d'
             st.rerun()

    # === HERRAMIENTAS (dentro del bloque de Ensayos de Estela) ===
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-card" style="border-left: 5px solid #10b981; height: 160px; margin-bottom: 10px;">
        <h3 style="color: #10b981; margin-top: 0; margin-bottom: 10px;">🔧 HERRAMIENTAS DE ESTELA</h3>
        <p style="color: #bbb; font-size: 0.95rem; margin-bottom: 0;">Utilidades para unir archivos fragmentados, extraer matrices de presión y generar archivos VTK (2D, 3D y 4D) para ParaView / Salome.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("ACCEDER A HERRAMIENTAS", key="btn_row_herr", use_container_width=True):
         st.session_state.seccion_actual = 'herramientas'
         st.rerun()

    # === GRUPO 2: BETZ ===
    st.markdown("<h3 style='margin-top: 3rem; color: #aaa; border-bottom: 1px solid #333; padding-bottom: 10px;'>🧪 ENSAYOS DE BETZ</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-card" style="border-left: 5px solid #f59e0b; height: 150px; margin-bottom: 10px;">
        <h3 style="color: #f59e0b; margin-top: 0; margin-bottom: 10px;">🧪 ENSAYO DE ALAS (BETZ)</h3>
        <p style="color: #bbb; font-size: 0.95rem; margin-bottom: 0;">Sección para el estudio y predicción de estela de perfiles de ala. Actualmente en diseño técnico y de algoritmos.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("ACCEDER A ENSAYOS", key="btn_row_ensayo", use_container_width=True):
         st.session_state.seccion_actual = 'ensayo_betz'
         st.rerun()

    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#444; font-size: 0.8rem;'>UTN HAEDO // DEPARTAMENTO DE INGENIERÍA AERONÁUTICA</div>", unsafe_allow_html=True)



elif st.session_state.seccion_actual == 'betz_2d':
    if st.session_state.configuracion_inicial:
        st.markdown("# 📊 VISUALIZACIÓN DE ESTELA 1D - Análisis Unidimensional")
        st.markdown("Análisis de perfiles de presión concatenados con extracción automática de tiempo y coordenadas")
    # --- Inicializar variables persistentes ---
    if "datos_procesados_betz2d" not in st.session_state:
        st.session_state.datos_procesados_betz2d = {}
    if "sub_archivos_betz2d" not in st.session_state:
        st.session_state.sub_archivos_betz2d = {}
    if "uploaded_files_betz2d" not in st.session_state:
        st.session_state.uploaded_files_betz2d = []

    # Paso 1: Configuración inicial
    with st.expander("💾 PASO 1: Configuración de Geometría y Sensores", expanded=True):
        st.markdown("""
        <div class="section-card" style="margin-bottom: 12px;">
            <h3 style="margin-top:0; color:white;">💾 PASO 1: CONFIGURACIÓN INICIAL</h3>
            <p style="color:#bbb; margin-bottom:0;">
                Defina los parámetros físicos del peine de sensores y el sistema de adquisición para el entorno 1D.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_config, col_ref = st.columns([2, 1])
        
        with col_config:
            st.markdown("<div style='padding: 1rem; border: 1px solid #333; border-radius: 8px; background-color: #000;'>", unsafe_allow_html=True)
            orden_sensores = st.selectbox(
                "Orden de lectura de sensores",
                ["asc", "des"],
                format_func=lambda x: "Ascendente (Sensor 1 → 12)" if x == "asc" else "Descendente (Sensor 12 → 1)",
                help="Ascendente: Sensor 1 abajo, 12 arriba. Descendente: Sensor 12 abajo, 1 arriba.",
                key="orden_1d_std"
            )

            sensor_referencia = st.selectbox(
                "Sensor de referencia (Toma 12)",
                [f"Sensor {i}" for i in range(1, 37)],
                index=11,
                help="Sensor físico conectado a la toma número 12.",
                key="sensor_ref_1d_std"
            )

            c1, c2 = st.columns(2)
            with c1:
                distancia_toma_12 = st.number_input(
                    "Distancia Toma 12 [mm]",
                    value=-120.0, step=1.0, format="%.1f",
                    help="Posición relativa al cero del traverser.",
                    key="dist_12_1d_std"
                )
            with c2:
                distancia_entre_tomas = st.number_input(
                    "Sep. entre tomas [mm]",
                    value=10.91, step=0.01, format="%.2f",
                    help="Distancia física entre centros de tomas.",
                    key="dist_entre_1d_std"
                )

            if st.button("💾 CONFIRMAR CONFIGURACIÓN", type="primary", use_container_width=True, key="btn_conf_1d_std"):
                st.session_state.configuracion_inicial = {
                    'orden': orden_sensores,
                    'sensor_referencia': sensor_referencia,
                    'distancia_toma_12': distancia_toma_12,
                    'distancia_entre_tomas': distancia_entre_tomas
                }
                st.success("✅ Configuración 1D guardada.")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #111; border: 1px dashed #444; border-radius: 8px; padding: 1rem; text-align: center; height: 100%;">
            <p style="color: #888; font-size: 0.8rem; margin-bottom: 10px;">REFERENCIA TÉCNICA</p>
            <img src="https://raw.githubusercontent.com/Juan-Cruz-de-la-Fuente/Laboratorio/main/Peine.jpg" 
                 style="max-width: 100%; border-radius: 4px; opacity: 0.8;">
        </div>
        """, unsafe_allow_html=True)

    # Paso 2: Carga de archivos
    # --- PASO 2: CARGA Y PROCESAMIENTO ---
    if st.session_state.configuracion_inicial:
        st.markdown("---")
        st.markdown("""
        <div class="section-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: white;"> PASO 2: IMPORTACIÓN DE ARCHIVOS CRUDOS</h3>
            <p style="color: #bbb; margin-bottom: 20px;">Cargue los archivos CSV de incertidumbre generados por el sistema de adquisición.</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Arrastre sus archivos CSV aquí",
            type=['csv'],
            accept_multiple_files=True,
            key="uploader_betz2d"
        )

        if uploaded_files:
            st.session_state.uploaded_files_betz2d = uploaded_files
        else:
            uploaded_files = st.session_state.uploaded_files_betz2d

        # Procesamiento y Generación de Salidas
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} archivos cargados en memoria.")
            
            # Contenedor de procesados
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            
            for archivo in uploaded_files:
                if archivo.name not in st.session_state.datos_procesados_betz2d:
                    with st.spinner(f"🔨 Procesando {archivo.name}..."):
                        datos = procesar_promedios(archivo, st.session_state.configuracion_inicial['orden'])
                        if datos is not None:
                            st.session_state.datos_procesados_betz2d[archivo.name] = datos
                            st.session_state.datos_procesados[archivo.name] = datos
                            
                            sub_archivos = crear_archivos_individuales_por_tiempo_y_posicion(datos, archivo.name)
                            if 'sub_archivos_generados' not in st.session_state:
                                st.session_state.sub_archivos_generados = {}
                            st.session_state.sub_archivos_generados.update(sub_archivos)
                
                # Mostrar tarjeta de archivo procesado con descargas rápidas
                if archivo.name in st.session_state.datos_procesados_betz2d:
                    datos = st.session_state.datos_procesados_betz2d[archivo.name]
                    nombre_base = extraer_nombre_base_archivo(archivo.name)
                    
                    with st.expander(f"✅ {archivo.name} (Procesado)", expanded=False):
                        c_d1, c_d2, c_d3 = st.columns(3)
                        
                        # Preparar datos
                        datos_ordenados_nombre = datos.sort_values("Archivo")
                        datos_ordenados_x = datos.sort_values("Pos_Y_Traverser")
                        datos_ordenados_tiempo = datos.sort_values("Tiempo_s")
                        
                        with c_d1:
                             csv_pk = datos_ordenados_nombre.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')
                             st.download_button("📥 Ord. por Nombre", csv_pk, f"{nombre_base}_nombre.csv", "text/csv", key=f"dn_{nombre_base}")
                        with c_d2:
                             csv_x = datos_ordenados_x.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')
                             st.download_button("📥 Ord. por Posición X", csv_x, f"{nombre_base}_x.csv", "text/csv", key=f"dx_{nombre_base}")
                        with c_d3:
                             csv_t = datos_ordenados_tiempo.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')
                             st.download_button("📥 Ord. por Tiempo", csv_t, f"{nombre_base}_time.csv", "text/csv", key=f"dt_{nombre_base}")
    else:
        st.info("⚠️ Configure la geometría en el Paso 1 para habilitar la carga de archivos.")


    # Mostrar datos procesados
    if st.session_state.datos_procesados:
        st.markdown("## 📋 Datos Procesados")
        for nombre_archivo, datos in st.session_state.datos_procesados.items():
            with st.expander(f"Ver datos de {nombre_archivo}"):
                st.dataframe(datos, use_container_width=True)
                tiempos_unicos = datos['Tiempo_s'].dropna().unique()
                coordenadas_unicas = datos[['Pos_Y_Traverser', 'Pos_Z_Base']].drop_duplicates()
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Tiempos encontrados:** {sorted(tiempos_unicos)}")
                with col2:
                    st.markdown("**Coordenadas encontradas:**")
                    st.dataframe(coordenadas_unicas, use_container_width=True)


    # --- PASO 3: ANÁLISIS DETALLADO ---
    if st.session_state.sub_archivos_generados:
        st.markdown("---")
        st.markdown("""
        <div class="section-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: white;">📊 PASO 3: RESUMEN DE SUB-ARCHIVOS</h3>
            <p style="color: #bbb; margin-bottom: 10px;">Gestión de todos los archivos individualizados generados (separados por tiempo y posición).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recolectar datos
        filas_resumen = []
        for clave, sub in st.session_state.sub_archivos_generados.items():
            filas_resumen.append({
                'Archivo Fuente': sub.get('archivo_origen', sub.get('archivo_fuente', '-')),
                'Tiempo': f"{sub.get('tiempo')}s",
                'Pos X': sub.get('x_traverser'),
                'Registros': len(sub.get('datos', [])),
                'Nombre Salida': sub.get('nombre_archivo', ''),
                'Clave': clave
            })
        
        df_resumen = pd.DataFrame(filas_resumen)
        
        # Métricas
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"<div style='text-align:center; padding:10px; background:#111; border-radius:8px; border:1px solid #333;'><h2 style='margin:0; color:#3b82f6;'>{len(df_resumen)}</h2><p style='margin:0; color:#888;'>Sub-archivos Generados</p></div>", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"<div style='text-align:center; padding:10px; background:#111; border-radius:8px; border:1px solid #333;'><h2 style='margin:0; color:#10b981;'>{df_resumen['Archivo Fuente'].nunique()}</h2><p style='margin:0; color:#888;'>Archivos Fuente</p></div>", unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"<div style='text-align:center; padding:10px; background:#111; border-radius:8px; border:1px solid #333;'><h2 style='margin:0; color:#8b5cf6;'>{df_resumen['Registros'].sum()}</h2><p style='margin:0; color:#888;'>Total Registros</p></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            df_resumen[['Archivo Fuente', 'Tiempo', 'Pos X', 'Registros', 'Nombre Salida']], 
            use_container_width=True,
            height=300
        )

        st.markdown("---")

        # --- 4) Agrupar por archivo origen para mostrar expanders ---
        grouped = {}
        for _, row in df_resumen.iterrows():
            archivo = row['Archivo Fuente']
            tiempo = row['Tiempo']
            grouped.setdefault(archivo, {}).setdefault(tiempo, []).append(row.to_dict())

        # Mostrar 1 expander por archivo origen
        for archivo_origen, tiempos_dict in grouped.items():
            num_tiempos = len(tiempos_dict)
            with st.expander(f" {archivo_origen} - {num_tiempos} tiempos", expanded=False):
                # Generar ZIP con todos los sub-archivos de este origen (para descargar de una)
                # Crearlo aquí en memoria y mostrar botón
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for tiempo, items in tiempos_dict.items():
                        for it in items:
                            clave = it['Clave']
                            df_sub = st.session_state.sub_archivos_generados[clave]['datos']
                            csv_bytes = df_sub.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')
                            zf.writestr(it['Nombre Salida'], csv_bytes)
                zip_bytes = buffer.getvalue()
                st.download_button(
                    label=f"📦 Descargar TODOS los sub-archivos ({archivo_origen}) .zip",
                    data=zip_bytes,
                    file_name=f"{archivo_origen}_subarchivos_{datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip",
                    key=f"zip_{archivo_origen}_{datetime.now().timestamp()}"
                )

                st.markdown("")

                # Para cada tiempo, listar sub-archivos (X) y dar botón CSV por cada uno
                for tiempo, items in sorted(tiempos_dict.items()):
                    st.markdown(f"####   {tiempo}")
                    for it in sorted(items, key=lambda r: (r['Pos X'] if pd.notna(r['Pos X']) else 1e9)):
                        clave = it['Clave']
                        nombre = it['Nombre Salida']
                        registros = it['Registros']
                        pos_x = it['Pos X']

                        col_a, col_b, col_c = st.columns([4, 1, 2])
                        col_a.markdown(f"**X{pos_x}** — `{nombre}`")
                        col_b.markdown(f"Registros: {registros}")

                        # generar bytes CSV para el botón
                        df_sub = st.session_state.sub_archivos_generados[clave]['datos']
                        csv_bytes = df_sub.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')

                        # key única por descarga (clave ya debería ser única)
                        dl_key = f"dl_{clave}_{datetime.now().timestamp()}"
                        col_c.download_button(
                            label="📥 Descargar CSV",
                            data=csv_bytes,
                            file_name=nombre,
                            mime="text/csv",
                            key=dl_key,
                            use_container_width=True
                        )

                    st.markdown("---")
    else:
        st.info("No hay sub-archivos generados aún. Subí y procesá archivos en Paso 2.")

    # --- PASO 5: GUARDAR EN DRIVE (1D) ---
    if st.session_state.sub_archivos_generados:
        st.markdown("---")
        st.markdown("""
        <div class="section-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: white;">&#x2601;&#xFE0F; PASO 5: GUARDAR EN DRIVE (1D)</h3>
            <p style="color: #bbb; margin-bottom: 0;">Sube un sub-archivo procesado directamente a la carpeta <b>ENSAYO DE ESTELA / 1D</b> de tu Drive.</p>
        </div>
        """, unsafe_allow_html=True)

        opciones_1d = list(st.session_state.sub_archivos_generados.keys())
        clave_sel_1d = st.selectbox("Seleccionar sub-archivo para subir:", opciones_1d, key="sel_drive_1d")

        if clave_sel_1d:
            sub_sel = st.session_state.sub_archivos_generados[clave_sel_1d]
            df_sub_sel = sub_sel['datos']
            nombre_sub_sel = sub_sel.get('nombre_archivo', f"{clave_sel_1d}.csv")
            csv_bytes_1d = df_sub_sel.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')

            col_1d_dl, col_1d_drive = st.columns(2)
            with col_1d_dl:
                st.download_button("&#x1F4E5; Descargar CSV", csv_bytes_1d,
                                   file_name=nombre_sub_sel, mime="text/csv", key="dl_1d_final")
            with col_1d_drive:
                if st.button("&#x2601;&#xFE0F; Guardar en Drive (1D)", key="save_1d_drive", use_container_width=True):
                    if auth.save_csv_1d(st.session_state.username, nombre_sub_sel, csv_bytes_1d):
                        st.success(f"✅ Subido a Drive → ENSAYO DE ESTELA/1D/{nombre_sub_sel}")
                    else:
                        st.error("Error al subir a Drive")

    # Paso 4: Sección de Gráficos
    st.markdown("## 📈 Paso 4: Sección de Gráficos")

    if st.session_state.datos_procesados:
        # Tomar el primer DataFrame procesado para obtener n_sensores
        primer_df = next(iter(st.session_state.datos_procesados.values()))
        n_sensores_detectados = primer_df.attrs.get("n_sensores", 12)

        posiciones_sensores = calcular_posiciones_sensores(
            st.session_state.configuracion_inicial['distancia_toma_12'],
            st.session_state.configuracion_inicial['distancia_entre_tomas'],
            n_sensores_detectados,
            st.session_state.configuracion_inicial['orden']
        )

        # Mostrar tabla de posiciones
        with st.expander("Ver posiciones calculadas de sensores"):
            pos_df = pd.DataFrame([
                {
                    'Sensor': sensor,
                    'Posición Y [mm]': pos['y'],
                    'Sensor Físico': pos['sensor_fisico']
                }
                for sensor, pos in posiciones_sensores.items()
            ])
            st.dataframe(pos_df, use_container_width=True)
    else:
        st.warning("⚠️ No hay datos procesados aún. Suba archivos en el Paso 2.")


    # Contenedor para los filtros de visualización
    with st.container(border=True):
        st.markdown("####  Filtros de Visualización")
        
        sub_archivos = st.session_state.sub_archivos_generados.values()
        archivos_opciones = sorted(list(set(sa['archivo_fuente'] for sa in sub_archivos)))
        x_opciones = sorted(list(set(sa['x_traverser'] for sa in sub_archivos)))
        tiempo_opciones = sorted(list(set(sa['tiempo'] for sa in sub_archivos)))

        col1, col2, col3 = st.columns(3)
        archivos_seleccionados = col1.multiselect(
            "Filtrar por Archivo Origen:",
            options=archivos_opciones,
            default=archivos_opciones,
            key="filtro_archivos_origen"
        )
        x_seleccionados = col2.multiselect(
            "Filtrar por Posición X:",
            options=x_opciones,
            default=x_opciones,
            key="filtro_posicion_x"
        )
        tiempos_seleccionados = col3.multiselect(
            "Filtrar por Tiempo (s):",
            options=tiempo_opciones,
            default=tiempo_opciones,
            key="filtro_tiempo_s"
        )

    # Filtrar sub-archivos según filtros seleccionados
    sub_archivos_filtrados = {
        clave: sub_archivo for clave, sub_archivo in st.session_state.sub_archivos_generados.items()
        if sub_archivo['archivo_fuente'] in archivos_seleccionados
        and sub_archivo['x_traverser'] in x_seleccionados
        and sub_archivo['tiempo'] in tiempos_seleccionados
    }

    # Selección de sub-archivos para gráfico concatenado
    st.markdown("### 🎯 Selección de Sub-archivos para Gráfico Concatenado")

    if not sub_archivos_filtrados:
        st.warning("No hay datos que coincidan con los filtros seleccionados.")
    else:
        sub_archivos_seleccionados = {}

        # Generar color único aleatorio para cada sub-archivo
            # Inicializar colores persistentes en la sesión
        if "colores_por_subarchivo" not in st.session_state:
            st.session_state.colores_por_subarchivo = {}

        # Asignar color solo a los sub-archivos que no tengan uno
        for clave in sub_archivos_filtrados.keys():
            if clave not in st.session_state.colores_por_subarchivo:
                st.session_state.colores_por_subarchivo[clave] = "#{:06x}".format(random.randint(0, 0xFFFFFF))

        colores_por_subarchivo = st.session_state.colores_por_subarchivo

        # Mostrar lista de selección con colores
        for i, (clave, sub_archivo) in enumerate(sorted(sub_archivos_filtrados.items())):
            col1, col2 = st.columns([3, 1])
            label = f"{sub_archivo['archivo_fuente']} - T{sub_archivo['tiempo']}s - X{sub_archivo['x_traverser']} - {len(sub_archivo['datos'])} registros"
            
            if col1.checkbox(label, key=f"incluir_{clave}_{i}"):
                sub_archivos_seleccionados[clave] = sub_archivo
            
            with col2:
                color_sub = colores_por_subarchivo[clave]
                st.markdown(
                    f'<div style="background: {color_sub}; height: 20px; width: 60px; border-radius: 3px; margin-top: 8px;"></div>',
                    unsafe_allow_html=True
                )

        # Generar gráfico concatenado si hay selecciones
        if sub_archivos_seleccionados:
            st.markdown("### 📊 Gráfico Concatenado Vertical Modo Betz")

            fig = go.Figure()
            for clave, sub_archivo in sub_archivos_seleccionados.items():
                color = colores_por_subarchivo[clave]
                z_datos, presion_datos = extraer_datos_para_grafico(sub_archivo, st.session_state.configuracion_inicial)
                if z_datos and presion_datos:
                    fig.add_trace(go.Scatter(
                        x=presion_datos,
                        y=z_datos,
                        mode='lines',
                        name=clave,
                        line=dict(color=color, width=2),
                        fill='tozerox',
                        opacity=0.7
                    ))

            fig.update_layout(
                title="Perfil de Presión Concatenado",
                xaxis_title="Presión Total [Pa]",
                yaxis_title="Altura Z [mm]",
                plot_bgcolor='rgba(0,0,0,0)',   # transparente en área del gráfico
                paper_bgcolor='rgba(0,0,0,0)',  # transparente en todo el lienzo
                font=dict(color='white'),       # texto en blanco para que se lea bien
                height=900,
                width=1600
            )
            st.plotly_chart(fig, use_container_width=False)

            total_puntos = len(sub_archivos_seleccionados) * 12
            st.success(f"✅ Gráfico vertical generado con {total_puntos} puntos de datos concatenados")

            # Exportaciones
            st.markdown("### 📤 Exportación")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                html_string = fig.to_html()
                st.download_button(
                    label="📊 Descargar Gráfico (HTML)",
                    data=html_string,
                    file_name=f"grafico_betz_vertical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )

            with col2:
                datos_exportar = []
                for clave, sub_archivo in sub_archivos_seleccionados.items():
                    datos_sub = sub_archivo['datos'].copy()
                    datos_sub['Sub_archivo'] = clave
                    datos_exportar.append(datos_sub)
                
                df_exportar = pd.concat(datos_exportar, ignore_index=True)
                csv_data = df_exportar.to_csv(
                    index=False, sep=';', encoding='utf-8-sig', decimal=','
                )
                st.download_button(
                    label="📋 Descargar Datos (CSV)",
                    data=csv_data,
                    file_name=f"datos_betz_vertical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col3:
                st.metric("Total de puntos", total_puntos)
        else:
            st.error("  No se pudo generar el gráfico. Verifique los datos seleccionados.")

    
        # NUEVA SECCIÓN: RESTA DE  ÁREAS
        # NUEVA SECCIÓN: RESTA DE  ÁREAS
        if st.session_state.sub_archivos_generados:
            st.markdown("---")
            st.markdown("## ➖ Análisis de Diferencias de  áreas")
            st.markdown("Selecciona dos sub-archivos para calcular la diferencia de áreas entre sus perfiles de presión")
            
            # Crear lista de opciones para los selectores
            opciones_subarchivos = sorted(list(st.session_state.sub_archivos_generados.keys()))
            
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                archivo_a = st.selectbox(
                    "📊 Archivo A (minuendo):",
                    opciones_subarchivos,
                    key="archivo_a_resta",
                    help="Seleccione el primer sub-archivo (del cual se restará el segundo)"
                )
            
            with col2:
                st.markdown("<div style='text-align: center; font-size: 2rem; margin-top: 2rem;'>➖</div>", unsafe_allow_html=True)
            
            with col3:
                archivo_b = st.selectbox(
                    "📊 Archivo B (sustraendo):",
                    opciones_subarchivos,
                    index=1 if len(opciones_subarchivos) > 1 else 0,
                    key="archivo_b_resta",
                    help="Seleccione el segundo sub-archivo (que será restado del primero)"
                )
            
            # --- INICIO DE LA ESTRUCTURA CORREGIDA ---

            # PARTE 1: Botón para CALCULAR. Su única misión es generar el gráfico y ponerlo en una "bandeja" temporal.
            if st.button("🔄 Calcular Diferencia de  áreas", type="primary", use_container_width=True):
                if archivo_a and archivo_b and archivo_a != archivo_b:
                    with st.spinner("Calculando diferencia de áreas..."):
                        fig_diferencia, diferencia_area = crear_grafico_diferencia_areas(
                            st.session_state.sub_archivos_generados[archivo_a],
                            st.session_state.sub_archivos_generados[archivo_b],
                            st.session_state.configuracion_inicial
                        )
                        if fig_diferencia:
                            # Guardamos TODO lo necesario en la sesión para mostrarlo después del reinicio de la página.
                            st.session_state.figura_diferencia_temporal = {
                                "fig": fig_diferencia,
                                "nombre": f"Dif. 2D: {archivo_a.split('_')[0]} vs {archivo_b.split('_')[0]}",
                                "area_diferencia": diferencia_area,
                                "archivo_a": archivo_a,
                                "archivo_b": archivo_b
                            }
                        else:
                            st.error("  No se pudo calcular la diferencia.")
                            if 'figura_diferencia_temporal' in st.session_state:
                                del st.session_state.figura_diferencia_temporal
                else:
                    st.warning("⚠️ Seleccione dos sub-archivos diferentes para calcular la diferencia.")

            # PARTE 2: Este bloque está AFUERA del anterior. Revisa si hay algo en la "bandeja" temporal.
            # Si hay algo, lo muestra junto con TODOS sus botones (Guardar, métricas, descarga).
            if 'figura_diferencia_temporal' in st.session_state:
                # Recuperamos los datos de la sesión que guardamos en la PARTE 1
                temp_data = st.session_state.figura_diferencia_temporal
                fig_diferencia = temp_data["fig"]
                nombre_guardado = temp_data["nombre"]
                diferencia_area = temp_data["area_diferencia"]
                archivo_a_calc = temp_data["archivo_a"]
                archivo_b_calc = temp_data["archivo_b"]
                
                # 1. Mostramos el gráfico
                st.plotly_chart(fig_diferencia, use_container_width=False)
                
                # 2. Mostramos el botón de "Guardar". Ahora sí funcionará.
                if st.button("💾 Guardar Diferencia para Visualizar", key="save_diff_2d_for_viz_final"):
                    if 'diferencias_guardadas' not in st.session_state:
                        st.session_state.diferencias_guardadas = {}
                    st.session_state.diferencias_guardadas[nombre_guardado] = fig_diferencia
                    st.success(f"✅ Gráfico '{nombre_guardado}' guardado permanentemente.")
                    # Borramos la figura temporal después de guardarla para limpiar la "bandeja"
                    del st.session_state.figura_diferencia_temporal
                    st.rerun()

                # 3. Mostramos las métricas y el resto de tu código (sin cambios)
                col_m1, col_m2, col_m3 = st.columns(3)
                z_a, p_a = extraer_datos_para_grafico(st.session_state.sub_archivos_generados[archivo_a_calc], st.session_state.configuracion_inicial)
                z_b, p_b = extraer_datos_para_grafico(st.session_state.sub_archivos_generados[archivo_b_calc], st.session_state.configuracion_inicial)
                area_a = calcular_area_bajo_curva(z_a, p_a)
                area_b = calcular_area_bajo_curva(z_b, p_b)
                
                with col_m1:
                    st.metric(f" área {st.session_state.sub_archivos_generados[archivo_a_calc]['archivo_fuente']}", f"{area_a:.2f} Pa·mm")
                with col_m2:
                    st.metric(f" área {st.session_state.sub_archivos_generados[archivo_b_calc]['archivo_fuente']}", f"{area_b:.2f} Pa·mm")
                with col_m3:
                    st.metric("Diferencia de  áreas", f"{diferencia_area:.2f} Pa·mm", delta=f"{diferencia_area:.2f}", delta_color="normal" if diferencia_area >= 0 else "inverse")
                
                if diferencia_area > 0:
                    st.success(f"✅ El área de **{st.session_state.sub_archivos_generados[archivo_a_calc]['archivo_fuente']}** es **{diferencia_area:.2f} Pa·mm** mayor que la de **{st.session_state.sub_archivos_generados[archivo_b_calc]['archivo_fuente']}**")
                elif diferencia_area < 0:
                    st.info(f"ℹ️ El área de **{st.session_state.sub_archivos_generados[archivo_b_calc]['archivo_fuente']}** es **{abs(diferencia_area):.2f} Pa·mm** mayor que la de **{st.session_state.sub_archivos_generados[archivo_a_calc]['archivo_fuente']}**")
                else:
                    st.info("ℹ️ Las áreas son prácticamente iguales")
                
                html_diferencia = fig_diferencia.to_html()
                st.download_button(
                    label="📊 Descargar Gráfico de Diferencia (HTML)",
                    data=html_diferencia,
                    file_name=f"diferencia_areas_{archivo_a_calc}_vs_{archivo_b_calc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )

elif st.session_state.seccion_actual == 'vis_2d_nueva':

    if 'configuracion_2d' not in st.session_state:
        st.session_state.configuracion_2d = {
            'orden': 'asc',
            'sensor_referencia': 'Sensor 12',
            'distancia_toma_12': -120.0,
            'distancia_entre_tomas': 10.0
        }
    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            📈 VISUALIZACIÓN DE ESTELA 2D
            </h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Mapeo de Campos de Presión Interactivo
            </h2>
        </div>
    """, unsafe_allow_html=True)

    # --- PASO 1: CONFIGURACIÓN INICIAL ---
    with st.expander("💾 PASO 1: Configuración de Geometría y Sensores", expanded=True):
        st.markdown("""
        <div class="section-card" style="margin-bottom: 12px;">
            <h3 style="margin-top:0; color:white;">💾 PASO 1: CONFIGURACIÓN INICIAL</h3>
            <p style="color:#bbb; margin-bottom:0;">
                Defina los parámetros físicos del peine de sensores y el sistema de adquisición para el entorno 2D.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_config, col_ref = st.columns([2, 1])
        
        with col_config:
            st.markdown("<div style='padding: 1rem; border: 1px solid #333; border-radius: 8px; background-color: #000;'>", unsafe_allow_html=True)
            orden_sensores = st.selectbox(
                "Orden de lectura de sensores",
                ["asc", "des"],
                format_func=lambda x: "Ascendente (Sensor 1 → 12)" if x == "asc" else "Descendente (Sensor 12 → 1)",
                key="orden_2d"
            )

            sensor_referencia = st.selectbox(
                "Sensor de referencia (Toma 12)",
                [f"Sensor {i}" for i in range(1, 37)],
                index=11, key="sensor_ref_2d"
            )

            c1, c2 = st.columns(2)
            with c1:
                distancia_toma_12 = st.number_input("Distancia Toma 12 [mm]", value=-120.0, step=1.0, format="%.1f", key="dist_12_2d")
            with c2:
                distancia_entre_tomas = st.number_input("Sep. entre tomas [mm]", value=10.00, step=0.01, format="%.2f", key="dist_entre_2d")

            if st.button("💾 CONFIRMAR CONFIGURACIÓN", type="primary", use_container_width=True, key="btn_conf_2d"):
                st.session_state.configuracion_2d = {
                    'orden': orden_sensores,
                    'sensor_referencia': sensor_referencia,
                    'distancia_toma_12': distancia_toma_12,
                    'distancia_entre_tomas': distancia_entre_tomas
                }
                st.success("✅ Configuración 2D guardada.")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with col_ref:
        st.markdown("""
        <div style="background: #111; border: 1px dashed #444; border-radius: 8px; padding: 1rem; text-align: center; height: 100%;">
            <p style="color: #888; font-size: 0.8rem; margin-bottom: 10px;">REFERENCIA TÉCNICA</p>
            <img src="https://raw.githubusercontent.com/Juan-Cruz-de-la-Fuente/Laboratorio/main/Peine.jpg" 
                 style="max-width: 100%; border-radius: 4px; opacity: 0.8;">
        </div>
        """, unsafe_allow_html=True)


    # --- PASO 2: IMPORTACIÓN DE ARCHIVOS ---
    st.markdown("---")
    st.markdown("""
    <div class="section-card" style="margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: white;"> PASO 2: IMPORTACIÓN DE ARCHIVOS CRUDOS</h3>
        <p style="color: #bbb; margin-bottom: 20px;">Cargue uno o múltiples archivos CSV crudos de ensayo para generar los planos espaciales en 2D.</p>
    </div>
    """, unsafe_allow_html=True)

    if 'archivos_2d_cargados' not in st.session_state:
        st.session_state.archivos_2d_cargados = {}

    uploaded_files_2d = st.file_uploader("Arrastre sus archivos CSV de Incertidumbre aquí", type=['csv'], accept_multiple_files=True, key="uploader_2d")
    uploaded_infinito_2d = st.file_uploader("🔗 'Valores en el infinito.txt' (Opcional - datos atmosféricos)", type=['txt', 'csv'], accept_multiple_files=False, key="upl_inf_2d")

    if uploaded_files_2d:
        st.markdown("<br>", unsafe_allow_html=True)
        for file_2d in uploaded_files_2d:
            nombre_archivo = file_2d.name.replace('.csv', '').replace('incertidumbre_', '')
            if nombre_archivo not in st.session_state.archivos_2d_cargados:
                with st.spinner(f"🔨 Procesando archivo {nombre_archivo}..."):
                    datos_procesados = procesar_promedios(file_2d, st.session_state.configuracion_2d['orden'], uploaded_infinito_2d)
                    if datos_procesados is not None:
                        st.session_state.archivos_2d_cargados[nombre_archivo] = datos_procesados
                        st.success(f"✅ Archivo extraído correctamente: {nombre_archivo}")
                        
    _archivos_mem_2d = st.session_state.archivos_2d_cargados
    try:
        _archivos_drv_2d = auth.get_user_files_2d(st.session_state.username)
    except:
        _archivos_drv_2d = []

    # Mostrar resumen interactivo de archivos con Progress list (como en 3D)
    if _archivos_mem_2d:
        st.markdown("### 📋 Archivos en Memoria")
        cols = st.columns(3)
        for idx, (nombre, datos) in enumerate(_archivos_mem_2d.items()):
            with cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(f"**📦 {nombre}**")
                    st.caption(f"{len(datos)} Puntos medidos • {len(datos['Tiempo_s'].unique())} Tiempos discretos")
                    st.progress(100)

    if _archivos_mem_2d or _archivos_drv_2d:
        # --- PASO 3: VISUALIZADOR INTERACTIVO ---
        st.markdown("---")
        st.markdown("""
        <div class="section-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: white;">📈 PASO 3: VISUALIZADOR 2D INTERACTIVO</h3>
            <p style="color: #bbb; margin-bottom: 20px;">Explore el plano de presiones seleccionando archivo y escala, interactuando nativamente con las herramientas del trazado (regla configurable).</p>
        </div>
        """, unsafe_allow_html=True)
        
        c_cfg, c_plot = st.columns([1, 2.5])
        with c_cfg:
            st.markdown("### 1. Parámetros de Escala")
            cuerda_mm = st.number_input("Cuerda Referencia [mm]", value=300.0, step=1.0)
            
            st.markdown("### 2. Selección de Plano")
            fuente_plot_2d = st.radio("Fuente de matriz:", ["Archivos Crudos (Memoria)", "Matriz Guardada (Drive)"])
            
            ejecutar_viz_2d = False
            if fuente_plot_2d == "Archivos Crudos (Memoria)":
                if not _archivos_mem_2d:
                    st.warning("⚠️ No hay archivos procesados en memoria.")
                else:
                    archivo_sel = st.selectbox("Seleccionar Archivo (X):", list(_archivos_mem_2d.keys()))
                    df_selected = _archivos_mem_2d[archivo_sel]
                    tiempos = df_selected['Tiempo_s'].dropna().unique()
                    tiempo_sel = st.selectbox("Seleccionar Tiempo:", sorted(tiempos))
                    ejecutar_viz_2d = True
            else:
                if not _archivos_drv_2d:
                    st.warning("⚠️ No hay matrices 2D en tu Drive.")
                else:
                    dict_drv_2d = {f"{a[1]} [{a[2][:10] if a[2] else ''}]": a for a in _archivos_drv_2d}
                    archivo_drv_sel = st.selectbox("Seleccionar Matriz Drive:", list(dict_drv_2d.keys()))
                    ejecutar_viz_2d = True

            st.markdown("### 3. Visualización")
            opciones_var_2d = ["Presión Total [Actual]", "rho_inf", "V_inf", "P_inf"]
            var_2d_sel = st.selectbox("📊 Variable a visualizar:", opciones_var_2d, key="var_2d_sel_ui")
            
            plot_type = st.selectbox("Render de Pixeles:", ["Contour Suavizado", "Mapa de Calor (Celdas)"])

            st.markdown("---")
            st.markdown("###  Medición de Trazos")
            st.info("La Regla ↘️ del gráfico mide nativamente en [mm]. Ingresa aquí el trazo medido para convertir a [c]:")
            long_leida = st.number_input("Longitud Leída [mm]:", value=0.0, step=1.0)
            if long_leida > 0:
                st.success(f"**Longitud Equiv:** {(long_leida / cuerda_mm):.3f} c")

        with c_plot:
            if ejecutar_viz_2d:
                with st.spinner("Ensamblando proyección de contornos 2D..."):
                    if fuente_plot_2d == "Archivos Crudos (Memoria)":
                        df_run = df_selected[df_selected['Tiempo_s'] == tiempo_sel].copy()
                        
                        results_2d = []
                        # Ensamblar la Matriz 2D extrayendo las posiciones absolutas
                        for _, row in df_run.iterrows():
                            y_trav = row.get('Pos_Y_Traverser')
                            z_base = row.get('Pos_Z_Base')
                            for col in df_run.columns:
                                num_sensor = obtener_numero_sensor_desde_columna(col)
                                if num_sensor is not None:
                                    val_presion = row[col]
                                    if pd.isna(val_presion): continue
                                    z_real = calcular_altura_absoluta_z(
                                        num_sensor, z_base, 
                                        st.session_state.configuracion_2d.get('distancia_toma_12', -120),
                                        st.session_state.configuracion_2d.get('distancia_entre_tomas', 10.0),
                                        12, st.session_state.configuracion_2d.get('orden', 'asc')
                                    )
                                    results_2d.append({
                                        'Y': y_trav, 
                                        'Z': z_real, 
                                        'Presion': val_presion,
                                        'rho_inf': row.get('rho_inf', 1.225),
                                        'V_inf': row.get('V_inf', 0.0),
                                        'P_inf': row.get('P_inf', 101325.0),
                                        'T_inf': row.get('T_inf', 15.0),
                                        'Timestamp': row.get('Timestamp')
                                    })
                                    
                        df_matriz = pd.DataFrame(results_2d)
                    else:
                        s_data = dict_drv_2d[archivo_drv_sel]
                        csv_bytes = auth.download_file_2d(s_data[0])
                        csv_str = csv_bytes.decode('utf-8-sig') if csv_bytes else ""
                        import io
                        df_matriz = pd.read_csv(io.StringIO(csv_str), sep=';', decimal=',')
                        if 'Y' not in df_matriz.columns or 'Z' not in df_matriz.columns or 'Presion' not in df_matriz.columns:
                            df_matriz = pd.read_csv(io.StringIO(csv_str), sep=',', decimal='.')
                if not df_matriz.empty:
                    df_matriz['Presion'] = calcular_variable_atmosferica(df_matriz, var_2d_sel)
                
                if df_matriz.empty:
                    st.error("  No se pudieron extraer datos espaciales (Y, Z, P). Comprueba tu archivo físico.")
                else:
                    y_plot = df_matriz['Y'].values
                    z_plot = df_matriz['Z'].values
                    val_plot = df_matriz['Presion'].values
                    eje_label = "mm"
                    z_title = "P [Pa]"
                    if var_2d_sel == "rho_inf":
                        z_title = "Densidad [kg/m³]"
                        hover_text = "Densidad: %{z:.2f} kg/m³"
                    elif var_2d_sel == "V_∞":
                        z_title = "V [m/s]"
                        hover_text = "Velocidad: %{z:.2f} m/s"
                    else:
                        hover_text = "Presión: %{z:.2f} Pa"
                        
                    cs_name = "Jet"

                    # Create Grid Interpolation cubic 150x150 res
                    y_grid_vals = np.linspace(y_plot.min(), y_plot.max(), 150)
                    z_grid_vals = np.linspace(z_plot.min(), z_plot.max(), 150)
                    Y_grid, Z_grid = np.meshgrid(y_grid_vals, z_grid_vals)
                    
                    try:
                        V_grid = griddata((y_plot, z_plot), val_plot, (Y_grid, Z_grid), method='cubic')
                        fig = go.Figure()
                        
                        if plot_type == "Contour Suavizado":
                            dtick_val = None
                            if "P_∞" in var_2d_sel: 
                                dtick_val = 1
                            elif "V_inf" in var_2d_sel:
                                dtick_val = 0.1
                            elif "rho_inf" in var_2d_sel:
                                dtick_val = 0.05
                            
                            c_args = dict(showlines=False)
                            if dtick_val:
                                c_args['start'] = np.nanmin(V_grid)
                                c_args['end'] = np.nanmax(V_grid)
                                c_args['size'] = dtick_val

                            fig.add_trace(go.Contour(
                                x=y_grid_vals, y=z_grid_vals, z=V_grid,
                                colorscale=cs_name, colorbar=dict(title=z_title),
                                contours=c_args,
                                hovertemplate=f"Y: %{{x:.2f}} mm<br>Z: %{{y:.2f}} mm<br>{hover_text}<extra></extra>"
                            ))
                        else:
                            fig.add_trace(go.Heatmap(
                                x=y_grid_vals, y=z_grid_vals, z=V_grid,
                                colorscale=cs_name, colorbar=dict(title=z_title),
                                hovertemplate=f"Y: %{{x:.2f}} mm<br>Z: %{{y:.2f}} mm<br>{hover_text}<extra></extra>"
                            ))

                        fig.update_layout(
                            title=dict(text=f"Proyección Espacial 2D: Presión [Pa]", font=dict(size=20, color="white")),
                            xaxis_title=dict(text=f"Envergadura (Y) [{eje_label}]", font=dict(color="white")),
                            yaxis_title=dict(text=f"Altura (Z) [{eje_label}]", font=dict(color="white")),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="white"),
                            dragmode="drawline",
                            newshape=dict(line_color="magenta", line_width=3, opacity=1.0),
                            margin=dict(l=60, r=60, t=60, b=60),
                            height=800
                        )
                        # Ancla física de ejes. Garantiza que la matriz mida proporcionalmente 1:1 en la pantalla
                        fig.update_xaxes(scaleanchor="y", scaleratio=1, showgrid=True, gridcolor="rgba(255,255,255,0.1)")
                        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")

                        plot_key = f"plot_2d_{st.session_state.get('plot_2d_key', 0)}"
                        st.plotly_chart(fig, use_container_width=True, config={
                            'modeBarButtonsToAdd': ['drawline', 'drawcircle', 'eraseshape'],
                            'displaylogo': False, 'scrollZoom': True
                        }, key=plot_key)
                        col_info, col_btn = st.columns([0.8, 0.2])
                        with col_info:
                            st.info("💡 **Gráfico Físico Proporcional:** Para borrar un dibujo puedes usar el botón la Goma (Erase active shape) en el menú del gráfico.")
                        with col_btn:
                            if st.button("🧹 Limpiar Dibujos", key="btn_clear_trazos", use_container_width=True):
                                st.session_state.plot_2d_key = st.session_state.get('plot_2d_key', 0) + 1
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error trazando proyecciones cúbicas: {e}")

        # --- GUARDAR MATRIZ 2D EN DRIVE ---
        st.markdown("---")
        st.markdown("""
        <div class="section-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: white;"> ? PASO 4: GUARDAR MATRIZ EN DRIVE (2D)</h3>
            <p style="color: #bbb; margin-bottom: 0;">Guarda la matriz de presiones del plano seleccionado en <b>ENSAYO DE ESTELA / 2D</b>.</p>
        </div>
        """, unsafe_allow_html=True)
        if not df_matriz.empty:
            c_2d_a, c_2d_b = st.columns(2)
            aoa_2d = c_2d_a.number_input(" Ángulo de Ataque [°]:", value=0.0, step=0.5, format="%.1f", key="aoa_2d_paso4")
            x_2d = c_2d_b.number_input(" Posición X (Estación) [mm]:", value=0.0, step=10.0, key="x_2d_paso4")

            _aoa_str_2d = str(int(aoa_2d)) if aoa_2d == int(aoa_2d) else f"{aoa_2d:.1f}"
            _aoa_str_2d = _aoa_str_2d.replace("-", "neg")
            nombre_auto_2d = f"2D-X{int(x_2d)}-OAO{_aoa_str_2d}-T{int(tiempo_sel)}s.csv"

            c_2d_chk, c_2d_nom = st.columns([0.15, 0.85])
            usar_custom_2d = c_2d_chk.checkbox("Nombre libre", key="custom_nom_2d")
            if usar_custom_2d:
                nombre_csv_2d = c_2d_nom.text_input("Nombre personalizado:", placeholder=nombre_auto_2d, key="nombre_2d_custom")
                if not nombre_csv_2d:
                    nombre_csv_2d = nombre_auto_2d
            else:
                nombre_csv_2d = nombre_auto_2d
                c_2d_nom.code(nombre_csv_2d)

            csv_bytes_2d = df_matriz.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')
            col_2d_dl, col_2d_drive = st.columns(2)
            with col_2d_dl:
                st.download_button("📥 Descargar Matriz 2D", csv_bytes_2d,
                                   file_name=nombre_csv_2d, mime="text/csv", key="dl_2d_matriz")
            with col_2d_drive:
                if st.button(" Guardar en Drive (2D)", key="save_2d_drive", use_container_width=True):
                    if auth.save_csv_2d(st.session_state.username, nombre_csv_2d, csv_bytes_2d):
                        st.success(f"✅ Subido a Drive → ENSAYO DE ESTELA/2D/{nombre_csv_2d}")
                    else:
                        st.error("Error al subir a Drive")

    # --- PASO 5: ANÁLISIS DE PAR ?METROS ATMOSFÉRICOS (INFINITO) ---
    if _archivos_mem_2d:
        st.markdown("---")
        st.markdown("""
        <div class="section-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: white;">📊 PASO 5: ANÁLISIS DE PARÁMETROS ATMOSFÉRICOS (INFINITO)</h3>
            <p style="color: #bbb; margin-bottom: 0;">Analice la estabilidad de las variables en el infinito durante todo el ensayo.</p>
        </div>
        """, unsafe_allow_html=True)

        # Recolectar datos de infinito de todos los archivos cargados
        inf_data_list = []
        for nombre, df in _archivos_mem_2d.items():
            if 'Timestamp' in df.columns:
                inf_cols = ['Timestamp', 'rho_inf', 'V_inf', 'P_inf', 'T_inf']
                # Filtrar columnas existentes (fallback para archivos viejos sin T_inf)
                cols_to_use = [c for c in inf_cols if c in df.columns]
                tmp_df = df[cols_to_use].copy()
                tmp_df['Archivo_Origen'] = nombre
                inf_data_list.append(tmp_df)
        
        if not inf_data_list:
            st.warning("⚠️ No se detectaron metadatos temporales en los archivos actuales. Por favor, selecciona los archivos y vuelve a cargarlos para activar este análisis.")
        else:
            df_inf_global = pd.concat(inf_data_list).drop_duplicates()
            # Formato real del timestamp: DDMMYYHHMMSS (ej: 260424144919 = 26/04/2024 14:49:19)
            df_inf_global['dt_val'] = pd.to_datetime(df_inf_global['Timestamp'].astype(str).str.split(',').str[0].str.strip(), format='%d%m%y%H%M%S', errors='coerce')
            # Fallback: intentar formato alternativo
            mask_fail = df_inf_global['dt_val'].isna()
            if mask_fail.any():
                df_inf_global.loc[mask_fail, 'dt_val'] = pd.to_datetime(
                    df_inf_global.loc[mask_fail, 'Timestamp'].astype(str).str.split(',').str[0].str.strip(),
                    format='%y%m%d%H%M%S', errors='coerce'
                )
            df_inf_global = df_inf_global.dropna(subset=['dt_val']).sort_values('dt_val')

            # Normalizar tiempo a t=0 usando segundos reales
            min_dt = df_inf_global['dt_val'].min()
            df_inf_global['Tiempo_Relativo_s'] = (df_inf_global['dt_val'] - min_dt).dt.total_seconds()
            
            c_inf_1, c_inf_2 = st.columns([1, 2])
            with c_inf_1:
                options_inf = [" ρ_∞", "V_∞", "P_∞", "T_∞"]
                # Filtrar opciones si el archivo es viejo y no tiene T_inf
                if 'T_inf' not in df_inf_global.columns:
                    st.caption("Nota: Temperatura no disponible en estos datos.")
                    options_inf.remove("T_∞")
                
                var_inf_plot = st.selectbox("Variable Atmosférica:", options_inf, key="var_inf_plot")
                tipo_inf_plot = st.radio("Tipo de Análisis:", ["Evolución Temporal", "Distribución Normal"], key="tipo_inf_plot")
                
                # Map variable name
                inf_col_map = {"rho_inf": "rho_inf", "V_inf": "V_inf", "P_inf": "P_inf", "T_inf": "T_inf"}
                col_plot = inf_col_map[var_inf_plot]
                unit_plot = {"rho_inf": "[kg/m³]", "V_inf": "[m/s]", "P_inf": "[Pa]", "T_inf": "[°C]"}[var_inf_plot]

            with c_inf_2:
                if tipo_inf_plot == "Evolución Temporal":
                    fig_inf = px.line(df_inf_global, x='Tiempo_Relativo_s', y=col_plot, 
                                     title=f"Evolución de {var_inf_plot} vs Tiempo",
                                     labels={'Tiempo_Relativo_s': 'Tiempo [s]', col_plot: f"{var_inf_plot} {unit_plot}"},
                                     markers=True)
                    fig_inf.update_traces(line_color='#00d1ff')
                else:
                    fig_inf = px.histogram(df_inf_global, x=col_plot, 
                                          title=f"Distribución de {var_inf_plot}",
                                          labels={col_plot: f"{var_inf_plot} {unit_plot}"},
                                          marginal="box",
                                          opacity=0.7)
                    fig_inf.update_traces(marker_color='#00d1ff')

                fig_inf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig_inf, use_container_width=True)


elif st.session_state.seccion_actual == 'analisis_vortices':

    # ── MOTOR DE DETECCIÓN AUTOM ?TICA ────────────────────────────────────────
    def _detectar_vortices(y_data, z_data, p_data):
        """
        Detección automática de vórtices basada en física.
        Discriminadores clave:
          1. Circularidad radial  → vórtice = mínimo rodeado en TODAS las dir.
          2. Aspecto compacto     → vórtice ≠ banda elongada (estela/soporte)
          3. Profundidad mínima   → evita ruido
          4. Tamaño razonable     → evita ruido pequeño y la estela completa
          5. Simetría en Y        → los vórtices de punta de ala son simétricos
        Sin parámetros de usuario.
        """
        from scipy.interpolate import griddata as _gd
        from scipy.ndimage import gaussian_filter, minimum_filter

        GRID = 200
        y_lin = np.linspace(y_data.min(), y_data.max(), GRID)
        z_lin = np.linspace(z_data.min(), z_data.max(), GRID)
        Yg, Zg = np.meshgrid(y_lin, z_lin)

        Pg = _gd((y_data, z_data), p_data, (Yg, Zg), method='linear')
        nan_m = np.isnan(Pg)
        if nan_m.any():
            Pg[nan_m] = _gd((y_data, z_data), p_data, (Yg[nan_m], Zg[nan_m]), method='nearest')

        Ps = gaussian_filter(Pg, sigma=2.0)

        p_bg    = float(np.nanpercentile(Ps, 97))
        p_floor = float(np.nanpercentile(Ps, 3))
        p_range = p_bg - p_floor
        if p_range < 1e-6:
            return [], Pg, y_lin, z_lin

        sz = max(7, GRID // 20)
        lm = (minimum_filter(Ps, size=sz) == Ps) & (Ps < p_bg - p_range * 0.05)
        cands = np.argwhere(lm)
        if len(cands) == 0:
            return [], Pg, y_lin, z_lin

        pv = Ps[cands[:, 0], cands[:, 1]]
        cands = cands[np.argsort(pv)]   # orden por profundidad

        dy = y_lin[1] - y_lin[0]
        dz = z_lin[1] - z_lin[0]
        y_min, y_max = y_lin[0], y_lin[-1]
        y_mid = (y_min + y_max) / 2.0
        y_span = y_max - y_min

        N_ANG   = 24
        MAX_R   = int(GRID * 0.35)
        REC_THR = 0.60          # recuperación del 60 % de la profundidad
        angles  = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False)
        sins    = np.sin(angles)
        coss    = np.cos(angles)

        scored = []
        for ri, ci in cands:
            y_val = float(y_lin[ci])
            z_val = float(z_lin[ri])
            
            # 1. Ignorar el centro (soporte/fuselaje)
            # El soporte suele estar en el medio. Bloqueamos el 12% central.
            if abs(y_val - y_mid) < y_span * 0.06:
                continue

            p_core = Ps[ri, ci]
            depth  = p_bg - p_core
            if depth < p_range * 0.04:
                continue

            radii     = []
            n_boundary = 0
            for k in range(N_ANG):
                found = None
                for r in range(2, MAX_R):
                    ri2 = int(round(ri + r * sins[k]))
                    ci2 = int(round(ci + r * coss[k]))
                    if not (0 <= ri2 < GRID and 0 <= ci2 < GRID):
                        n_boundary += 1
                        break
                    # Criterio de recuperación local
                    if Ps[ri2, ci2] >= p_core + depth * REC_THR:
                        found = r
                        break
                if found is not None:
                    radii.append(found)

            # 2. Discriminadores morfológicos estrictos
            if n_boundary > N_ANG // 3: continue  # Toca mucho borde
            if len(radii) < N_ANG * 0.70: continue # No cierra (estela abierta)

            r_arr  = np.array(radii, dtype=float)
            r_mean = r_arr.mean()
            r_std  = r_arr.std()
            cov    = r_std / (r_mean + 1e-6)
            
            # Circularidad: castigo fuerte a formas elongadas (como la estela del soporte)
            compactness = max(0.0, 1.0 - cov * 2.5)
            if compactness < 0.35:
                continue

            # Relación de aspecto (Z vs Y) para descartar bandas verticales
            semi_y_est = float(np.mean(r_arr * np.abs(np.cos(angles[:len(radii)])))) * dy
            semi_z_est = float(np.mean(r_arr * np.abs(np.sin(angles[:len(radii)])))) * dz
            aspect_ratio = semi_z_est / (semi_y_est + 1e-6)
            
            if aspect_ratio > 2.5 or aspect_ratio < 0.4:
                continue # Demasiado estirado (soporte o estela laminar)

            r_real_mm = r_mean * max(dy, dz)
            dom = max(y_span, z_lin[-1] - z_lin[0])
            r_frac = r_real_mm / dom
            if r_frac < 0.005 or r_frac > 0.40:
                continue

            # Score final: prioriza profundidad, circularidad y distancia al centro
            dist_centro_norm = abs(y_val - y_mid) / (y_span / 2)
            score = (compactness * 0.4) + (min(depth / p_range * 2, 1.0) * 0.4) + (dist_centro_norm * 0.2)

            scored.append(dict(
                ri=ri, ci=ci,
                y=y_val, z=z_val,
                p_core=p_core, depth=depth,
                compactness=compactness, score=score,
                semi_y=max(semi_y_est, dy), semi_z=max(semi_z_est, dz),
                area=np.pi * semi_y_est * semi_z_est,
            ))

        if not scored:
            return [], Pg, y_lin, z_lin

        # 3. NMS y Simetría
        scored.sort(key=lambda x: -x['score'])
        min_sep = GRID * 0.08
        selected = []
        for c in scored:
            if not any(np.hypot(c['ri'] - s['ri'], c['ci'] - s['ci']) < min_sep for s in selected):
                selected.append(c)
        
        # Limitar a los 6 mejores candidatos
        selected = selected[:6]


        # Emparejamiento simétrico (bonus de puntuación)
        sym_tol_y = (y_lin[-1] - y_lin[0]) * 0.09
        sym_tol_z = (z_lin[-1] - z_lin[0]) * 0.09
        paired = set()
        for i, a in enumerate(selected):
            if i in paired:
                continue
            y_mirror = 2 * y_mid - a['y']
            for j, b in enumerate(selected):
                if j <= i or j in paired:
                    continue
                if abs(b['y'] - y_mirror) < sym_tol_y and abs(b['z'] - a['z']) < sym_tol_z:
                    paired.add(i); paired.add(j)
                    selected[i]['has_pair'] = True
                    selected[j]['has_pair'] = True
                    break

        # Construir polígonos elípticos para visualización
        vortices = []
        for v in selected:
            th = np.linspace(0, 2 * np.pi, 120)
            vy = (v['y'] + v['semi_y'] * np.cos(th)).tolist()
            vz = (v['z'] + v['semi_z'] * np.sin(th)).tolist()
            vortices.append({
                'id'         : f"V{len(vortices)+1}",
                'y'          : v['y'],
                'z'          : v['z'],
                'p_min'      : v['p_core'],
                'depth'      : v['depth'],
                'semi_y'     : round(v['semi_y'], 2),
                'semi_z'     : round(v['semi_z'], 2),
                'area'       : v['area'],
                'compactness': round(v['compactness'], 3),
                'score'      : round(v['score'], 3),
                'has_pair'   : v.get('has_pair', False),
                'poly_y'     : vy,
                'poly_z'     : vz,
            })

        return vortices, Pg, y_lin, z_lin
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🌀 ANÁLISIS DE VÓRTICES
            </h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Detección Automática Basada en Física · Sin Configuración
            </h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0d1f35;border:1px solid #1e4060;border-radius:8px;padding:12px;margin-bottom:16px;">
    <p style="color:#93c5fd;margin:0;font-size:0.88rem;">
    🧠 <b>Motor automático:</b> detecta vórtices por su forma circular, aislamiento radial y simetría.
    Rechaza automáticamente la estela (elongada), el soporte (central) y el ruido (demasiado pequeño/grande).
    </p></div>
    """, unsafe_allow_html=True)

    fuente_vortices = st.radio(
        "Fuente de datos:",
        [" Subir CSV (Y, Z, Presion)", "🧠 Desde Archivos 2D en Memoria", " Cargar desde Drive"],
        horizontal=True, key="fuente_vortices_v2"
    )

    df_matriz = pd.DataFrame()

    if fuente_vortices == " Subir CSV (Y, Z, Presion)":
        up_csv = st.file_uploader("CSV con columnas Y, Z, Presion (sep=; dec=,)", type=['csv'], key="up_csv_vortex")
        if up_csv:
            try:
                df_matriz = pd.read_csv(up_csv, sep=';', decimal=',')
                if not {'Y','Z','Presion'}.issubset(df_matriz.columns):
                    df_matriz = pd.read_csv(up_csv, sep=',', decimal='.')
                if not {'Y','Z','Presion'}.issubset(df_matriz.columns):
                    st.error("  El CSV debe tener columnas: Y, Z, Presion")
                    df_matriz = pd.DataFrame()
                else:
                    st.success(f"✅ {len(df_matriz)} puntos cargados")
            except Exception as _e:
                st.error(f"Error leyendo CSV: {_e}")

    elif fuente_vortices == "🧠 Desde Archivos 2D en Memoria":
        if 'archivos_2d_cargados' not in st.session_state or not st.session_state.archivos_2d_cargados:
            st.warning("⚠️ No hay matrices en memoria. Cargá primero desde Vis. Estela 2D.")
        else:
            archivo_sel = st.selectbox("Archivo:", list(st.session_state.archivos_2d_cargados.keys()), key="sel_arch_vort")
            df_selected = st.session_state.archivos_2d_cargados[archivo_sel]
            tiempos = sorted(df_selected['Tiempo_s'].dropna().unique())
            tiempo_sel = st.selectbox("Tiempo [s]:", tiempos, key="sel_tiempo_vort")
            df_run = df_selected[df_selected['Tiempo_s'] == tiempo_sel].copy()
            results_2d = []
            for _, row in df_run.iterrows():
                y_trav = row.get('Pos_Y_Traverser')
                z_base = row.get('Pos_Z_Base')
                for col in df_run.columns:
                    num_sensor = obtener_numero_sensor_desde_columna(col)
                    if num_sensor is not None:
                        val_p = row[col]
                        if pd.isna(val_p): continue
                        z_real = calcular_altura_absoluta_z(
                            num_sensor, z_base,
                            st.session_state.configuracion_2d.get('distancia_toma_12', -120),
                            st.session_state.configuracion_2d.get('distancia_entre_tomas', 10.0),
                            12, st.session_state.configuracion_2d.get('orden', 'asc')
                        )
                        results_2d.append({'Y': y_trav, 'Z': z_real, 'Presion': val_p})
            df_matriz = pd.DataFrame(results_2d)
            if not df_matriz.empty:
                st.success(f"✅ {len(df_matriz)} puntos ensamblados")

    else:  # Drive
        try:
            archivos_2d_drive = auth.get_user_files_2d(st.session_state.username)
        except Exception:
            archivos_2d_drive = []
        if not archivos_2d_drive:
            st.info("No hay matrices 2D en Drive.")
        else:
            dict_drv = {f"{a[1]} [{a[2][:10] if a[2] else ''}]": a for a in archivos_2d_drive}
            sel_drv = st.selectbox("Seleccionar:", list(dict_drv.keys()), key="sel_drv_vort")
            if sel_drv:
                raw = auth.download_file_2d(dict_drv[sel_drv][0])
                if raw:
                    import io as _io
                    df_matriz = pd.read_csv(_io.BytesIO(raw), sep=';', decimal=',')
                    if not {'Y','Z','Presion'}.issubset(df_matriz.columns):
                        df_matriz = pd.read_csv(_io.BytesIO(raw), sep=',', decimal='.')
                    if {'Y','Z','Presion'}.issubset(df_matriz.columns):
                        st.success(f"✅ {len(df_matriz)} puntos desde Drive")
                    else:
                        st.error("  Columnas Y, Z, Presion no encontradas")
                        df_matriz = pd.DataFrame()

    # ── EJECUCIÓN AUTOM ?TICA ────────────────────────────────────────────────
    if not df_matriz.empty and {'Y','Z','Presion'}.issubset(df_matriz.columns):

        df_clean = df_matriz[['Y','Z','Presion']].dropna()
        if len(df_clean) < 10:
            st.error("Insuficientes puntos válidos para la detección.")
        else:
            with st.spinner(" Detectando vórtices automáticamente..."):
                try:
                    vortices, P_grid, y_lin, z_lin = _detectar_vortices(
                        df_clean['Y'].values,
                        df_clean['Z'].values,
                        df_clean['Presion'].values
                    )
                except Exception as _ex:
                    st.error(f"Error en detección: {_ex}")
                    vortices, P_grid, y_lin, z_lin = [], None, None, None

            if P_grid is None:
                st.warning("No se pudo interpolar el campo de presión.")
            else:
                p_bg = float(np.nanpercentile(P_grid, 97))
                y_plot_raw = df_clean['Y'].values
                y_mid_sym  = (y_plot_raw.min() + y_plot_raw.max()) / 2.0

                # ── GRÁFICO PRINCIPAL ───────────────────────────────────────
                n_det = len(vortices)
                if n_det == 0:
                    st.warning("⚠️ No se detectaron vórtices. El campo de presión no presenta mínimos circulares aislados.")
                else:
                    st.success(f"✅ {n_det} vórtice{'s' if n_det>1 else ''} detectado{'s' if n_det>1 else ''}")

                st.markdown("### 📊 Campo de Presión y Vórtices Detectados")
                fig = go.Figure()

                # Fondo: campo de presión
                fig.add_trace(go.Contour(
                    x=y_lin, y=z_lin, z=P_grid,
                    colorscale='Jet',
                    colorbar=dict(title="P [Pa]"),
                    contours=dict(showlines=True, coloring='heatmap'),
                    hovertemplate="Y: %{x:.1f}<br>Z: %{y:.1f}<br>P: %{z:.2f}<extra></extra>"
                ))

                colors = ['#ffffff','#ffff00','#00ff88','#ff6600','#ff00ff','#00cfff','#ff4444','#44ff44']
                for k, v in enumerate(vortices):
                    col = colors[k % len(colors)]
                    label_extra = " 🔗par" if v['has_pair'] else ""
                    fig.add_trace(go.Scatter(
                        x=v['poly_y'], y=v['poly_z'],
                        mode='lines', fill='toself',
                        fillcolor='rgba(255,255,255,0.10)',
                        line=dict(color=col, width=3),
                        name=f"{v['id']}{label_extra}",
                        hovertemplate=(
                            f"<b>{v['id']}</b><br>"
                            f"Centro: ({v['y']:.1f}, {v['z']:.1f}) mm<br>"
                            f"ΔP: {v['depth']:.2f} Pa<br>"
                            f"Circularidad: {v['compactness']:.2f}<extra></extra>"
                        )
                    ))
                    fig.add_trace(go.Scatter(
                        x=[v['y']], y=[v['z']],
                        mode='markers+text',
                        marker=dict(symbol='x', size=14, color=col, line=dict(width=3)),
                        text=[v['id']], textposition='top center',
                        textfont=dict(color=col, size=13, family='Orbitron'),
                        showlegend=False,
                        hovertemplate=f"Núcleo {v['id']}<br>P={v['p_min']:.2f} Pa<extra></extra>"
                    ))

                # Línea de simetría
                fig.add_vline(
                    x=y_mid_sym, line=dict(color='cyan', width=1.5, dash='dash'),
                    annotation_text=f"Eje Y={y_mid_sym:.0f}mm", annotation_font_color='cyan'
                )

                fig.update_layout(
                    xaxis_title="Y [mm]", yaxis_title="Z [mm]",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"), height=750,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='#333')
                )
                fig.update_xaxes(scaleanchor="y", scaleratio=1)
                st.plotly_chart(fig, use_container_width=True)

                # ── TABLA DE RESULTADOS ─────────────────────────────────────
                if vortices:
                    st.markdown("### 📋 Vórtices Detectados")
                    df_v = pd.DataFrame([{
                        "ID"              : v['id'],
                        "Centro Y [mm]"   : round(v['y'], 2),
                        "Centro Z [mm]"   : round(v['z'], 2),
                        "P núcleo [Pa]"   : round(v['p_min'], 2),
                        "ΔP → P∞ [Pa]"   : round(v['depth'], 2),
                        "Semi-eje Y [mm]" : v['semi_y'],
                        "Semi-eje Z [mm]" : v['semi_z'],
                        "área [mm²]"      : round(v['area'], 2),
                        "R equiv. [mm]"   : round(np.sqrt(v['area'] / np.pi), 2),
                        "Circularidad"    : v['compactness'],
                        "Par simétrico"   : "✅" if v['has_pair'] else "—",
                    } for v in vortices])
                    st.dataframe(df_v, use_container_width=True, hide_index=True)

                    # ── CIRCULACIÓN ─────────────────────────────────────────
                    st.markdown("### 🌀 Estimación de Circulación (Γ)")
                    rho_v = 1.225
                    V_v   = 30.0
                    if 'df_inf_global' in dir() or 'df_inf_global' in locals():
                        try:
                            rho_v = float(df_inf_global['rho_inf'].mean())
                            V_v   = float(df_inf_global['V_inf'].mean())
                        except Exception:
                            pass
                    st.caption(f" rho_inf = {rho_v:.3f} kg/m³  |  Vinf = {V_v:.2f} m/s")
                    df_gamma = pd.DataFrame([{
                        "ID"     : v['id'],
                        "Γ [m²/s]": round(
                            np.sqrt(2 * abs(v['depth']) / (rho_v + 1e-9))
                            * (np.sqrt(v['area'] / np.pi) / 1000), 4)
                    } for v in vortices])
                    st.table(df_gamma)

                    # ── GEOMETR ?A INTER-CENTROS ─────────────────────────────
                    if len(vortices) > 1:
                        st.markdown("####  Distancias entre Núcleos")
                        pairs_dist = []
                        for i in range(len(vortices)):
                            for j in range(i+1, len(vortices)):
                                dy_ = vortices[i]['y'] - vortices[j]['y']
                                dz_ = vortices[i]['z'] - vortices[j]['z']
                                pairs_dist.append({
                                    "Enlace": f"{vortices[i]['id']} ↔ {vortices[j]['id']}",
                                    "ΔY [mm]": round(abs(dy_), 2),
                                    "ΔZ [mm]": round(abs(dz_), 2),
                                    "Dist. [mm]": round(np.hypot(dy_, dz_), 2)
                                })
                        st.table(pd.DataFrame(pairs_dist))

                    # ── SIMETR ?A Y GUIÑADA ──────────────────────────────────
                    st.markdown("---")
                    st.markdown("### ⚖️ Análisis de Simetría — Estimación de Guiñada (beta)")
                    y_min_d, y_max_d = y_plot_raw.min(), y_plot_raw.max()
                    st.markdown(f"""
                    <div style="background:#0d1f35;border:1px solid #1e4060;border-radius:8px;padding:12px;margin-bottom:12px;">
                    <p style="color:#93c5fd;font-size:0.85rem;margin:0;line-height:1.8;">
                    <b>Eje de simetría Y:</b> {y_mid_sym:.1f} mm
                    (rango [{y_min_d:.1f} → {y_max_d:.1f}] mm)<br>
                    Si área izquierda ≠ área derecha → posible guiñada beta ≠ 0.
                    </p></div>
                    """, unsafe_allow_html=True)

                    area_izq = sum(v['area'] for v in vortices if v['y'] < y_mid_sym)
                    area_der = sum(v['area'] for v in vortices if v['y'] >= y_mid_sym)

                    area_desg = [{
                        "Vórtice"       : v['id'],
                        "Centro Y [mm]" : round(v['y'], 2),
                        "Centro Z [mm]" : round(v['z'], 2),
                        "área [mm²]"    : round(v['area'], 2),
                        "Semiplano"     : "Izquierda (Y<mid)" if v['y'] < y_mid_sym else "Derecha (Y≥mid)"
                    } for v in vortices]
                    st.dataframe(pd.DataFrame(area_desg), use_container_width=True, hide_index=True)

                    total_area = area_izq + area_der
                    if total_area > 0:
                        asim     = (area_der - area_izq) / total_area
                        beta_est = asim * 15.0
                        col_iz, col_der, col_asim, col_beta = st.columns(4)
                        col_iz.metric("área Izquierda [mm²]", f"{area_izq:.1f}")
                        col_der.metric("área Derecha [mm²]",  f"{area_der:.1f}")
                        col_asim.metric("Asimetría (±1)", f"{asim:+.3f}")
                        col_beta.metric("β estimado [°]",  f"{beta_est:+.2f}°")
                        st.markdown(f"""
                        <div style="background:#111;border-radius:8px;padding:12px;margin-top:8px;">
                        <p style="color:#aaa;margin:0 0 6px 0;font-size:0.8rem;">Distribución  Izquierda | Derecha →</p>
                        <div style="display:flex;height:20px;border-radius:4px;overflow:hidden;">
                        <div style="background:#ef4444;width:{50*(1+asim):.1f}%;"></div>
                        <div style="background:#3b82f6;width:{50*(1-asim):.1f}%;"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-top:4px;">
                        <span style="color:#ef4444;font-size:0.75rem;">Izq {area_izq:.0f} mm²</span>
                        <span style="color:white;font-size:0.8rem;"><b>β ≈ {beta_est:+.1f}°</b></span>
                        <span style="color:#3b82f6;font-size:0.75rem;">{area_der:.0f} mm² Der</span>
                        </div></div>
                        """, unsafe_allow_html=True)

                        if abs(asim) < 0.05:
                            st.success("✅ Distribución simétrica — sin guiñada detectada (β ≈ 0°)")
                        elif abs(asim) < 0.15:
                            st.warning(f"⚠️ Ligera asimetría. Posible guiñada: β ≈ {beta_est:+.1f}°")
                        else:
                            lado = "derecha (+Y)" if asim > 0 else "izquierda (−Y)"
                            st.error(f"  Asimetría significativa en {lado}. β ≈ {beta_est:+.1f}°")
                    else:
                        st.info("Sin área total detectable para el análisis de simetría.")



elif st.session_state.seccion_actual == 'ensayo_betz':
    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🧪 ENSAYO DE BETZ
            </h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Sección en desarrollo
            </h2>
        </div>
    """, unsafe_allow_html=True)
    st.info("Esta sección se encuentra actualmente en desarrollo y no contiene funcionalidades activas por el momento.")

elif st.session_state.seccion_actual == 'modelos':
    st.markdown("""
    <div class="header-container">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            📦 GESTOR DE MODELOS 3D
        </h1>
        <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Importa o carga modelos de referencia para visualización 4D
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # --- INIT SESSION STATE ---
    if 'modelo_modo' not in st.session_state:
        st.session_state.modelo_modo = 'bd'   # 'bd' o 'importar'
    if 'modelo_cg' not in st.session_state:
        st.session_state.modelo_cg = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    if 'modelo_nombre_bd' not in st.session_state:
        st.session_state.modelo_nombre_bd = None  # None = nuevo, str = actualizar

    # ─────────────────────────────────────────────────────────────────────────
    # LAYOUT PRINCIPAL: izq = configuración, der = preview
    # ─────────────────────────────────────────────────────────────────────────
    c_conf, c_preview = st.columns([1.2, 2])

    with c_conf:

        # ── SELECTOR DE MODO ──────────────────────────────────────────────
        st.markdown("### 🔽 Fuente del Modelo")
        modo_opts = {"🗄 Cargar de Base de Datos": "bd", "📂 Importar Archivo (STL / CSV)": "importar"}
        modo_sel_label = st.radio(
            "Seleccionar fuente:",
            list(modo_opts.keys()),
            index=0 if st.session_state.modelo_modo == 'bd' else 1,
            horizontal=True,
            key="modo_modelo_radio",
            label_visibility="collapsed"
        )
        st.session_state.modelo_modo = modo_opts[modo_sel_label]

        st.markdown("---")

        # ── MODO: CARGAR DE BASE DE DATOS ────────────────────────────────
        if st.session_state.modelo_modo == 'bd':
            try:
                saved_objs = auth.get_user_objects(st.session_state.username)
            except AttributeError:
                st.error("Error: Función get_user_objects no encontrada en auth.py.")
                saved_objs = []

            if not saved_objs:
                st.info("No hay modelos guardados en la base de datos. Importa uno nuevo.")
            else:
                obj_labels = {f"📦 {name} ({o_type}) — {f_date}": (obj_id, name, o_type, d_json, f_date)
                              for obj_id, name, o_type, d_json, f_date in saved_objs}
                sel_label = st.selectbox("Seleccionar modelo guardado:", list(obj_labels.keys()), key="sel_modelo_bd")
                
                if sel_label:
                    obj_id, name, o_type, d_json, f_date = obj_labels[sel_label]
                    
                    # Info del modelo
                    st.markdown(f"""
                    <div style="background:#111; border:1px solid #333; border-radius:8px; padding:12px; margin-bottom:12px;">
                        <p style="color:#888; margin:0; font-size:0.8rem;">Nombre</p>
                        <p style="color:white; font-weight:bold; margin:0 0 8px 0;">{name}</p>
                        <p style="color:#888; margin:0; font-size:0.8rem;">Tipo / Guardado</p>
                        <p style="color:#aaa; margin:0;">{o_type} | {f_date}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Restaurar CG si está guardado en el JSON
                    try:
                        _data_preview = json.loads(d_json)
                        if 'cg' in _data_preview:
                            _cg = _data_preview['cg']
                            st.session_state.modelo_cg = {'x': _cg.get('x', 0.0), 'y': _cg.get('y', 0.0), 'z': _cg.get('z', 0.0)}
                    except:
                        pass

                    col_load, col_del = st.columns(2)
                    with col_load:
                        if st.button("📥 Seleccionar este modelo", use_container_width=True, type="primary", key="btn_select_bd"):
                            try:
                                data_loaded = json.loads(d_json)
                                data_loaded['x'] = np.array(data_loaded['x'])
                                data_loaded['y'] = np.array(data_loaded['y'])
                                data_loaded['z'] = np.array(data_loaded['z'])
                                if 'i' in data_loaded:
                                    data_loaded['i'] = np.array(data_loaded['i'])
                                    data_loaded['j'] = np.array(data_loaded['j'])
                                    data_loaded['k'] = np.array(data_loaded['k'])
                                # Restaurar CG guardado
                                if 'cg' in data_loaded:
                                    st.session_state.modelo_cg = data_loaded['cg']

                                st.session_state.objeto_referencia_4d = data_loaded
                                st.session_state.objeto_referencia_base = data_loaded.copy()
                                st.session_state.modelo_nombre_bd = name
                                st.success(f"✅ '{name}' cargado.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error al cargar: {e}")
                    with col_del:
                        if st.button("🗑️ Eliminar", use_container_width=True, key="btn_del_bd"):
                            try:
                                auth.delete_user_object(obj_id)
                                st.session_state.modelo_nombre_bd = None
                                st.rerun()
                            except:
                                st.error("Error al eliminar")

        # ── MODO: IMPORTAR ARCHIVO ────────────────────────────────────────
        else:
            st.markdown("##### 📂 Cargar Archivo STL o CSV")
            st.caption("El modelo se importa con los ejes del archivo. Los desplazamientos y rotaciones se configuran desde la sección 4D.")

            use_auto_center_imp = st.checkbox(" Auto-centrar objeto al importar (centroide → origen)", value=True, key="auto_center_imp")

            file_obj = st.file_uploader("Cargar archivo (STL o CSV):", type=['csv', 'stl'], key="uploader_modelo_imp")

            if file_obj and st.button("📥 Procesar e importar", type="primary", use_container_width=True, key="btn_importar_modelo"):
                file_ext = file_obj.name.split('.')[-1].lower()
                x_points = y_points = z_points = faces_i = faces_j = faces_k = None
                obj_type = None

                try:
                    if file_ext == 'csv':
                        df_obj = pd.read_csv(file_obj, sep=None, engine='python')
                        cols_map = {c.lower(): c for c in df_obj.columns}
                        if 'x' in cols_map and 'y' in cols_map and 'z' in cols_map:
                            x_points = df_obj[cols_map['x']].values
                            y_points = df_obj[cols_map['y']].values
                            z_points = df_obj[cols_map['z']].values
                            obj_type = 'scatter'
                        else:
                            st.error("El CSV requiere columnas X, Y, Z")

                    elif file_ext == 'stl':
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                            tmp.write(file_obj.getvalue())
                            tmp_path = tmp.name
                        mesh = pv.read(tmp_path)
                        os.unlink(tmp_path)
                        if not mesh.is_all_triangles:
                            mesh = mesh.triangulate()
                        points = mesh.points
                        x_points = points[:, 0]
                        y_points = points[:, 1]
                        z_points = points[:, 2]
                        faces = mesh.faces.reshape(-1, 4)[:, 1:]
                        faces_i = faces[:, 0]
                        faces_j = faces[:, 1]
                        faces_k = faces[:, 2]
                        obj_type = 'mesh'

                    if x_points is not None:
                        if use_auto_center_imp:
                            cx = (np.min(x_points) + np.max(x_points)) / 2
                            cy = (np.min(y_points) + np.max(y_points)) / 2
                            cz = (np.min(z_points) + np.max(z_points)) / 2
                            x_points -= cx; y_points -= cy; z_points -= cz

                        obj_data = {'type': obj_type, 'x': x_points, 'y': y_points, 'z': z_points, 'name': file_obj.name}
                        if obj_type == 'mesh':
                            obj_data.update({'i': faces_i, 'j': faces_j, 'k': faces_k})

                        st.session_state.objeto_referencia_4d = obj_data
                        st.session_state.objeto_referencia_base = obj_data.copy()
                        st.session_state.modelo_nombre_bd = None  # es nuevo → botón GUARDAR
                        st.success(f"✅ Importado: {file_obj.name}")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error procesando archivo: {e}")


        # ── SISTEMA DE REFERENCIA Y CENTRO DE GRAVEDAD ───────────────────
        if 'objeto_referencia_4d' in st.session_state:
            st.markdown("---")
            st.markdown("""
            <div style="background:#0a1628; border:1px solid #1e3a5f; border-radius:8px; padding:14px; margin-bottom:14px;">
                <h4 style="color:#60a5fa; margin:0 0 8px 0;">🧭 Sistema de Referencia del Modelo</h4>
                <p style="color:#93c5fd; font-size:0.85rem; margin:0; line-height:1.6;">
                    <b>Origen (Datum):</b> Nariz del avión<br>
                    <b>X+</b> → hacia la cola &nbsp;|&nbsp; <b>Y+</b> → semiala derecha &nbsp;|&nbsp; <b>Z+</b> → techo
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("#### 🎯 Centro de Gravedad (CG) — Punto de Rotación")
            st.caption("El CG define el pivote para el cabeceo (Alpha) y la guiñada (Beta) en la visualización 4D.")

            cg_cols = st.columns(3)
            st.session_state.modelo_cg['x'] = cg_cols[0].number_input("CG — X [mm desde nariz]", value=float(st.session_state.modelo_cg.get('x', 0.0)), step=5.0, format="%.1f", key="cg_x")
            st.session_state.modelo_cg['y'] = cg_cols[1].number_input("CG — Y [mm]", value=float(st.session_state.modelo_cg.get('y', 0.0)), step=5.0, format="%.1f", key="cg_y")
            st.session_state.modelo_cg['z'] = cg_cols[2].number_input("CG — Z [mm]", value=float(st.session_state.modelo_cg.get('z', 0.0)), step=5.0, format="%.1f", key="cg_z")

            if st.button(" Auto-centrar CG (X=0, Y/Z = centroide del modelo)", use_container_width=True, key="btn_autocg"):
                obj_actual = st.session_state.objeto_referencia_4d
                x_arr = np.array(obj_actual['x']); y_arr = np.array(obj_actual['y']); z_arr = np.array(obj_actual['z'])
                st.session_state.modelo_cg['x'] = 0.0
                st.session_state.modelo_cg['y'] = float((np.min(y_arr) + np.max(y_arr)) / 2)
                st.session_state.modelo_cg['z'] = float((np.min(z_arr) + np.max(z_arr)) / 2)
                st.success(f"✅ CG auto-calculado: X=0, Y={st.session_state.modelo_cg['y']:.1f}, Z={st.session_state.modelo_cg['z']:.1f} mm")
                st.rerun()

            # ── NOMBRE DEL MODELO ────────────────────────────────────────
            st.markdown("---")
            nombre_actual = st.session_state.objeto_referencia_4d.get('name', 'MiModelo')
            nombre_modelo = st.text_input(" Nombre del modelo:", value=nombre_actual, key="nombre_modelo_final")

            # ── BOTONES FINALES ──────────────────────────────────────────
            c_btn1, c_btn2 = st.columns(2)

            # GUARDAR / ACTUALIZAR
            es_existente = st.session_state.modelo_nombre_bd is not None
            lbl_save = f"🔄 ACTUALIZAR '{st.session_state.modelo_nombre_bd}'" if es_existente else "💾 GUARDAR modelo"

            with c_btn1:
                if st.button(lbl_save, use_container_width=True, type="primary", key="btn_guardar_modelo"):
                    obj_to_save = st.session_state.objeto_referencia_4d.copy()
                    obj_to_save['name'] = nombre_modelo
                    obj_to_save['cg'] = st.session_state.modelo_cg.copy()

                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.ndarray): return obj.tolist()
                            return json.JSONEncoder.default(self, obj)

                    try:
                        json_str = json.dumps(obj_to_save, cls=NumpyEncoder)
                        if auth.save_user_object(st.session_state.username, nombre_modelo, obj_to_save['type'], json_str):
                            st.success(f"✅ '{nombre_modelo}' guardado en la BD.")
                            st.session_state.modelo_nombre_bd = nombre_modelo
                        else:
                            st.error("Error al guardar en la base de datos.")
                    except Exception as e:
                        st.error(f"Error serializando: {e}")

            # USAR EN LA PÁGINA
            with c_btn2:
                if st.button("✅ USAR MODELO EN LA PÁGINA", use_container_width=True, key="btn_usar_modelo"):
                    st.session_state.objeto_referencia_4d['name'] = nombre_modelo
                    st.session_state.objeto_referencia_4d['cg'] = st.session_state.modelo_cg.copy()
                    st.success(f"✅ Modelo '{nombre_modelo}' activado. CG: X={st.session_state.modelo_cg['x']:.1f}, Y={st.session_state.modelo_cg['y']:.1f}, Z={st.session_state.modelo_cg['z']:.1f} mm")

    # ── PREVIEW 3D ────────────────────────────────────────────────────────
    with c_preview:
        st.markdown("### ⚙️ Vista Previa 3D")
        if 'objeto_referencia_4d' in st.session_state:
            obj = st.session_state.objeto_referencia_4d
            cg = st.session_state.modelo_cg
            fig_prev = go.Figure()

            if obj['type'] == 'mesh':
                fig_prev.add_trace(go.Mesh3d(
                    x=obj['x'], y=obj['y'], z=obj['z'],
                    i=obj['i'], j=obj['j'], k=obj['k'],
                    color='#4a90d9', opacity=0.75, name=obj['name'],
                    alphahull=0, showscale=False, lighting=dict(ambient=0.4, diffuse=0.8)
                ))
            elif obj['type'] == 'scatter':
                fig_prev.add_trace(go.Scatter3d(
                    x=obj['x'], y=obj['y'], z=obj['z'],
                    mode='markers', marker=dict(size=2, color='#4a90d9', opacity=0.8),
                    name=obj['name']
                ))

            # Mostrar CG como punto rojo
            fig_prev.add_trace(go.Scatter3d(
                x=[cg['x']], y=[cg['y']], z=[cg['z']],
                mode='markers+text',
                marker=dict(size=10, color='#ff4444', symbol='cross'),
                text=["CG"], textposition="top center",
                textfont=dict(color='#ff4444', size=12),
                name="Centro de Gravedad"
            ))

            fig_prev.update_layout(
                scene=dict(
                    aspectmode='data',
                    xaxis_title="X (Longitudinal)",
                    yaxis_title="Y (Transversal)",
                    zaxis_title="Z (Vertical)",
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=0, r=0, b=0, t=30),
                height=550,
                legend=dict(font=dict(color='white'))
            )
            st.plotly_chart(fig_prev, use_container_width=True)

            # Info del modelo activo
            x_arr = np.array(obj['x']); y_arr = np.array(obj['y']); z_arr = np.array(obj['z'])
            st.markdown(f"""
            <div style="background:#111; border:1px solid #333; border-radius:8px; padding:12px; margin-top:8px;">
                <p style="color:#888; margin:0; font-size:0.8rem;">Modelo activo</p>
                <p style="color:white; font-weight:bold; margin:0 0 6px 0;">{obj.get('name','—')}</p>
                <p style="color:#aaa; margin:0; font-size:0.8rem;">
                    Tipo: {obj.get('type','—')} | Vértices: {len(x_arr):,}<br>
                    X: [{np.min(x_arr):.1f}, {np.max(x_arr):.1f}] mm<br>
                    Y: [{np.min(y_arr):.1f}, {np.max(y_arr):.1f}] mm<br>
                    Z: [{np.min(z_arr):.1f}, {np.max(z_arr):.1f}] mm
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0a0a0a; border:2px dashed #333; border-radius:12px; padding:3rem; text-align:center; margin-top:2rem;">
                <p style="font-size:3rem; margin:0;">📦</p>
                <p style="color:#666; margin:8px 0 0 0;">Ningún modelo cargado aún.<br>Carga uno desde la base de datos o importa un archivo STL/CSV.</p>
            </div>
            """, unsafe_allow_html=True)



elif st.session_state.seccion_actual == '3d' or st.session_state.seccion_actual == 'betz_3d':
    st.markdown("# 🌪️ VISUALIZACIÓN DE ESTELA 3D - Análisis Tridimensional")
    st.markdown("Análisis 3D con superficie interactiva de presiones")
    
    # Paso 1: Configuración inicial
    with st.expander("💾 PASO 1: Configuración de Geometría y Sensores", expanded=True):
        st.markdown("""
        <div class="section-card" style="margin-bottom: 12px;">
            <h3 style="margin-top:0; color:white;">💾 PASO 1: CONFIGURACIÓN INICIAL</h3>
            <p style="color:#bbb; margin-bottom:0;">
                Defina los parámetros físicos del peine de sensores y el sistema de adquisición para el entorno 3D.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Reorganizar: datos a la izquierda, imagen más pequeña a la derecha
        col_datos, col_imagen = st.columns([2, 1])

        with col_datos:
            st.markdown("### GEOMETRÍA")
            
            # Orden de sensores
            orden_sensores = st.selectbox(
                "Orden de lectura de sensores:",
                ["asc", "des"],
                format_func=lambda x: "Ascendente (sensor 1 más abajo al 12 más arriba)" if x == "asc" else "Descendente (sensor 12 más abajo y sensor 1 más arriba)",
                help="Define cómo se leen los datos de los sensores en relación a su posición física",
                key="orden_3d"
            )
            
            # Pregunta sobre el sensor de referencia
            st.info(" **Pregunta:** ¿Qué sensor corresponde a la toma número 12 (la que se encuentra cerca del piso)?")
            sensor_referencia = st.selectbox(
                "Sensor de referencia (toma 12):",
                [f"Sensor {i}" for i in range(1, 13)],
                index=11,  # Por defecto Sensor 12
                help="Seleccione el sensor que corresponde a la toma física número 12",
                key="sensor_ref_3d"
            )
            
            distancia_toma_12 = st.number_input(
                "Distancia de la toma 12 a la posición X=0, Y=0 (coordenadas del traverser) [mm]:",
                value=-120.0,
                step=1.0,
                format="%.1f",
                help="Distancia en mm desde el punto de referencia del traverser",
                key="dist_toma_3d"
            )
            
            distancia_entre_tomas = st.number_input(
                "Distancia entre tomas [mm]:",
                value=10.0,
                step=0.01,
                format="%.2f",
                help="Distancia física entre tomas consecutivas según el plano técnico",
                key="dist_entre_3d"
            )
            
            # Guardar configuración
            if st.button("💾 Guardar Configuración 3D", type="primary", key="save_3d"):
                st.session_state.configuracion_3d = {
                    'orden': orden_sensores,
                    'sensor_referencia': sensor_referencia,
                    'distancia_toma_12': distancia_toma_12,
                    'distancia_entre_tomas': distancia_entre_tomas
                }
                st.success("✅ Configuración 3D guardada correctamente")
                st.rerun()

    with col_imagen:
        st.markdown("###  Diagrama de Referencia")
        st.markdown("""
        <div style="background: #f8fafc; border: 2px dashed #e5e7eb; border-radius: 12px; padding: 2rem; text-align: center; color: #64748b;">
            <h4> Diagrama de Referencia</h4>
            <p>Aquí iría el diagrama técnico de sensores</p>
            <p><small>Subir imagen del plano técnico</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Mostrar configuración actual
    if st.session_state.get('configuracion_3d'):
        st.markdown("---")
        # Estilo de configuración actual
        st.markdown("""
        <div class="section-card" style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: white;">⚙️ CONFIGURACIÓN ACTIVA</h3>
                <span style="color: #666; font-size: 0.8rem;">Parámetros 3D</span>
            </div>
            <div style="display: flex; gap: 20px; margin-top: 15px;">
                <div style="background: #111; padding: 10px; border-radius: 6px; border: 1px solid #333; flex: 1;">
                    <p style="color: #888; font-size: 0.8rem; margin: 0;">Orden</p>
                    <p style="color: white; font-weight: bold; margin: 0;">{}</p>
                </div>
                <div style="background: #111; padding: 10px; border-radius: 6px; border: 1px solid #333; flex: 1;">
                    <p style="color: #888; font-size: 0.8rem; margin: 0;">Toma 12</p>
                    <p style="color: white; font-weight: bold; margin: 0;">{:.1f} mm</p>
                </div>
                <div style="background: #111; padding: 10px; border-radius: 6px; border: 1px solid #333; flex: 1;">
                    <p style="color: #888; font-size: 0.8rem; margin: 0;">Sep. Tomas</p>
                    <p style="color: white; font-weight: bold; margin: 0;">{:.2f} mm</p>
                </div>
            </div>
        </div>
        """.format(
            st.session_state.configuracion_3d['orden'].upper(),
            st.session_state.configuracion_3d['distancia_toma_12'],
            st.session_state.configuracion_3d['distancia_entre_tomas']
        ), unsafe_allow_html=True)
        
        # --- PASO 2: CARGA DE ARCHIVOS 3D ---
        st.markdown("""
        <div class="section-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: white;"> PASO 2: IMPORTACIÓN DE VOLÚMENES 3D</h3>
            <p style="color: #bbb; margin-bottom: 20px;">Cargue múltiples archivos CSV para generar superficies tridimensionales.</p>
        </div>
        """, unsafe_allow_html=True)

        # Almacenar múltiples archivos 3D
        if 'archivos_3d_cargados' not in st.session_state:
            st.session_state.archivos_3d_cargados = {}

        uploaded_files_3d = st.file_uploader(
            "Arrastre sus archivos CSV 3D aquí",
            type=['csv'],
            accept_multiple_files=True,
            key="uploader_betz3d"
        )
        uploaded_infinito_3d = st.file_uploader("🔗 'Valores en el infinito.txt' (Opcional - datos atmosféricos)", type=['txt', 'csv'], accept_multiple_files=False, key="upl_inf_3d")
        
        if uploaded_files_3d:
            st.markdown("<br>", unsafe_allow_html=True)
            for uploaded_file_3d in uploaded_files_3d:
                nombre_archivo = uploaded_file_3d.name.replace('.csv', '').replace('incertidumbre_', '')
                
                if nombre_archivo not in st.session_state.archivos_3d_cargados:
                    with st.spinner(f" Procesando geometría 3D para {nombre_archivo}..."):
                        datos_3d = procesar_promedios(uploaded_file_3d, st.session_state.configuracion_3d['orden'], uploaded_infinito_3d)
                        
                        if datos_3d is not None:
                            st.session_state.archivos_3d_cargados[nombre_archivo] = datos_3d
                            
                            sub_archivos_3d = crear_sub_archivos_3d_por_tiempo_y_posicion(datos_3d, nombre_archivo)
                            st.session_state.sub_archivos_3d_generados.update(sub_archivos_3d)
                            
                            st.success(f"✅ Geometría reconstruida: {nombre_archivo}")

        # Mostrar archivos cargados (Resumen)
        if st.session_state.archivos_3d_cargados:
            st.markdown("### 📋 Resumen de Carga")
            
            # Grid layout for files
            cols = st.columns(3)
            for idx, (nombre, datos) in enumerate(st.session_state.archivos_3d_cargados.items()):
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.markdown(f"**📦 {nombre}**")
                        st.caption(f"{len(datos)} Puntos • {len(datos['Tiempo_s'].unique())} Tiempos")
                        st.progress(100) # Visual indicator that it is ready

            st.markdown("---")
            
            # --- PASO 3: VISUALIZACIÓN ---
            st.markdown("""
            <div class="section-card" style="margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: white;">🌪️ PASO 3: INTERACCIÓN 3D</h3>
                <p style="color: #bbb; margin-bottom: 20px;">Explore la superficie generada, ajuste la cámara y analice la distribución de presiones.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🎛️ Controles Globales de Escena")
            st.info("Visualización estándar activa.")
            # Controls removed as per user request
            mostrar_cuerpo = False # Default disabled
            aspect_ratio = "auto" # Default auto

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        opciones_var_3d = ["Presión Total [Actual]", "rho_inf", "V_inf", "P_inf"]
        var_3d_sel = st.selectbox("📊 Variable a visualizar:", opciones_var_3d, key="var_3d_sel_ui")

        # 🔘 Checkbox para activar/desactivar puntos medidos
        mostrar_puntos_3d = st.checkbox("🔘 Mostrar puntos originales (Nube de puntos)", value=False, key="mostrar_puntos_3d")

        if st.session_state.archivos_3d_cargados:
            st.markdown("#### 📂 Selección de Geometría a Visualizar")
            
            # Mostrar archivos como botones seleccionables
            archivos_disponibles = list(st.session_state.archivos_3d_cargados.keys())
            
            # Crear columnas para mostrar archivos
            cols_per_row = 3
            for i in range(0, len(archivos_disponibles), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, col in enumerate(cols):
                    if i + j < len(archivos_disponibles):
                        nombre_archivo = archivos_disponibles[i + j]
                        datos_archivo = st.session_state.archivos_3d_cargados[nombre_archivo]
                        
                        with col:
                            # Crear card para cada archivo
                            st.markdown(f"""
                            <div style="
                                background: white;
                                border: 2px solid #e5e7eb;
                                border-radius: 12px;
                                padding: 1rem;
                                margin: 0.5rem 0;
                                text-align: center;
                                transition: all 0.3s ease;
                                cursor: pointer;
                            ">
                                <h4 style="color: #08596C; margin-bottom: 0.5rem;">📊 {nombre_archivo}</h4>
                                <p style="color: #6b7280; font-size: 0.9rem; margin-bottom: 0.5rem;">
                                    {len(datos_archivo)} registros
                                </p>
                                <p style="color: #6b7280; font-size: 0.8rem;">
                                    Tiempos: {len(datos_archivo['Tiempo_s'].dropna().unique())}<br>
                                    Pos. Y (Trav): {len(datos_archivo['Pos_Y_Traverser'].dropna().unique())}<br>
                                    Pos. Z (Base): {len(datos_archivo['Pos_Z_Base'].dropna().unique())}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if st.button(f"  Ver Superficie Completa", key=f"ver_mesh3d_{nombre_archivo}", use_container_width=True):
                                with st.spinner(f"Construyendo superficie completa para {nombre_archivo}..."):
                                    # Llamada a la NUEVA función de graficación con 300 puntos
                                    fig_individual = crear_superficie_delaunay_3d(
                                        datos_archivo,
                                        st.session_state.configuracion_3d,
                                        nombre_archivo,
                                        mostrar_puntos=mostrar_puntos_3d,  #  Aquí
                                        variable=var_3d_sel
                                    )
                                    
                                    if fig_individual:
                                        st.plotly_chart(fig_individual, use_container_width=False)
                                        
                                        # Información del archivo
                                        st.success(f"✅ Superficie de malla 3D generada para: **{nombre_archivo}** usando {len(fig_individual.data[0].x)} vértices.")
                                        
                                        # Botón de descarga
                                        html_individual = fig_individual.to_html()
                                        st.download_button(
                                            label=f"📊 Descargar Malla 3D (HTML)",
                                            data=html_individual,
                                            file_name=f"mesh_3d_{nombre_archivo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                            mime="text/html",
                                            key=f"download_mesh3d_{nombre_archivo}"
                                        )
                                    else:
                                        st.error(f"  No se pudo generar la superficie de malla 3D para {nombre_archivo}")   
        
        # --- NUEVO PASO 4: DIFERENCIA ENTRE SUPERFICIES ---
        st.markdown("""
        <div class="section-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: white;">📉 PASO 4: ANÁLISIS DE DIFERENCIAS</h3>
            <p style="color: #bbb; margin-bottom: 20px;">Calcule y visualice la diferencia aritmética entre dos superficies medidas (A - B).</p>
        </div>
        """, unsafe_allow_html=True)

        if len(st.session_state.get('archivos_3d_cargados', {})) >= 2:
            archivos_disponibles_diff = list(st.session_state.get('archivos_3d_cargados', {}).keys())

            col1_diff, col2_diff = st.columns(2)
            with col1_diff:
                st.markdown("**Superficie A (Base)**")
                archivo_a = st.selectbox(
                    "Minuendo",
                    archivos_disponibles_diff,
                    label_visibility="collapsed",
                    key="diff_3d_a"
                )
            with col2_diff:
                st.markdown("**Superficie B (Resta)**")
                archivo_b = st.selectbox(
                    "Sustraendo",
                    archivos_disponibles_diff,
                    index=1 if len(archivos_disponibles_diff) > 1 else 0,
                    label_visibility="collapsed",
                    key="diff_3d_b"
                )
            
            # 🔘 Checkbox para activar/desactivar puntos en diferencia
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            mostrar_puntos_diff = st.checkbox("🔘 Incluir puntos de medición en el gráfico de diferencia", value=True, key="mostrar_puntos_diff")

            # PARTE 1: El botón "Calcular" solo genera el gráfico y lo guarda en una "bandeja" temporal.
            if st.button("Calcular Diferencia de Superficies", use_container_width=True, type="primary"):
                if archivo_a == archivo_b:
                    st.error("Por favor, selecciona dos archivos diferentes para comparar.")
                else:
                    with st.spinner(f"Calculando diferencia entre '{archivo_a}' y '{archivo_b}'..."):
                        datos_a = st.session_state.archivos_3d_cargados[archivo_a]
                        datos_b = st.session_state.archivos_3d_cargados[archivo_b]
                        
                        fig_diferencia_3d = crear_superficie_diferencia_delaunay_3d(
                            datos_a,
                            datos_b,
                            archivo_a,
                            archivo_b,
                            st.session_state.configuracion_3d,
                            mostrar_puntos=mostrar_puntos_diff  #  Aquí
                        )

                        if fig_diferencia_3d:
                            # Guardamos la figura y su nombre en la sesión para que sobrevivan al reinicio de la página.
                            st.session_state.figura_diferencia_temporal_3d = {
                                "fig": fig_diferencia_3d,
                                "nombre": f"Dif. 3D: {archivo_a} vs {archivo_b}"
                            }
                        else:
                            if 'figura_diferencia_temporal_3d' in st.session_state:
                                del st.session_state.figura_diferencia_temporal_3d
            
            # PARTE 2: Este bloque (fuera del anterior) revisa si hay algo en la "bandeja" temporal y lo muestra.
            if 'figura_diferencia_temporal_3d' in st.session_state:
                temp_data_3d = st.session_state.figura_diferencia_temporal_3d
                fig_diferencia_3d = temp_data_3d["fig"]
                nombre_guardado_3d = temp_data_3d["nombre"]

                # 1. Mostramos el gráfico
                st.plotly_chart(fig_diferencia_3d, use_container_width=False)
                
                # 2. Mostramos el botón "Guardar". Ahora sí funcionará.
                if st.button("💾 Guardar Diferencia para Visualizar", key="save_diff_3d_for_viz_final"):
                    if 'diferencias_guardadas' not in st.session_state:
                        st.session_state.diferencias_guardadas = {}
                    
                    st.session_state.diferencias_guardadas[nombre_guardado_3d] = fig_diferencia_3d
                    st.success(f"✅ Gráfico '{nombre_guardado_3d}' guardado permanentemente.")
                    
                    # Borramos la figura temporal después de guardarla para limpiar la "bandeja"
                    del st.session_state.figura_diferencia_temporal_3d
                    st.rerun()

        else:
            st.info("Carga al menos dos archivos 3D para poder calcular una diferencia.")

        def extraer_pos_x_estacion(nombre_archivo):
            """
            Intenta extraer la posición X (Estación) del nombre del archivo.
            Patrones buscados: X100, Est100, Station100, _100mm, etc.
            """
            import re
            nombre = os.path.basename(str(nombre_archivo))
            # Patrones comunes: X100, X-100, X_100, Est100, etc.
            patrones = [
                r'[Xx][_\-]?(-?\d+)',        # X100, X-100, X_100
                r'[Ee]st[_\-]?(-?\d+)',      # Est100
                r'[Ss]tation[_\-]?(-?\d+)',  # Station100
                r'[Pp]os[_\-]?(-?\d+)'       # Pos100
            ]
            
            for p in patrones:
                m = re.search(p, nombre)
                if m:
                    try:
                        return float(m.group(1))
                    except:
                        continue
            return 0.0

# --- PASO 5: GUARDAR SUPERFICIE EN BASE DE DATOS ---
        st.markdown("""
        <div class="section-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: white;">💾 PASO 5: PERSISTENCIA DE DATOS</h3>
            <p style="color: #bbb; margin-bottom: 0;">Almacene la superficie procesada en la base de datos centralizada.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.get('archivos_3d_cargados'):
            
            # Seleccionar archivo para guardar
            archivos_para_guardar = list(st.session_state.archivos_3d_cargados.keys())
            archivo_guardar = st.selectbox("Seleccionar Archivo para Guardar:", archivos_para_guardar, key="sel_guardar_bd")
            
            if archivo_guardar:
                df_guardar = st.session_state.archivos_3d_cargados[archivo_guardar]
                
                # Filtrar por tiempo si hay múltiples
                tiempos_g = df_guardar['Tiempo_s'].dropna().unique()
                tiempo_original = tiempos_g[0] if len(tiempos_g) > 0 else 5
                tiempo_g_sel = st.number_input("Ingresar Tiempo:", value=5, step=1, key="tiempo_guardar_bd")
                
                # Filtrar datos
                if tiempo_g_sel in tiempos_g:
                    df_filtrado_g = df_guardar[df_guardar['Tiempo_s'] == tiempo_g_sel].copy()
                else:
                    df_filtrado_g = df_guardar[df_guardar['Tiempo_s'] == tiempo_original].copy()
                df_filtrado_g['Tiempo_s'] = tiempo_g_sel
                
                # INTELIGENCIA: Pre-calcular X y Nombre basado en archivo
                x_detectado = extraer_pos_x_estacion(archivo_guardar)

                c_nm1, c_nm2 = st.columns(2)
                aoa_3d = c_nm1.number_input(" Ángulo de Ataque [mm]:", value=0.0, step=0.5, format="%.1f", key="aoa_3d")
                x_detectado_inp = c_nm2.number_input(" Posición X (Estación) [mm]:", value=x_detectado, step=10.0, key="x_3d_inp")

                # Nombre auto-sugerido: 3D-Xpos-OAOgrados-Tts  (editable)
                _aoa_str = str(int(aoa_3d)) if aoa_3d == int(aoa_3d) else f"{aoa_3d:.1f}"
                _aoa_str = _aoa_str.replace("-", "neg")
                nombre_base_sugerido = f"3D-X{int(x_detectado_inp)}-OAO{_aoa_str}-T{int(tiempo_g_sel)}s"

                c_g1, c_g2 = st.columns([0.15, 0.85])
                usar_custom_3d = c_g1.checkbox("Nombre libre", key="custom_nom_3d")
                if usar_custom_3d:
                    nombre_surf = c_g2.text_input("Nombre personalizado:", placeholder=nombre_base_sugerido, key="nombre_surf_custom")
                    if not nombre_surf:
                        nombre_surf = nombre_base_sugerido
                else:
                    nombre_surf = nombre_base_sugerido
                    c_g2.code(nombre_surf)
                
                if st.button("💾 Guardar en Base de Datos", key="btn_guardar_bd"):
                    # Convertir a matriz (Y, Z, Presion)
                    results_g = []
                    for _, row in df_filtrado_g.iterrows():
                        y_trav = row.get('Pos_Y_Traverser')
                        z_base = row.get('Pos_Z_Base')
                        for col in df_filtrado_g.columns:
                            num_sensor = obtener_numero_sensor_desde_columna(col)
                            if num_sensor is not None:
                                val_presion = row[col]
                                if pd.isna(val_presion): continue
                                z_real = calcular_altura_absoluta_z(
                                    num_sensor, z_base,
                                    st.session_state.configuracion_3d.get('distancia_toma_12', -120),
                                    st.session_state.configuracion_3d.get('distancia_entre_tomas', 10.91),
                                    12,
                                    st.session_state.configuracion_3d.get('orden', 'asc')
                                )
                                results_g.append({'Y': y_trav, 'Z': z_real, 'Presion': val_presion})

                    df_final_g = pd.DataFrame(results_g)
                    if not df_final_g.empty:
                        json_str = df_final_g.to_json(orient='records')
                        if auth.save_surface_data(st.session_state.username, nombre_surf, json_str):
                            st.success(f"✅ Superficie 3D guardada: **{nombre_surf}**")
                        else:
                            st.error("Error al guardar en base de datos.")
                    else:
                        st.error("No se pudieron extraer datos válidos (Y, Z, Presion).")
        
        else:
            st.info("ℹ️ Para guardar una superficie, primero debe procesar al menos un archivo CSV en el **Paso 2**.")


elif st.session_state.seccion_actual == 'betz_4d':
    st.markdown("""
    <div class="header-container">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🌌 VISUALIZACIÓN DE ESTELA 4D
        </h1>
        <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Visualización Multidimensional y Animación
        </h2>
    </div>
    """, unsafe_allow_html=True)

    #   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈  ?
    # PASO 1: GUARDAR PLANO 4D EN DRIVE
    #   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈  ?
    with st.expander("💾 PASO 1: Cargar y Guardar Plano 4D en Base de Datos", expanded=True):
        st.markdown("""
        <div class="section-card" style="margin-bottom: 12px;">
            <h3 style="margin-top:0; color:white;">💾 PASO 1: GUARDAR PLANO 4D</h3>
            <p style="color:#bbb; margin-bottom:0;">
                Procese un archivo de incertidumbre y guárdelo en la carpeta <strong>4D</strong> del Drive con su posición X.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- Config sensores ---
        config_4d = mostrar_configuracion_sensores("4d")

        # --- Upload ---
        archivo_4d = st.file_uploader(
            "📂 Subir archivo CSV de incertidumbre (4D):",
            type=['csv'], key="upload_4d_paso1"
        )
        uploaded_infinito_4d = st.file_uploader("🔗 'Valores en el infinito.txt' (Opcional - datos atmosféricos)", type=['txt', 'csv'], accept_multiple_files=False, key="upl_inf_4d")

        if archivo_4d and config_4d:
            datos_4d = procesar_promedios(archivo_4d, config_4d.get('orden', 'asc'), uploaded_infinito_4d)

            if datos_4d is not None:
                tiempos_4d = sorted(datos_4d['Tiempo_s'].dropna().unique())
                t4d_original = tiempos_4d[0] if len(tiempos_4d) > 0 else 5
                t4d_sel = st.number_input("Ingresar Tiempo:", value=5, step=1, key="t4d_sel")
                if t4d_sel in tiempos_4d:
                    df_4d_filtrado = datos_4d[datos_4d['Tiempo_s'] == t4d_sel].copy()
                else:
                    df_4d_filtrado = datos_4d[datos_4d['Tiempo_s'] == t4d_original].copy()
                df_4d_filtrado['Tiempo_s'] = t4d_sel

                x_pos_4d = st.number_input(
                    " Posición X (Estación) [mm]:",
                    value=0.0, step=10.0, key="x_pos_4d"
                )

                aoa_4d = st.number_input(" Ángulo de Ataque [mm]:", value=0.0, step=0.5, format="%.1f", key="aoa_4d")

                # Nombre auto-sugerido: 4D-Xpos-OAOgrados-Tts  (editable)
                _aoa_str_4d = str(int(aoa_4d)) if aoa_4d == int(aoa_4d) else f"{aoa_4d:.1f}"
                _aoa_str_4d = _aoa_str_4d.replace("-", "neg")
                nombre_sugerido_4d = f"4D-X{int(x_pos_4d)}-OAO{_aoa_str_4d}-T{int(t4d_sel)}s"

                c_4d1, c_4d2 = st.columns([0.15, 0.85])
                usar_custom_4d = c_4d1.checkbox("Nombre libre", key="custom_nom_4d")
                if usar_custom_4d:
                    nombre_4d = c_4d2.text_input("Nombre personalizado:", placeholder=nombre_sugerido_4d, key="nombre_4d_inp")
                    if not nombre_4d:
                        nombre_4d = nombre_sugerido_4d
                else:
                    nombre_4d = nombre_sugerido_4d
                    c_4d2.code(nombre_4d)
                

                if st.button("💾 Guardar Plano 4D en Drive", key="btn_guardar_4d"):
                    results_4d = []
                    for _, row in df_4d_filtrado.iterrows():
                        y_trav = row.get('Pos_Y_Traverser')
                        z_base = row.get('Pos_Z_Base')
                        for col4 in df_4d_filtrado.columns:
                            num_s = obtener_numero_sensor_desde_columna(col4)
                            if num_s is not None:
                                val_p = row[col4]
                                if pd.isna(val_p): continue
                                z_real = calcular_altura_absoluta_z(
                                    num_s, z_base,
                                    config_4d.get('distancia_toma_12', -120),
                                    config_4d.get('distancia_entre_tomas', 10.91),
                                    12,
                                    config_4d.get('orden', 'asc')
                                )
                                results_4d.append({'Y': y_trav, 'Z': z_real, 'Presion': val_p})

                    df_4d_final = pd.DataFrame(results_4d)
                    if not df_4d_final.empty:
                        json_4d = df_4d_final.to_json(orient='records')
                        if auth.save_surface_data_4d(st.session_state.username, nombre_4d, x_pos_4d, json_4d):
                            st.success(f"✅ Plano 4D guardado: **{nombre_4d}** en X={x_pos_4d} mm")
                            st.cache_data.clear()
                        else:
                            st.error("  Error al guardar el plano 4D.")
                    else:
                        st.error("No se pudieron extraer datos válidos.")

    st.markdown("---")


    st.markdown("---")

    #   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈  ?
    # PASO 2: VISUALIZACIÓN 4D
    #   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈   ≈  ?
    st.markdown("### 🌌 Paso 2: Visualización 4D")

    # --- INIT session state para modelo 4D ---
    if 'modelo_4d_alpha' not in st.session_state: st.session_state.modelo_4d_alpha = 0.0
    if 'modelo_4d_beta'  not in st.session_state: st.session_state.modelo_4d_beta  = 0.0
    if 'modelo_4d_dx'    not in st.session_state: st.session_state.modelo_4d_dx    = 0.0
    if 'modelo_4d_dy'    not in st.session_state: st.session_state.modelo_4d_dy    = 0.0
    if 'modelo_4d_dz'    not in st.session_state: st.session_state.modelo_4d_dz    = 0.0

    # Helper local: aplicar alpha/beta + traslación al modelo
    def _aplicar_pose_modelo_4d(obj_base, alpha_deg, beta_deg, dx, dy, dz, cg):
        """Retorna (x, y, z) del modelo rotado en torno al CG y luego trasladado.
        Convención:
          Alpha (+) = cabeceo → nariz sube  → rotación +Y en rotate_points
          Beta  (+) = guiñada → nariz hacia ala derecha (+Y) → rotación -Z en rotate_points
        """
        x = np.array(obj_base['x'], dtype=float) - cg['x']
        y = np.array(obj_base['y'], dtype=float) - cg['y']
        z = np.array(obj_base['z'], dtype=float) - cg['z']
        # Rotar: angle_x=0, angle_y=alpha, angle_z=-beta
        x, y, z = rotate_points(x, y, z, 0.0, float(alpha_deg), float(-beta_deg))
        x = x + cg['x'] + dx
        y = y + cg['y'] + dy
        z = z + cg['z'] + dz
        return x, y, z

    # Cargar superficies 4D del usuario
    try:
        mis_superficies = auth.get_user_surfaces_4d(st.session_state.username)
    except AttributeError:
        st.error("Error conectando con base de datos (función get_user_surfaces_4d no encontrada).")
        mis_superficies = []

    if not mis_superficies:
        st.info("⚠️ No tienes planos 4D guardados. Usá el Paso 1 para procesar y guardar planos.")
    else:
        dict_superficies = {f"{s[1]} (X={s[2]}) [{s[3]}]": s for s in mis_superficies}

        # ── COLUMNAS PRINCIPALES ──────────────────────────────────────────
        c4d_left, c4d_right = st.columns([1.1, 2.5])

        with c4d_left:
            # ── VARIABLE Y MODO DE SELECCIÓN ─────────────────────────────
            opciones_var_4d = ["Presión Total [Actual]", " ρ_∞", "V_∞", "P_∞"]
            var_4d_sel = st.selectbox("📊 Variable a visualizar:", opciones_var_4d, key="var_4d_sel_ui")

            modo_sel_4d = st.radio(
                "Modo de selección de planos:",
                ["✅ Individual", " Por Posición X", "🎯 Por AOA"],
                horizontal=True,
                key="modo_sel_4d"
            )

            # Helper AOA
            def _extraer_aoa_4d(nombre):
                m = re.search(r'OAO(neg)?(\d+(?:[.,]\d+)?)', str(nombre), re.IGNORECASE)
                if m:
                    return (-1 if m.group(1) else 1) * float(str(m.group(2)).replace(',', '.'))
                return None

            if modo_sel_4d == "✅ Individual":
                sel_labels = st.multiselect("Seleccionar planos:", list(dict_superficies.keys()), key="sel_4d_main")

            elif modo_sel_4d == " Por Posición X":
                # Agrupar por X
                x_positions = sorted(set(s[2] for s in mis_superficies))
                x_sel = st.multiselect("Posiciones X [mm]:", x_positions, default=x_positions, key="sel_x_4d")
                sel_labels = [lbl for lbl, s in dict_superficies.items() if s[2] in x_sel]
                st.caption(f"📊 {len(sel_labels)} planos en {len(x_sel)} posiciones X")

            else:  # Por AOA
                all_aoas_4d = sorted(set(
                    _extraer_aoa_4d(s[1]) for s in mis_superficies
                    if _extraer_aoa_4d(s[1]) is not None
                ))
                if not all_aoas_4d:
                    st.warning("No se detectaron AOAs en los nombres (formato esperado: OAO{N} o OAOneg{N})")
                    sel_labels = []
                else:
                    aoas_sel_4d = st.multiselect(
                        "AOAs [°]:", [f"{a}°" for a in all_aoas_4d],
                        default=[f"{a}°" for a in all_aoas_4d],
                        key="sel_aoa_4d"
                    )
                    aoas_num_4d = [float(a.replace('°','')) for a in aoas_sel_4d]
                    sel_labels = [lbl for lbl, s in dict_superficies.items()
                                  if _extraer_aoa_4d(s[1]) in aoas_num_4d]
                    st.caption(f"📊 {len(sel_labels)} planos seleccionados")

            st.markdown("---")

            # ── VISUALIZACIÓN Y ESCALA ──────────────────────────────────────
            st.markdown("##### 🎨 Opciones de Visualización")
            c_vis1, c_vis2, c_vis3 = st.columns(3)
            vis_modelo = c_vis1.selectbox("Modelo 3D:", ["Azul Translúcido", "Negro Mate", "Plata Metalizada", "Puntos"], index=0, key="vis_mod_4d")
            vis_bg = c_vis2.selectbox("Fondo:", ["Oscuro (Negro)", "Claro (Blanco)"], index=0, key="vis_bg_4d")
            vis_ejes = c_vis3.checkbox("Mostrar Ejes 3D", value=True, key="vis_ejes_4d")
            
            pressure_scale = st.slider("Escala de Relieve (Presión→X):", 0.1, 10.0, 1.0, 0.1, key="scale_4d_viz")

            # ── POSICIONAMIENTO DEL MODELO 3D ─────────────────────────────
            if 'objeto_referencia_4d' in st.session_state:
                st.markdown("---")
                st.markdown("""
                <div style="background:#0d1f35; border:1px solid #1e4060; border-radius:8px; padding:12px; margin-bottom:8px;">
                    <h5 style="color:#60a5fa; margin:0 0 6px 0;">✈️ Posición y Actitud del Modelo</h5>
                    <p style="color:#93c5fd; font-size:0.78rem; margin:0; line-height:1.6;">
                        <b>Alpha α (+)</b> → nariz sube (plano XZ)<br>
                        <b>Beta β (+)</b> → nariz hacia ala derecha (plano XY)<br>
                        Rotación en torno al CG definido en Modelos.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                c_ab1, c_ab2 = st.columns(2)
                st.session_state.modelo_4d_alpha = c_ab1.number_input(
                    "α Alpha [°]", value=float(st.session_state.modelo_4d_alpha),
                    min_value=-90.0, max_value=90.0, step=1.0, format="%.1f", key="inp_alpha_4d"
                )
                st.session_state.modelo_4d_beta = c_ab2.number_input(
                    "β Beta [°]", value=float(st.session_state.modelo_4d_beta),
                    min_value=-90.0, max_value=90.0, step=1.0, format="%.1f", key="inp_beta_4d"
                )

                st.markdown("**Traslación [mm]:**")
                c_t1, c_t2, c_t3 = st.columns(3)
                st.session_state.modelo_4d_dx = c_t1.number_input("dX", value=float(st.session_state.modelo_4d_dx), step=10.0, format="%.1f", key="inp_dx_4d")
                st.session_state.modelo_4d_dy = c_t2.number_input("dY", value=float(st.session_state.modelo_4d_dy), step=10.0, format="%.1f", key="inp_dy_4d")
                st.session_state.modelo_4d_dz = c_t3.number_input("dZ", value=float(st.session_state.modelo_4d_dz), step=10.0, format="%.1f", key="inp_dz_4d")

                if st.button("🔄 Resetear posición del modelo", key="btn_reset_pose_4d", use_container_width=True):
                    st.session_state.modelo_4d_alpha = 0.0
                    st.session_state.modelo_4d_beta  = 0.0
                    st.session_state.modelo_4d_dx    = 0.0
                    st.session_state.modelo_4d_dy    = 0.0
                    st.session_state.modelo_4d_dz    = 0.0
                    st.rerun()

        with c4d_right:
            if sel_labels:
                # Cargar datos
                loaded_dfs = {}
                all_pressures = []
                for label in sel_labels:
                    s_data = dict_superficies[label]
                    try:
                        df = pd.DataFrame(json.loads(s_data[4]))
                        df['Presion'] = calcular_variable_atmosferica(df, var_4d_sel)
                        loaded_dfs[label] = df
                        if 'Presion' in df.columns:
                            all_pressures.extend(df['Presion'].dropna().tolist())
                    except Exception as e:
                        st.warning(f"Error cargando {label}: {e}")

                g_min, g_max = (min(all_pressures), max(all_pressures)) if all_pressures else (0, 1)

                if st.button("🚀 Generar Escena 4D", key="btn_render_4d", type="primary", use_container_width=True):
                    fig_4d = go.Figure()

                    # Modelo 3D con Alpha/Beta aplicados
                    if 'objeto_referencia_4d' in st.session_state:
                        obj_b = st.session_state.objeto_referencia_base if 'objeto_referencia_base' in st.session_state else st.session_state.objeto_referencia_4d
                        cg_4d = st.session_state.get('modelo_cg', {'x': 0.0, 'y': 0.0, 'z': 0.0})
                        xm, ym, zm = _aplicar_pose_modelo_4d(
                            obj_b,
                            st.session_state.modelo_4d_alpha,
                            st.session_state.modelo_4d_beta,
                            st.session_state.modelo_4d_dx,
                            st.session_state.modelo_4d_dy,
                            st.session_state.modelo_4d_dz,
                            cg_4d
                        )
                        obj = st.session_state.objeto_referencia_4d
                        if vis_modelo == "Puntos" or obj['type'] == 'scatter':
                            fig_4d.add_trace(go.Scatter3d(
                                x=xm, y=ym, z=zm,
                                mode='markers', marker=dict(size=2, color='#888', opacity=0.5),
                                name=obj.get('name', 'Modelo')
                            ))
                        else:
                            if vis_modelo == "Negro Mate":
                                color_m, opac = '#222222', 1.0
                                lighting = dict(ambient=0.3, diffuse=0.5, specular=0.1, roughness=0.9)
                            elif vis_modelo == "Plata Metalizada":
                                color_m, opac = '#e0e0e0', 1.0
                                lighting = dict(ambient=0.4, diffuse=0.8, specular=1.0, roughness=0.1)
                            else: # Azul Translúcido
                                color_m, opac = '#5588cc', 0.3
                                lighting = dict(ambient=0.4, diffuse=0.8)
                                
                            fig_4d.add_trace(go.Mesh3d(
                                x=xm, y=ym, z=zm,
                                i=obj['i'], j=obj['j'], k=obj['k'],
                                color=color_m, opacity=opac, name=obj.get('name','Modelo'),
                                alphahull=0, showscale=False,
                                lighting=lighting
                            ))

                    # Planos de presión
                    for label in sel_labels:
                        df = loaded_dfs.get(label)
                        if df is None: continue
                        s_data = dict_superficies[label]
                        x_base = s_data[2]
                        try:
                            df_clean = df.dropna(subset=['Y', 'Z', 'Presion']).drop_duplicates(subset=['Y', 'Z'])
                            if len(df_clean) < 3: continue
                            tri = Delaunay(df_clean[['Y', 'Z']].values)
                            p_ref = df_clean['Presion'].max()
                            x_def = x_base - ((df_clean['Presion'] - p_ref) * pressure_scale)
                            fig_4d.add_trace(go.Mesh3d(
                                x=x_def, y=df_clean['Y'], z=df_clean['Z'],
                                i=tri.simplices[:,0], j=tri.simplices[:,1], k=tri.simplices[:,2],
                                intensity=df_clean['Presion'],
                                customdata=np.column_stack([np.full(len(df_clean), x_base), df_clean['Presion']]),
                                hovertemplate="X: %{customdata[0]:.1f} mm<br>Y: %{y:.1f} mm<br>Z: %{z:.1f} mm<br>Val: %{customdata[1]:.2f}<extra></extra>",
                                colorscale='Jet', cmin=g_min, cmax=g_max,
                                showscale=True, opacity=1.0, flatshading=False,
                                name=f"{s_data[1]} (X={x_base})"
                            ))
                        except Exception as e:
                            st.warning(f"Error ploteando {s_data[1]}: {e}")

                    bg_color = '#0e1117' if "Oscuro" in vis_bg else '#ffffff'
                    font_color = 'white' if "Oscuro" in vis_bg else 'black'
                    
                    axis_props = dict(
                        showgrid=vis_ejes, zeroline=vis_ejes, showticklabels=vis_ejes,
                        showaxeslabels=vis_ejes, showbackground=False
                    )

                    fig_4d.update_layout(
                        title=f"Escena 4D — α={st.session_state.modelo_4d_alpha:.1f}° β={st.session_state.modelo_4d_beta:.1f}°",
                        scene=dict(
                            aspectmode='data',
                            xaxis=dict(title="X (Estación)" if vis_ejes else "", autorange="reversed", **axis_props),
                            yaxis=dict(title="Y (Envergadura)" if vis_ejes else "", **axis_props),
                            zaxis=dict(title="Z (Altura)" if vis_ejes else "", **axis_props)
                        ),
                        paper_bgcolor=bg_color,
                        plot_bgcolor=bg_color,
                        font=dict(color=font_color),
                        height=750,
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    st.session_state['fig_4d_cache'] = fig_4d

                # Mostrar figura cacheada
                if 'fig_4d_cache' in st.session_state:
                    st.plotly_chart(st.session_state['fig_4d_cache'], use_container_width=True)
            else:
                st.info("Seleccioná al menos un plano para visualizar.")



elif st.session_state.seccion_actual == 'animacion_4d':
    st.markdown("""
    <div class="header-container">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🎬 ANIMACIÓN 4D
        </h1>
        <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Interpolación de planos de presión y cabeceo del modelo geométrico
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # ── INIT SESSION STATE ────────────────────────────────────────────────────
    if 'anim4d_grillas' not in st.session_state: st.session_state.anim4d_grillas = None
    if 'anim4d_aoa_range' not in st.session_state: st.session_state.anim4d_aoa_range = None
    if 'anim4d_x_range' not in st.session_state: st.session_state.anim4d_x_range = None
    if 'anim4d_pmin' not in st.session_state: st.session_state.anim4d_pmin = 0.0
    if 'anim4d_pmax' not in st.session_state: st.session_state.anim4d_pmax = 1.0
    # Caché de frames: lista de bytes PNG ya renderizados
    if 'anim4d_frames_cache' not in st.session_state: st.session_state.anim4d_frames_cache = []
    # Metadata de la última sesión renderizada
    if 'anim4d_session_meta' not in st.session_state: st.session_state.anim4d_session_meta = {}

    # Helper: extraer AOA del nombre
    def _aoa_from_name_anim(nombre):
        m = re.search(r'OAO(neg)?(\d+(?:[.,]\d+)?)', str(nombre), re.IGNORECASE)
        if m:
            return (-1 if m.group(1) else 1) * float(str(m.group(2)).replace(',', '.'))
        return None

    # Helper: aplicar pose
    def _pose_anim(obj_base, alpha_deg, beta_deg, cg):
        x = np.array(obj_base['x'], dtype=float) - cg['x']
        y = np.array(obj_base['y'], dtype=float) - cg['y']
        z = np.array(obj_base['z'], dtype=float) - cg['z']
        x, y, z = rotate_points(x, y, z, 0.0, float(alpha_deg), float(-beta_deg))
        return x + cg['x'], y + cg['y'], z + cg['z']

    # Cargar planos 4D
    try:
        mis_superficies_anim = auth.get_user_surfaces_4d(st.session_state.username)
    except AttributeError:
        st.error("Error conectando con base de datos (función get_user_surfaces_4d no encontrada).")
        mis_superficies_anim = []

    if not mis_superficies_anim:
        st.info("⚠️ No hay planos 4D guardados. Ve a **Vis. Estela 4D → Paso 1** para guardar planos primero.")
    else:
        dict_sup_anim = {f"{s[1]} (X={s[2]}mm) [{s[3][:10] if s[3] else ''}]": s for s in mis_superficies_anim}

        # ── PASO 1: SELECCIÓN DE PLANOS ──────────────────────────────────────
        c_sel_left, c_sel_right = st.columns([1.2, 2])

        with c_sel_left:
            st.markdown("### 📂 Paso 1: Selección de Planos")

            opciones_var_anim4d = ["Presión Total [Actual]", " ρ_∞", "V_∞", "P_∞"]
            var_anim_sel = st.selectbox("📊 Variable a visualizar:", opciones_var_anim4d, key="var_anim4d_sel")

            modo_fil_anim = st.radio(
                "Filtrar por:",
                ["✅ Individual", " Por Plano (X)", "🎯 Por AOA"],
                key="modo_fil_anim4d",
                horizontal=True
            )

            if modo_fil_anim == "✅ Individual":
                sel_anim_labels = st.multiselect("Seleccionar planos:", list(dict_sup_anim.keys()), key="sel_anim_ind")

            elif modo_fil_anim == " Por Plano (X)":
                x_positions_anim = sorted(set(s[2] for s in mis_superficies_anim))
                x_sel_anim = st.multiselect("Posiciones X [mm]:", x_positions_anim, default=x_positions_anim, key="sel_x_anim")
                sel_anim_labels = [lbl for lbl, s in dict_sup_anim.items() if s[2] in x_sel_anim]
                st.caption(f"📊 {len(sel_anim_labels)} planos en {len(x_sel_anim)} posiciones X")

            else:  # Por AOA
                all_aoas_anim = sorted(set(
                    _aoa_from_name_anim(s[1]) for s in mis_superficies_anim
                    if _aoa_from_name_anim(s[1]) is not None
                ))
                if not all_aoas_anim:
                    st.warning("⚠️ No se detectaron AOAs en los nombres (formato: OAO{N} o OAOneg{N})")
                    sel_anim_labels = []
                else:
                    aoas_sel_anim = st.multiselect(
                        "Seleccionar AOAs [°]:", [f"{a}°" for a in all_aoas_anim],
                        default=[f"{a}°" for a in all_aoas_anim], key="sel_aoas_anim4d"
                    )
                    aoas_num_anim = [float(a.replace('°', '')) for a in aoas_sel_anim]
                    sel_anim_labels = [lbl for lbl, s in dict_sup_anim.items()
                                       if _aoa_from_name_anim(s[1]) in aoas_num_anim]
                    st.caption(f"📊 {len(sel_anim_labels)} planos seleccionados")

            pressure_scale_anim = st.slider("Escala de Relieve [presión→X]:", 0.1, 10.0, 1.0, 0.1, key="scale_anim_interp")
            mostrar_modelo_anim = st.checkbox("Mostrar modelo 3D", value=True, key="show_model_anim")

        with c_sel_right:
            if sel_anim_labels:
                st.markdown("### 🔢 Paso 2: Pre-computar Interpolación")
                st.info("Computá la grilla una vez. Luego el slider moverá el gráfico al instante sin recalcular.")

                if st.button("⚡ Pre-computar interpolación", type="primary", use_container_width=True, key="btn_precompute"):
                    from scipy.interpolate import griddata as _gd_anim

                    # Cargar todos los planos seleccionados
                    items_precomp = []
                    all_y_pc, all_z_pc = [], []
                    all_p_pc = []

                    prog_pc = st.progress(0)
                    status_pc = st.empty()

                    for fi, lbl in enumerate(sel_anim_labels):
                        s_data = dict_sup_anim[lbl]
                        try:
                            df_tmp = pd.DataFrame(json.loads(s_data[4]))
                            df_tmp['Presion'] = calcular_variable_atmosferica(df_tmp, var_anim_sel)
                            df_c = df_tmp.dropna(subset=['Y', 'Z', 'Presion'])
                            aoa_v = _aoa_from_name_anim(s_data[1])
                            items_precomp.append({
                                'aoa': aoa_v if aoa_v is not None else 0.0,
                                'x': float(s_data[2]),
                                'df': df_c,
                                'name': s_data[1]
                            })
                            all_y_pc.extend(df_c['Y'].tolist())
                            all_z_pc.extend(df_c['Z'].tolist())
                            all_p_pc.extend(df_c['Presion'].tolist())
                        except Exception as e:
                            st.warning(f"Error cargando {lbl}: {e}")
                        prog_pc.progress((fi + 1) / len(sel_anim_labels))

                    if items_precomp:
                        # Ordenar por AOA
                        items_precomp.sort(key=lambda d: d['aoa'])
                        aoa_arr_pc = np.array([d['aoa'] for d in items_precomp])
                        x_arr_pc   = np.array([d['x']   for d in items_precomp])

                        # --- Estrategia de grilla: puntos REALES del plano ---
                        # Usamos la unión de todos los puntos YZ como dominio de interpolación.
                        # Esto preserva todos los puntos de medición reales sin perder resolución.
                        all_yz_pts = np.column_stack([all_y_pc, all_z_pc])
                        # Eliminar duplicados para la grilla base
                        _, uniq_idx = np.unique(np.round(all_yz_pts, 4), axis=0, return_index=True)
                        Y_base = np.array(all_y_pc)[uniq_idx]
                        Z_base = np.array(all_z_pc)[uniq_idx]

                        # Interpolar cada plano a la base de puntos reales
                        grillas_pc = []
                        status_pc.text("Interpolando planos a la grilla de puntos reales...")
                        for ii, item in enumerate(items_precomp):
                            df_g = item['df']
                            P_g = _gd_anim(
                                (df_g['Y'].values, df_g['Z'].values), df_g['Presion'].values,
                                (Y_base, Z_base), method='linear'
                            )
                            # Fallback NaN: completar con nearest neighbor
                            nan_mask = np.isnan(P_g)
                            if nan_mask.any():
                                P_nn = _gd_anim(
                                    (df_g['Y'].values, df_g['Z'].values), df_g['Presion'].values,
                                    (Y_base[nan_mask], Z_base[nan_mask]), method='nearest'
                                )
                                P_g[nan_mask] = P_nn
                            grillas_pc.append(P_g)
                            prog_pc.progress((ii + 1) / len(items_precomp))

                        # Guardar en session_state
                        st.session_state.anim4d_grillas = {
                            'aoa_arr': aoa_arr_pc,
                            'x_arr': x_arr_pc,
                            'Y': Y_base,   # 1D array de puntos Y reales
                            'Z': Z_base,   # 1D array de puntos Z reales
                            'grillas': grillas_pc,  # cada grilla es 1D array de igual longitud
                            'items': items_precomp,
                            'var': var_anim_sel,
                            'modo': 'puntos_reales'
                        }
                        st.session_state.anim4d_pmin = float(np.nanmin(all_p_pc))
                        st.session_state.anim4d_pmax = float(np.nanmax(all_p_pc))
                        st.session_state.anim4d_aoa_range = (float(aoa_arr_pc.min()), float(aoa_arr_pc.max()))
                        status_pc.empty(); prog_pc.empty()
                        n_pts = len(Y_base)
                        st.success(f"✅ Pre-computación completada: {len(items_precomp)} planos | {n_pts:,} puntos reales | AOA {aoa_arr_pc.min():.1f}° → {aoa_arr_pc.max():.1f}°")
                        st.rerun()

        # ── PASO 3: VISUALIZACIÓN INTERACTIVA ────────────────────────────────
        if st.session_state.anim4d_grillas is not None:
            st.markdown("---")
            st.markdown("### 🎛️ Paso 3: Visualización Interactiva")

            g = st.session_state.anim4d_grillas
            aoa_min_v, aoa_max_v = st.session_state.anim4d_aoa_range
            pmin_v = st.session_state.anim4d_pmin
            pmax_v = st.session_state.anim4d_pmax

            c_ctrl_anim, c_plot_anim = st.columns([1, 2.5])

            with c_ctrl_anim:
                if aoa_min_v != aoa_max_v:
                    alpha_slider = st.slider(
                        f"α Alpha [°]  ({aoa_min_v:.1f}° → {aoa_max_v:.1f}°):",
                        min_value=float(aoa_min_v), max_value=float(aoa_max_v),
                        value=float(aoa_min_v), step=0.5, key="alpha_slider_anim4d"
                    )
                else:
                    alpha_slider = float(aoa_min_v)
                    st.info(f"Solo un AOA: {aoa_min_v}°")

                st.markdown("##### 🎨 Visualización")
                vis_modelo_a = st.selectbox("Modelo 3D:", ["Azul Translúcido", "Negro Mate", "Plata Metalizada", "Puntos"], index=0, key="vis_mod_anim")
                c_va1, c_va2 = st.columns(2)
                vis_bg_a = c_va1.selectbox("Fondo:", ["Oscuro (Negro)", "Claro (Blanco)"], index=0, key="vis_bg_anim")
                vis_ejes_a = c_va2.checkbox("Mostrar Ejes 3D", value=True, key="vis_ejes_anim")
                
                st.markdown("---")
                sc_anim = st.slider("Escala Relieve:", 0.1, 10.0, pressure_scale_anim, 0.1, key="sc_anim_live")

                # Info del plano interpolado
                aoa_arr = g['aoa_arr']
                idx_lo = max(0, min(int(np.searchsorted(aoa_arr, alpha_slider)) - 1, len(aoa_arr) - 2))
                idx_hi = idx_lo + 1
                t_v = (alpha_slider - aoa_arr[idx_lo]) / (aoa_arr[idx_hi] - aoa_arr[idx_lo]) if aoa_arr[idx_hi] != aoa_arr[idx_lo] else 0.0
                x_v = (1 - t_v) * g['x_arr'][idx_lo] + t_v * g['x_arr'][idx_hi]

                st.markdown(f"""
                <div style="background:#111; border:1px solid #333; border-radius:8px; padding:10px; margin-top:10px;">
                    <p style="color:#888; font-size:0.75rem; margin:0;">Interpolación activa</p>
                    <p style="color:white; font-size:0.9rem; margin:4px 0;">α = {alpha_slider:.1f}°</p>
                    <p style="color:#aaa; font-size:0.75rem; margin:0;">
                        Entre {g['items'][idx_lo]['name']}<br>
                        y {g['items'][idx_hi]['name']}<br>
                        t = {t_v:.2f} | X ≈ {x_v:.1f} mm
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if st.button("🗑️ Limpiar pre-computación", key="btn_clear_precomp", use_container_width=True):
                    st.session_state.anim4d_grillas = None
                    st.session_state.anim4d_aoa_range = None
                    st.rerun()

            with c_plot_anim:
                # Interpolar presión al vuelo (rápido, la grilla ya está lista)
                P_interp = (1 - t_v) * g['grillas'][idx_lo] + t_v * g['grillas'][idx_hi]
                # Y y Z son ahora arrays 1D de puntos reales
                Y_v = g['Y']   # 1D
                Z_v = g['Z']   # 1D
                mask_v = ~np.isnan(P_interp)

                fig_live = go.Figure()

                # Modelo 3D rotado
                if mostrar_modelo_anim and 'objeto_referencia_4d' in st.session_state:
                    obj_anim = st.session_state.objeto_referencia_base if 'objeto_referencia_base' in st.session_state else st.session_state.objeto_referencia_4d
                    cg_anim = st.session_state.get('modelo_cg', {'x': 0.0, 'y': 0.0, 'z': 0.0})
                    xm_a, ym_a, zm_a = _pose_anim(obj_anim, alpha_slider, 0.0, cg_anim)
                    obj_ref = st.session_state.objeto_referencia_4d
                    if vis_modelo_a == "Puntos" or obj_ref['type'] == 'scatter':
                        fig_live.add_trace(go.Scatter3d(
                            x=xm_a, y=ym_a, z=zm_a,
                            mode='markers', marker=dict(size=2, color='#888', opacity=0.5),
                            name="Modelo"
                        ))
                    else:
                        if vis_modelo_a == "Negro Mate":
                            color_m, opac = '#222222', 1.0
                            lighting = dict(ambient=0.3, diffuse=0.5, specular=0.1, roughness=0.9)
                        elif vis_modelo_a == "Plata Metalizada":
                            color_m, opac = '#e0e0e0', 1.0
                            lighting = dict(ambient=0.4, diffuse=0.8, specular=1.0, roughness=0.1)
                        else: # Azul Translúcido
                            color_m, opac = '#5588cc', 0.3
                            lighting = dict(ambient=0.4, diffuse=0.8)

                        fig_live.add_trace(go.Mesh3d(
                            x=xm_a, y=ym_a, z=zm_a,
                            i=obj_ref['i'], j=obj_ref['j'], k=obj_ref['k'],
                            color=color_m, opacity=opac, name="Modelo",
                            alphahull=0, showscale=False,
                            lighting=lighting
                        ))

                # Plano interpolado — con todos los puntos reales
                # Escala: pmin_v / pmax_v son el mín/máx GLOBAL de todos los planos seleccionados
                if mask_v.any():
                    Y_ok = Y_v[mask_v]
                    Z_ok = Z_v[mask_v]
                    P_ok = P_interp[mask_v]
                    P_max_ref = pmax_v # Usar máximo global para no desplazar el plano
                    X_def_v = x_v - ((P_ok - P_max_ref) * sc_anim)

                    try:
                        from scipy.spatial import Delaunay as _Del
                        pts_yz = np.column_stack([Y_ok, Z_ok])
                        tri_v = _Del(pts_yz)
                        fig_live.add_trace(go.Mesh3d(
                            x=X_def_v, y=Y_ok, z=Z_ok,
                            i=tri_v.simplices[:,0], j=tri_v.simplices[:,1], k=tri_v.simplices[:,2],
                            intensity=P_ok,
                            colorscale='Jet', cmin=pmin_v, cmax=pmax_v,
                            showscale=True, opacity=1.0, flatshading=True,
                            name=f"Presión (α={alpha_slider:.1f}°)",
                            colorbar=dict(title=dict(text=g.get('var','Presión'), side='right'))
                        ))
                    except Exception as e_tri:
                        fig_live.add_trace(go.Scatter3d(
                            x=X_def_v, y=Y_ok, z=Z_ok,
                            mode='markers',
                            marker=dict(size=3, color=P_ok, colorscale='Jet', showscale=True,
                                        cmin=pmin_v, cmax=pmax_v),
                            name=f"Presión (α={alpha_slider:.1f}°)"
                        ))
                bg_color_a = '#0e1117' if "Oscuro" in vis_bg_a else '#ffffff'
                font_color_a = 'white' if "Oscuro" in vis_bg_a else 'black'
                axis_props_a = dict(showgrid=vis_ejes_a, zeroline=vis_ejes_a, showticklabels=vis_ejes_a, showaxeslabels=vis_ejes_a, showbackground=False)

                fig_live.update_layout(
                    title=f"α = {alpha_slider:.1f}°",
                    scene=dict(
                        aspectmode='data',
                        xaxis=dict(title="X (Estación)" if vis_ejes_a else "", autorange="reversed", **axis_props_a),
                        yaxis=dict(title="Y (Envergadura)" if vis_ejes_a else "", **axis_props_a),
                        zaxis=dict(title="Z (Altura)" if vis_ejes_a else "", **axis_props_a)
                    ),
                    paper_bgcolor=bg_color_a,
                    plot_bgcolor=bg_color_a,
                    font=dict(color=font_color_a),
                    height=620,
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                st.plotly_chart(fig_live, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🎥 Paso 4: Generar Animación GIF")
            st.caption("💡 Matplotlib puro — sin Chrome ni kaleido. Dos modos: 2D suave (contourf) o 4D isométrico con modelo.")

            st.info("💡 **Nota de Rendimiento:** Generar el GIF a alta resolución puede ser muy lento en servidores online (Streamlit Cloud) comparado con una ejecución local, debido a las limitaciones de CPU y al pesado procesamiento gráfico. Si usas la app online, prueba resoluciones bajas y menos frames.")
            c_gif0, c_gif1, c_gif2, c_gif3 = st.columns(4)
            tipo_gif  = c_gif0.radio("Tipo:", ["🗺 2D suave", "🚀 4D"], index=1, key="tipo_gif_sel")
            fps_gif   = c_gif1.slider("FPS:", 1, 60, 18, key="fps_gif_anim")
            n_pas_gif = c_gif2.slider("N° frames:", 5, 120, 60, key="npasos_gif")
            sc_gif    = c_gif3.slider("× relieve (4D):", 0.1, 20.0, 8.1, 0.1, key="sc_gif_anim")
            
            c_opt1, c_opt2 = st.columns(2)
            dpi_gif   = c_opt1.selectbox("Resolución (DPI):", [110, 150, 200, 300, 600], index=4, key="dpi_gif_anim", help="Resolución de imagen. Valores altos mejoran enormemente la calidad, pero aumentan exponencialmente el tiempo de generación.")
            formato_anim = c_opt2.selectbox("Formato de Exportación:", ["GIF (Clásico, max 256 colores)", "MP4 (Alta Calidad True Color)"], index=1, key="formato_anim_sel")

            elev_gif, azim_gif = 25, -135
            if "4D" in tipo_gif:
                st.markdown("##### 📷 Posición de Cámara (Vista 4D)")
                c_cam1, c_cam2, c_cam3 = st.columns(3)
                preset_cam = c_cam1.selectbox("Preajustes:", ["Isométrica", "Opuesta a Isométrica", "Frente (aguas abajo)", "Lateral", "Planta", "Personalizada"], index=5, key="preset_cam_gif")
                if preset_cam == "Isométrica": elev_def, azim_def = 25, -135
                elif preset_cam == "Opuesta a Isométrica": elev_def, azim_def = 25, 45
                elif preset_cam == "Frente (aguas abajo)": elev_def, azim_def = 0, -180
                elif preset_cam == "Lateral": elev_def, azim_def = 0, -90
                elif preset_cam == "Planta": elev_def, azim_def = 90, -90
                else: elev_def, azim_def = 25, 45

                elev_gif = c_cam2.slider("Elevación [°]", -90, 90, elev_def, disabled=(preset_cam != "Personalizada"), key="elev_gif_anim")
                azim_gif = c_cam3.slider("Azimut [°]", -180, 180, azim_def, disabled=(preset_cam != "Personalizada"), key="azim_gif_anim")
                if preset_cam != "Personalizada": elev_gif, azim_gif = elev_def, azim_def

            c_btn1, c_btn2 = st.columns(2)
            btn_preview = c_btn1.button(" Previsualizar Vista (1 frame)", use_container_width=True)
            btn_generar = c_btn2.button("🎥 Generar GIF Completo", type="primary", use_container_width=True)

            if btn_preview or btn_generar:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as _plt
                import matplotlib.cm as _cm
                import matplotlib.colors as _mcolors
                from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa
                from scipy.interpolate import griddata as _gd_gif
                from scipy.spatial import Delaunay as _Del_gif
                from scipy.interpolate import LinearNDInterpolator as _LND

                alpha_range_gif = np.linspace(aoa_min_v, aoa_max_v, n_pas_gif)
                if btn_preview:
                    alpha_range_gif = [alpha_range_gif[0]] # Solo renderiza el primer frame para previsualizar

                status_gif = st.empty()
                prog_gif   = st.progress(0)
                frames_gif = []
                temp_dir_gif = tempfile.mkdtemp()
                norm_gif = _mcolors.Normalize(vmin=pmin_v, vmax=pmax_v)

                # Grilla regular densa para contourf (2D suave) o superficie suave (4D)
                N_GRID = 200
                y_lim = (float(g['Y'].min()), float(g['Y'].max()))
                z_lim = (float(g['Z'].min()), float(g['Z'].max()))
                y_reg = np.linspace(y_lim[0], y_lim[1], N_GRID)
                z_reg = np.linspace(z_lim[0], z_lim[1], N_GRID)
                Yr, Zr = np.meshgrid(y_reg, z_reg)

                # --- LÍMITES GLOBALES (Para fijar cámara 4D y evitar saltos) ---
                x_lim_min = float(np.min(g['x_arr'])) - ((pmax_v - pmin_v) * sc_gif)
                x_lim_max = float(np.max(g['x_arr']))
                y_lim_min, y_lim_max = y_lim
                z_lim_min, z_lim_max = z_lim

                if "4D" in tipo_gif and 'objeto_referencia_4d' in st.session_state:
                    obj_lim = st.session_state.get('objeto_referencia_base', st.session_state.objeto_referencia_4d)
                    cg_lim = st.session_state.get('modelo_cg', {'x': 0.0, 'y': 0.0, 'z': 0.0})
                    if 'x' in obj_lim and len(obj_lim['x']) > 0:
                        x_base = np.array(obj_lim['x']) - cg_lim['x']
                        y_base = np.array(obj_lim['y']) - cg_lim['y']
                        z_base = np.array(obj_lim['z']) - cg_lim['z']
                        
                        x_min_mod, x_max_mod = float('inf'), float('-inf')
                        y_min_mod, y_max_mod = float('inf'), float('-inf')
                        z_min_mod, z_max_mod = float('inf'), float('-inf')
                        
                        # Muestreamos los límites reales del modelo rotado en lugar de usar un radio esférico máximo
                        alphas_para_limites = np.linspace(aoa_min_v, aoa_max_v, n_pas_gif)
                        for a in alphas_para_limites:
                            xr, yr, zr = rotate_points(x_base, y_base, z_base, 0, a, 0)
                            x_min_mod = min(x_min_mod, float(np.min(xr)))
                            x_max_mod = max(x_max_mod, float(np.max(xr)))
                            y_min_mod = min(y_min_mod, float(np.min(yr)))
                            y_max_mod = max(y_max_mod, float(np.max(yr)))
                            z_min_mod = min(z_min_mod, float(np.min(zr)))
                            z_max_mod = max(z_max_mod, float(np.max(zr)))
                            
                        x_lim_min = min(x_lim_min, x_min_mod)
                        x_lim_max = max(x_lim_max, x_max_mod)
                        y_lim_min = min(y_lim_min, y_min_mod)
                        y_lim_max = max(y_lim_max, y_max_mod)
                        z_lim_min = min(z_lim_min, z_min_mod)
                        z_lim_max = max(z_lim_max, z_max_mod)

                try:
                    tri_interp = None
                    last_mask_sum = -1
                    for fi, alpha_i in enumerate(alpha_range_gif):
                        if not btn_preview:
                            status_gif.text(f"Frame {fi+1}/{n_pas_gif}  α={alpha_i:.1f}°")
                        else:
                            status_gif.text(f"Generando previsualización para α={alpha_i:.1f}°...")

                        idx_lo_g = max(0, min(int(np.searchsorted(g['aoa_arr'], alpha_i)) - 1, len(g['aoa_arr']) - 2))
                        idx_hi_g = idx_lo_g + 1
                        denom_t = g['aoa_arr'][idx_hi_g] - g['aoa_arr'][idx_lo_g]
                        t_g = (alpha_i - g['aoa_arr'][idx_lo_g]) / denom_t if denom_t != 0 else 0.0
                        P_g  = (1 - t_g) * g['grillas'][idx_lo_g] + t_g * g['grillas'][idx_hi_g]
                        x_g  = (1 - t_g) * g['x_arr'][idx_lo_g]  + t_g * g['x_arr'][idx_hi_g]
                        mask_g = ~np.isnan(P_g)

                        bg_color_mpl = '#0e1117' if "Oscuro" in vis_bg_a else '#ffffff'
                        text_color_mpl = 'white' if "Oscuro" in vis_bg_a else 'black'

                        if "2D" in tipo_gif:
                            # ── 2D: contourf suave sobre grilla densa ─────────────
                            fig_mpl, ax_mpl = _plt.subplots(figsize=(9, 7), facecolor=bg_color_mpl)
                            ax_mpl.set_facecolor(bg_color_mpl)

                            if mask_g.any():
                                # Interpolar puntos reales a grilla regular
                                # Optimización: reutilizar triangulación Delaunay para acelerar FPS
                                Y_ok_g, Z_ok_g, P_ok_g = g['Y'][mask_g], g['Z'][mask_g], P_g[mask_g]
                                current_mask_sum = mask_g.sum()
                                if current_mask_sum != last_mask_sum or tri_interp is None:
                                    tri_interp = _Del_gif(np.column_stack([Y_ok_g, Z_ok_g]))
                                    last_mask_sum = current_mask_sum
                                interp_func = _LND(tri_interp, P_ok_g)
                                Pr = interp_func(Yr, Zr)
                                # Relleno smooth con contourf
                                cf = ax_mpl.contourf(Yr, Zr, Pr, levels=40, cmap='jet', norm=norm_gif)
                                # Isolíneas encima
                                ax_mpl.contour(Yr, Zr, Pr, levels=12, colors='white', linewidths=0.4, alpha=0.4)
                                cb = fig_mpl.colorbar(cf, ax=ax_mpl, label=g.get('var', 'Presión [Pa]'))
                                cb.ax.yaxis.label.set_color(text_color_mpl)
                                cb.ax.tick_params(colors=text_color_mpl)

                            # Línea de simetría
                            y_mid_g = (y_lim[0] + y_lim[1]) / 2
                            ax_mpl.axvline(y_mid_g, color='cyan', lw=1.2, ls='--', alpha=0.7, label=f'Y_mid={y_mid_g:.0f}')
                            ax_mpl.set_xlim(y_lim); ax_mpl.set_ylim(z_lim)
                            ax_mpl.set_aspect('equal', 'box')
                            
                            ax_mpl.set_title(f"α = {alpha_i:.1f}°  |  Plano YZ — {g.get('var','Presión')}",
                                             color=text_color_mpl, fontsize=13, pad=10)
                                             
                            if not vis_ejes_a:
                                ax_mpl.axis('off')
                            else:
                                ax_mpl.set_xlabel("Y [mm]", color=text_color_mpl)
                                ax_mpl.set_ylabel("Z [mm]", color=text_color_mpl)
                                ax_mpl.tick_params(colors=text_color_mpl)
                                for sp in ax_mpl.spines.values(): sp.set_edgecolor('#444' if "Oscuro" in vis_bg_a else '#ccc')

                            ax_mpl.legend(fontsize=8, facecolor=bg_color_mpl, labelcolor=text_color_mpl,
                                          edgecolor='#444' if "Oscuro" in vis_bg_a else '#ccc', loc='upper right')

                        else:
                            # ── 4D con modelo ──────────────────────────
                            fig_mpl = _plt.figure(figsize=(11, 8), facecolor=bg_color_mpl)
                            ax3 = fig_mpl.add_subplot(111, projection='3d')
                            ax3.set_facecolor(bg_color_mpl)

                            # Plano de presión
                            if mask_g.any():
                                Y_ok_g = g['Y'][mask_g]; Z_ok_g = g['Z'][mask_g]; P_ok_g = P_g[mask_g]
                                # Interpolar a grilla regular para superficie suave
                                current_mask_sum = mask_g.sum()
                                if current_mask_sum != last_mask_sum or tri_interp is None:
                                    tri_interp = _Del_gif(np.column_stack([Y_ok_g, Z_ok_g]))
                                    last_mask_sum = current_mask_sum
                                interp_func = _LND(tri_interp, P_ok_g)
                                Pr3 = interp_func(Yr, Zr)
                                P_ref_g = pmax_v # Usar máximo global para no desplazar el plano
                                Xr3 = x_g - ((Pr3 - P_ref_g) * sc_gif)
                                # Renderizar con plot_surface para una superficie perfectamente lisa y sólida
                                facecolors_surf = _cm.get_cmap('jet')(norm_gif(Pr3))
                                ax3.plot_surface(Xr3, Yr, Zr, facecolors=facecolors_surf,
                                                 shade=False, alpha=0.9, antialiased=True, 
                                                 rstride=1, cstride=1, linewidth=0)

                            # Modelo 3D rotado
                            if 'objeto_referencia_4d' in st.session_state:
                                obj_b = st.session_state.get('objeto_referencia_base',
                                        st.session_state.objeto_referencia_4d)
                                cg_g2 = st.session_state.get('modelo_cg', {'x': 0.0, 'y': 0.0, 'z': 0.0})
                                xm_g2, ym_g2, zm_g2 = _pose_anim(obj_b, alpha_i, 0.0, cg_g2)
                                
                                if vis_modelo_a == "Puntos" or obj_b['type'] == 'scatter':
                                    c_mod, op_mod = '#888888', 0.5
                                    ax3.scatter(xm_g2, ym_g2, zm_g2, c=c_mod, s=1, alpha=op_mod, linewidths=0)
                                else:
                                    if vis_modelo_a == "Negro Mate":
                                        c_mod, op_mod = '#222222', 1.0
                                    elif vis_modelo_a == "Plata Metalizada":
                                        c_mod, op_mod = '#aaaaaa', 1.0
                                    else: # Azul Translúcido
                                        c_mod, op_mod = '#5588cc', 0.35
                                    
                                    try:
                                        triangles = np.column_stack([obj_b['i'], obj_b['j'], obj_b['k']])
                                        ax3.plot_trisurf(xm_g2, ym_g2, zm_g2, triangles=triangles,
                                                         color=c_mod, alpha=op_mod, shade=True, linewidth=0, antialiased=True)
                                    except Exception:
                                        ax3.scatter(xm_g2, ym_g2, zm_g2, c=c_mod, s=1, alpha=op_mod, linewidths=0)

                            if not vis_ejes_a:
                                ax3.set_axis_off()
                            else:
                                ax3.set_xlabel("X", color=text_color_mpl, fontsize=9)
                                ax3.set_ylabel("Y", color=text_color_mpl, fontsize=9)
                                ax3.set_zlabel("Z", color=text_color_mpl, fontsize=9)
                                ax3.tick_params(colors=text_color_mpl, labelsize=7)
                                ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
                                edge_c = '#333' if "Oscuro" in vis_bg_a else '#ddd'
                                ax3.xaxis.pane.set_edgecolor(edge_c); ax3.yaxis.pane.set_edgecolor(edge_c); ax3.zaxis.pane.set_edgecolor(edge_c)
                                
                            # Fijar límites estáticos para que el plano y la cámara no tiemblen en X
                            ax3.set_xlim(x_lim_min, x_lim_max)
                            ax3.set_ylim(y_lim_min, y_lim_max)
                            ax3.set_zlim(z_lim_min, z_lim_max)
                            try:
                                ax3.set_box_aspect((x_lim_max - x_lim_min, y_lim_max - y_lim_min, z_lim_max - z_lim_min))
                            except AttributeError:
                                pass # Fallback para versiones antiguas de matplotlib
                                
                            # Vista
                            ax3.view_init(elev=elev_gif, azim=azim_gif)
                            title_obj = ax3.set_title(f"α = {alpha_i:.1f}°  |  Vista 4D (Elev: {elev_gif}°, Azim: {azim_gif}°)",
                                          color=text_color_mpl, fontsize=12, pad=12)
                            
                            # Invertir eje X (avance del avión)
                            ax3.invert_xaxis()
                            fig_mpl.tight_layout()

                        fp_gif = os.path.join(temp_dir_gif, f"frame_{fi:03d}.png")
                        fig_mpl.savefig(fp_gif, dpi=dpi_gif, bbox_inches='tight', facecolor=bg_color_mpl)
                        _plt.close(fig_mpl)
                        frames_gif.append(fp_gif)
                        if not btn_preview:
                            prog_gif.progress((fi + 1) / n_pas_gif)

                    if btn_preview:
                        status_gif.empty()
                        st.image(frames_gif[0], caption="Previsualización del Frame 1")
                    else:
                        status_gif.text("Compilando Animación...")
                        ext = ".mp4" if "MP4" in formato_anim else ".gif"
                        gif_path_anim = os.path.join(temp_dir_gif, f"animacion_4d{ext}")
                        images_gif = [imageio.imread(f) for f in frames_gif]
                        
                        if ext == ".mp4":
                            imageio.mimsave(gif_path_anim, images_gif, fps=fps_gif, macro_block_size=2)
                        else:
                            imageio.mimsave(gif_path_anim, images_gif, fps=fps_gif, loop=0)
                        
                        # Guardar animación en sesión
                        with open(gif_path_anim, "rb") as fg:
                            st.session_state['ultimo_gif_anim4d'] = fg.read()
                        st.session_state['ultimo_gif_nombre'] = f"animacion_2d{ext}" if "2D" in tipo_gif else f"animacion_4d_iso{ext}"
                        st.session_state['ultimo_gif_detalles'] = f"✅ Animación generada: {n_pas_gif} frames · {fps_gif} FPS · {tipo_gif} · {ext.upper()}"
                        st.session_state['ultimo_gif_ext'] = ext
                        # Guardar frames en caché de sesión
                        frames_bytes_cache = []
                        for fp in frames_gif:
                            with open(fp, 'rb') as _fb: frames_bytes_cache.append(_fb.read())
                        st.session_state['anim4d_frames_cache'] = frames_bytes_cache
                        st.session_state['anim4d_session_meta'] = {
                            'tipo': tipo_gif, 'n_frames': n_pas_gif, 'fps': fps_gif,
                            'dpi': dpi_gif, 'aoa_min': aoa_min_v, 'aoa_max': aoa_max_v,
                            'variable': g.get('var', ''), 'sc_gif': sc_gif,
                            'planos': [d['name'] for d in g.get('items', [])],
                            'fecha': datetime.now().isoformat()
                        }

                except Exception as e_gif:
                    st.error(f"Error generando GIF: {e_gif}")
                    import traceback; st.code(traceback.format_exc())
                finally:
                    status_gif.empty(); prog_gif.empty()

            # ── Fuera del botón: mostrar animación + UI de Drive ──────────────
            if 'ultimo_gif_anim4d' in st.session_state:
                st.success(st.session_state['ultimo_gif_detalles'])
                if st.session_state.get('ultimo_gif_ext') == '.mp4':
                    st.video(st.session_state['ultimo_gif_anim4d'], format="video/mp4", autoplay=True, loop=True)
                else:
                    st.image(st.session_state['ultimo_gif_anim4d'])
                st.download_button("📥 Descargar Animación", st.session_state['ultimo_gif_anim4d'],
                                   file_name=st.session_state['ultimo_gif_nombre'],
                                   mime="video/mp4" if st.session_state.get('ultimo_gif_ext') == '.mp4' else "image/gif",
                                   key="dl_gif_anim4d_persistent")

            # ── Guardar frames en Drive ────────────────────────────────────────
            if st.session_state.get('anim4d_frames_cache'):
                st.markdown("---")
                st.markdown("### 💾 Paso 5: Guardar Interpolación en Drive")
                st.info(f"📦 {len(st.session_state['anim4d_frames_cache'])} frames en caché. "
                        f"Guardálos en Drive para reutilizarlos sin regenerar.")
                c_save1, c_save2 = st.columns([2, 1])
                nombre_sesion_drive = c_save1.text_input(
                    "Nombre de la sesión (carpeta en Drive):",
                    value=f"Interp_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    key="nombre_sesion_anim_drive",
                    help="Será la carpeta: ANIMACION / [nombre] con todos los frames PNG + metadata."
                )
                if c_save2.button("☁️ Guardar en Drive", type="primary", use_container_width=True, key="btn_save_sesion_drive"):
                    if nombre_sesion_drive.strip():
                        with st.spinner(f"Subiendo {len(st.session_state['anim4d_frames_cache'])} frames a Drive..."):
                            try:
                                sid = auth.save_animation_session(
                                    st.session_state.username,
                                    nombre_sesion_drive.strip(),
                                    st.session_state['anim4d_frames_cache'],
                                    st.session_state.get('anim4d_session_meta', {})
                                )
                                if sid:
                                    st.success(f"✅ Sesión **'{nombre_sesion_drive}'** guardada en Drive "
                                               f"({len(st.session_state['anim4d_frames_cache'])} frames).")
                                else:
                                    st.error("Error al crear la carpeta en Drive.")
                            except Exception as _e_save:
                                st.error(f"Error: {_e_save}")
                    else:
                        st.warning("Ingresá un nombre para la sesión.")

            # ── Cargar frames desde Drive (recompilar sin regenerar) ───────────
            st.markdown("---")
            with st.expander("📂 Cargar Sesión de Frames Guardada (desde Drive)", expanded=False):
                st.caption("Cargá frames previamente guardados y recompilá el video con distintos FPS, formato o estética.")
                try:
                    sesiones_drive = auth.list_animation_sessions(st.session_state.username)
                except Exception:
                    sesiones_drive = []
                if not sesiones_drive:
                    st.info("No hay sesiones guardadas aún. Generá un GIF completo y guardalo en Drive (Paso 5).")
                else:
                    dict_ses = {f"{s[1]} ({s[2][:10] if s[2] else ''})": s for s in sesiones_drive}
                    sel_ses = st.selectbox("Seleccionar sesión:", list(dict_ses.keys()), key="sel_ses_drive")
                    c_load1, c_load2 = st.columns(2)
                    if c_load1.button("⬇️ Cargar frames en memoria", use_container_width=True, key="btn_load_ses"):
                        with st.spinner("Descargando frames desde Drive..."):
                            try:
                                frames_loaded, meta_loaded = auth.load_animation_session(dict_ses[sel_ses][0])
                                if frames_loaded:
                                    st.session_state['anim4d_frames_cache'] = frames_loaded
                                    st.session_state['anim4d_session_meta'] = meta_loaded
                                    st.success(f"✅ {len(frames_loaded)} frames cargados desde '{dict_ses[sel_ses][1]}'.")
                                    if meta_loaded:
                                        st.json(meta_loaded)
                                else:
                                    st.warning("La sesión está vacía o no tiene frames.")
                            except Exception as _e_load:
                                st.error(f"Error cargando sesión: {_e_load}")

                    # Recompilar desde frames cargados
                    if st.session_state.get('anim4d_frames_cache') and c_load2.button(
                        "🎬 Recompilar Video", type="primary", use_container_width=True, key="btn_recompile"
                    ):
                        cached = st.session_state['anim4d_frames_cache']
                        meta_c = st.session_state.get('anim4d_session_meta', {})
                        fps_rc   = st.slider("FPS (recompilación):", 1, 60, meta_c.get('fps', 18), key="fps_recomp")
                        fmt_rc   = st.selectbox("Formato:", ["GIF", "MP4"], index=1, key="fmt_recomp")
                        ext_rc   = ".mp4" if fmt_rc == "MP4" else ".gif"
                        with st.spinner("Compilando video desde frames en caché..."):
                            try:
                                import imageio as _iio
                                tmp_rc = tempfile.mkdtemp()
                                # Escribir frames a disco
                                paths_rc = []
                                for _idx, _fb in enumerate(cached):
                                    _fp = os.path.join(tmp_rc, f"frame_{_idx:04d}.png")
                                    with open(_fp, 'wb') as _fh: _fh.write(_fb)
                                    paths_rc.append(_fp)
                                imgs_rc = [_iio.imread(p) for p in paths_rc]
                                out_rc  = os.path.join(tmp_rc, f"recompilado{ext_rc}")
                                if ext_rc == ".mp4":
                                    _iio.mimsave(out_rc, imgs_rc, fps=fps_rc, macro_block_size=2)
                                else:
                                    _iio.mimsave(out_rc, imgs_rc, fps=fps_rc, loop=0)
                                with open(out_rc, 'rb') as _fg:
                                    vid_rc = _fg.read()
                                st.session_state['ultimo_gif_anim4d'] = vid_rc
                                st.session_state['ultimo_gif_ext'] = ext_rc
                                n_rc = len(cached)
                                st.session_state['ultimo_gif_nombre'] = f"recompilado{ext_rc}"
                                st.session_state['ultimo_gif_detalles'] = f"✅ Recompilado desde caché: {n_rc} frames · {fps_rc} FPS · {fmt_rc}"
                                st.rerun()
                            except Exception as _e_rc:
                                st.error(f"Error recompilando: {_e_rc}")


elif st.session_state.seccion_actual == 'interpolacion_aoa':
    st.title("📊 ANÁLISIS DE INTERPOLACIÖN")
    st.write("Determine la cantidad mínima de planos angulares ($\alpha$) para representar el fenómeno físico.")

    import drive_api as _dapi
    import json
    import io

    # Inicializar estados
    if 'results_interp' not in st.session_state: st.session_state.results_interp = None
    if 'interp_planos_memoria' not in st.session_state: st.session_state.interp_planos_memoria = []
    
    # ── PASO 0: SELECCIÓN DE ORIGEN DE DATOS ──────────────────────────────────
    st.markdown("### 📥 Paso 0: Cargar Planos para Análisis")
    
    modo_carga = st.radio(
        "Seleccionar origen de datos:",
        ["🗄️ Base de Datos (Planos Guardados)", "📂 Explorar Google Drive (Archivos CSV)"],
        horizontal=True, key="modo_carga_interp"
    )

    if modo_carga == "🗄️ Base de Datos (Planos Guardados)":
        try:
            mis_planos_db = auth.get_user_surfaces_4d(st.session_state.username)
        except:
            mis_planos_db = []
            
        if not mis_planos_db:
            st.info("No hay planos guardados en la base de datos 4D.")
        else:
            estaciones_db = sorted(list(set(p[2] for p in mis_planos_db)))
            x_est_db = st.selectbox("Seleccionar Estación (X) [mm]:", estaciones_db, key="x_db_interp")
            
            planos_db_filtrados = [p for p in mis_planos_db if p[2] == x_est_db]
            
            if st.button("📥 Cargar planos de la Base de Datos", use_container_width=True):
                nuevos_planos = []
                for p in planos_db_filtrados:
                    # Extraer AOA del nombre
                    m = re.search(r'OAO(neg)?(\d+(?:[.,]\d+)?)', str(p[1]), re.IGNORECASE)
                    aoa_val = ((-1 if m.group(1) else 1) * float(str(m.group(2)).replace(',', '.'))) if m else 0.0
                    nuevos_planos.append({'name': p[1], 'aoa': aoa_val, 'json': p[4], 'x': p[2]})
                
                st.session_state.interp_planos_memoria = nuevos_planos
                st.success(f"Cargados {len(nuevos_planos)} planos desde la base de datos.")

    else:  # Explorar Google Drive
        st.markdown("#### 📂 Navegador de Drive")
        if 'drive_interp_folder_id' not in st.session_state:
            st.session_state.drive_interp_folder_id = None
            st.session_state.drive_interp_path = []

        if st.session_state.drive_interp_folder_id is None:
            with st.spinner("Conectando con carpeta 4D en Drive..."):
                f4d_id = _dapi.get_folder_4d(st.session_state.username)
                if f4d_id:
                    st.session_state.drive_interp_folder_id = f4d_id
                    st.session_state.drive_interp_path = [(f4d_id, "📁 Carpeta 4D")]

        # Breadcrumbs
        if st.session_state.drive_interp_path:
            path_str = " / ".join([p[1] for p in st.session_state.drive_interp_path])
            st.caption(f"📍 {path_str}")
        
        # Botón volver
        if len(st.session_state.drive_interp_path) > 1:
            if st.button("⬅️ Subir nivel"):
                st.session_state.drive_interp_path.pop()
                st.session_state.drive_interp_folder_id = st.session_state.drive_interp_path[-1][0]
                st.rerun()

        # Listar carpetas y archivos
        if st.session_state.drive_interp_folder_id:
            items = _dapi.list_folder_contents(st.session_state.drive_interp_folder_id)
            if items:
                carpetas = [i for i in items if i['mimeType'] == 'application/vnd.google-apps.folder']
                archivos_csv = [i for i in items if i['name'].lower().endswith('.csv')]
                
                if carpetas:
                    c_sel = st.selectbox("📁 Carpetas:", ["-- Seleccionar carpeta --"] + [f"{c['name']}" for c in carpetas])
                    if c_sel != "-- Seleccionar carpeta --":
                        c_id = next(c['id'] for c in carpetas if c['name'] == c_sel)
                        st.session_state.drive_interp_path.append((c_id, c_sel))
                        st.session_state.drive_interp_folder_id = c_id
                        st.rerun()

                if archivos_csv:
                    st.markdown("---")
                    sel_files_drv = st.multiselect("📄 Seleccionar archivos CSV para procesar:", [a['name'] for a in archivos_csv])
                    
                    if sel_files_drv:
                        config_interp = mostrar_configuracion_sensores("interp_drv")
                        x_pos_drv = st.number_input("Posición X (Estación) para estos archivos [mm]:", value=0.0)
                        
                        if st.button("⚙️ Procesar e Importar al Análisis", type="primary", use_container_width=True):
                            nuevos_planos_drv = []
                            prog_drv = st.progress(0)
                            for i, f_name in enumerate(sel_files_drv):
                                f_id = next(a['id'] for a in archivos_csv if a['name'] == f_name)
                                content = _dapi.download_file(f_id)
                                if content:
                                    try:
                                        df_p = procesar_promedios(io.BytesIO(content), config_interp.get('orden','asc'))
                                        if df_p is not None:
                                            m = re.search(r'OAO(neg)?(\d+(?:[.,]\d+)?)', f_name, re.IGNORECASE)
                                            aoa_v = ((-1 if m.group(1) else 1) * float(str(m.group(2)).replace(',', '.'))) if m else 0.0
                                            
                                            res_p = []
                                            for _, row in df_p.iterrows():
                                                y_t = row.get('Pos_Y_Traverser')
                                                z_b = row.get('Pos_Z_Base')
                                                for col in df_p.columns:
                                                    ns = obtener_numero_sensor_desde_columna(col)
                                                    if ns is not None:
                                                        vp = row[col]
                                                        if pd.isna(vp): continue
                                                        zr = calcular_altura_absoluta_z(ns, z_b, config_interp.get('distancia_toma_12',-120), config_interp.get('distancia_entre_tomas',10.91), 12, config_interp.get('orden','asc'))
                                                        res_p.append({'Y': y_t, 'Z': zr, 'Presion': vp})
                                            
                                            df_final = pd.DataFrame(res_p)
                                            nuevos_planos_drv.append({
                                                'name': f_name, 'aoa': aoa_v, 'x': x_pos_drv,
                                                'json': df_final.to_json(orient='records')
                                            })
                                    except Exception as e:
                                        st.error(f"Error procesando {f_name}: {e}")
                                prog_drv.progress((i+1)/len(sel_files_drv))
                            
                            st.session_state.interp_planos_memoria.extend(nuevos_planos_drv)
                            st.success(f"Importados {len(nuevos_planos_drv)} planos desde Drive.")
            else:
                st.write("Carpeta vacía.")

    # ── PASO 1: GESTIÓN DE PLANOS EN MEMORIA ──────────────────────────────────
    if st.session_state.interp_planos_memoria:
        st.markdown("---")
        st.markdown("### 📋 Planos cargados para el análisis")
        df_mem = pd.DataFrame([{'Nombre': p['name'], 'AOA [°]': p['aoa'], 'X [mm]': p['x']} for p in st.session_state.interp_planos_memoria])
        st.dataframe(df_mem, use_container_width=True)
        
        if st.button("🗑️ Limpiar todos los planos", type="secondary"):
            st.session_state.interp_planos_memoria = []
            st.session_state.results_interp = None
            st.rerun()

        # ── PASO 2: ANÁLISIS DE CONVERGENCIA ──────────────────────────────────
        st.markdown("---")
        st.markdown("### ⚙️ Paso 2: Análisis de Convergencia y Predicción de Ángulos")

        planos_ready = sorted(st.session_state.interp_planos_memoria, key=lambda x: x['aoa'])
        if len(planos_ready) < 3:
            st.warning("Se necesitan al menos 3 planos para realizar el análisis.")
        else:
            confianza_t = st.slider("Exactitud Objetivo [%]:", 50, 99, 90, key="slider_conf_interp")

            if st.button("🚀 Iniciar Análisis", type="primary", use_container_width=True):
                with st.spinner("Calculando convergencia de planos angulares..."):
                    angulos = [p['aoa'] for p in planos_ready]
                    grillas_full = []
                    for p in planos_ready:
                        df_tmp = pd.read_json(io.StringIO(p['json']))
                        y_lin = np.linspace(df_tmp['Y'].min(), df_tmp['Y'].max(), 100)
                        z_lin = np.linspace(df_tmp['Z'].min(), df_tmp['Z'].max(), 100)
                        Ym, Zm = np.meshgrid(y_lin, z_lin)
                        g_std = griddata((df_tmp['Y'], df_tmp['Z']), df_tmp['Presion'], (Ym, Zm), method='linear')
                        grillas_full.append(g_std)

                    res_list = []
                    N = len(grillas_full)
                    for k in range(1, N):
                        combs = list(itertools.combinations(range(N), k))
                        if len(combs) > 100: combs = random.sample(combs, 100)
                        mejor_acc_k = 0
                        mejor_cfg_k = ""   # ← bug corregido (era mejor_config_k)
                        for comb in combs:
                            idx_k = sorted(list(comb))
                            acc_list = []
                            for i in range(N):
                                if i in idx_k: continue
                                lo = next((idx for idx in reversed(idx_k) if idx < i), None)
                                hi = next((idx for idx in idx_k if idx > i), None)
                                if lo is not None and hi is not None:
                                    t = (angulos[i] - angulos[lo]) / (angulos[hi] - angulos[lo])
                                    pred = (1-t)*grillas_full[lo] + t*grillas_full[hi]
                                elif lo is not None: pred = grillas_full[lo]
                                elif hi is not None: pred = grillas_full[hi]
                                else: pred = np.zeros_like(grillas_full[0])
                                mask = ~np.isnan(grillas_full[i]) & ~np.isnan(pred)
                                if np.any(mask):
                                    corr = np.corrcoef(grillas_full[i][mask], pred[mask])[0, 1]
                                    acc_list.append(max(0, corr))
                                else:
                                    acc_list.append(0)
                            acc_m = np.mean(acc_list) if acc_list else 1.0
                            if acc_m > mejor_acc_k:
                                mejor_acc_k = acc_m
                                mejor_cfg_k = ", ".join([f"{angulos[idx]}°" for idx in idx_k])
                        res_list.append({'Planos': k, 'Exactitud [%]': round(mejor_acc_k*100, 2), 'Mejor Configuración': mejor_cfg_k})

                    # ── Análisis por intervalos: error de interpolación entre pares de planos ──
                    gap_errors = []
                    for i in range(N - 1):
                        # Interpolar el punto medio entre planos i e i+1
                        aoa_mid = (angulos[i] + angulos[i+1]) / 2
                        pred_mid = 0.5 * grillas_full[i] + 0.5 * grillas_full[i+1]
                        # Error relativo: variación de presión entre ambos extremos del intervalo
                        diff = np.nanstd(grillas_full[i+1] - grillas_full[i])
                        gap_errors.append({
                            'Intervalo': f"{angulos[i]}° → {angulos[i+1]}°",
                            'α_inicio': angulos[i], 'α_fin': angulos[i+1],
                            'Δα [°]': round(angulos[i+1] - angulos[i], 2),
                            'Variación (σ ΔP)': round(float(diff), 4)
                        })

                    st.session_state.results_interp = {
                        'tabla': pd.DataFrame(res_list),
                        'angulos': angulos,
                        'gap_errors': pd.DataFrame(gap_errors),
                        'N': N
                    }
                    st.rerun()

    # ── Mostrar Resultados ────────────────────────────────────────────────────
    if st.session_state.results_interp is not None:
        res = st.session_state.results_interp
        # Compatibilidad con versión antigua (era un DataFrame directo)
        if isinstance(res, pd.DataFrame):
            df_r = res
            gap_df = None
            angulos_res = []
        else:
            df_r = res['tabla']
            gap_df = res.get('gap_errors')
            angulos_res = res.get('angulos', [])

        st.markdown("---")
        st.subheader("📈 Curva de Convergencia")

        # Gráfico de convergencia
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=df_r['Planos'], y=df_r['Exactitud [%]'],
            mode='lines+markers', name='Exactitud',
            line=dict(color='#60a5fa', width=2),
            marker=dict(size=8)
        ))
        fig_conv.add_hline(y=confianza_t, line_dash="dash", line_color="#f59e0b",
                           annotation_text=f"Objetivo: {confianza_t}%")
        fig_conv.update_layout(
            title="Exactitud vs N° de Planos Utilizados",
            xaxis_title="N° de Planos (base de la interpolación)",
            yaxis_title="Exactitud [%]",
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
            font=dict(color='white'), height=350
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        # Resultado óptimo
        opt = df_r[df_r['Exactitud [%]'] >= confianza_t].head(1)
        if not opt.empty:
            st.success(
                f"🎯 **RESULTADO ÓPTIMO:** Con **{opt.iloc[0]['Planos']} planos** "
                f"(configuración: {opt.iloc[0]['Mejor Configuración']}) "
                f"se alcanza el **{opt.iloc[0]['Exactitud [%]']}%** de exactitud."
            )
        else:
            max_row = df_r.loc[df_r['Exactitud [%]'].idxmax()]
            st.warning(
                f"⚠️ Con los planos actuales el máximo alcanzable es **{max_row['Exactitud [%]']}%** "
                f"usando {max_row['Planos']} planos. Se necesitan más ángulos de medición."
            )

        st.dataframe(df_r, use_container_width=True)

        # ── Análisis de intervalos: dónde falta densidad ─────────────────────
        if gap_df is not None and not gap_df.empty:
            st.markdown("---")
            st.subheader("🔍 Análisis de Intervalos: ¿Dónde hay más cambio del fenómeno?")
            st.caption(
                "La **Variación (σ ΔP)** mide cuánto cambia el campo de presiones entre cada par de planos. "
                "Intervalos con mayor variación son los que más necesitan planos intermedios."
            )

            fig_gaps = go.Figure()
            fig_gaps.add_trace(go.Bar(
                x=gap_df['Intervalo'], y=gap_df['Variación (σ ΔP)'],
                marker_color='#f87171', name='Variación entre planos'
            ))
            fig_gaps.update_layout(
                title="Variación del campo de presiones por intervalo angular",
                xaxis_title="Intervalo [α° → α°]",
                yaxis_title="σ(ΔPresión) [Pa]",
                paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                font=dict(color='white'), height=320
            )
            st.plotly_chart(fig_gaps, use_container_width=True)

            # Recomendar ángulos adicionales
            st.markdown("#### 💡 Ángulos Recomendados para Medir")
            # Ordenar intervalos por mayor variación → son los que más necesitan refuerzo
            gaps_ordenados = gap_df.sort_values('Variación (σ ΔP)', ascending=False)
            rec_rows = []
            for _, row in gaps_ordenados.iterrows():
                alpha_mid = round((row['α_inicio'] + row['α_fin']) / 2, 2)
                rec_rows.append({
                    'Intervalo Crítico': row['Intervalo'],
                    'Δα [°]': row['Δα [°]'],
                    'Variación': row['Variación (σ ΔP)'],
                    'α Recomendado a Medir [°]': alpha_mid
                })
            df_rec = pd.DataFrame(rec_rows)
            st.dataframe(df_rec, use_container_width=True)

            # Resumen de la recomendación principal
            if not gaps_ordenados.empty:
                top_gap = gaps_ordenados.iloc[0]
                alpha_sug = round((top_gap['α_inicio'] + top_gap['α_fin']) / 2, 2)
                st.info(
                    f"📌 **Recomendación principal:** El intervalo más crítico es "
                    f"**{top_gap['Intervalo']}** (Δα = {top_gap['Δα [°]']}°, "
                    f"variación = {top_gap['Variación (σ ΔP)']:.4f} Pa). "
                    f"Se recomienda medir un plano en **α = {alpha_sug}°** para capturar mejor el fenómeno."
                )
elif st.session_state.seccion_actual == 'herramientas':

    st.markdown("""
    <div class="header-container">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🔧 HERRAMIENTAS DE PROCESAMIENTO
        </h1>
        <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Herramientas avanzadas para el procesamiento y análisis de datos aerodinámicos
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar variables de sesión si no existen
    if 'archivos_unidos' not in st.session_state:
        st.session_state.archivos_unidos = None
    if 'matriz_presiones' not in st.session_state:
        st.session_state.matriz_presiones = None
    if 'archivo_vtk' not in st.session_state:
        st.session_state.archivo_vtk = None
    
    # --- HERRAMIENTA 1: Unir Archivos de Incertidumbre ---
    with st.container():
        st.markdown("""
        <div class="section-card" style="border-left: 5px solid #0ea5e9; margin-bottom: 10px;">
            <h3 style="color: #0ea5e9; margin: 0;"> 01. UNIÓN DE ARCHIVOS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        c_desc, c_func = st.columns([1, 2])
        
        with c_desc:
            st.markdown("""
            <p style="color: #ccc; font-size: 0.95rem;">
                Utilidad para combinar múltiples archivos CSV de incertidumbre en un único conjunto de datos.
                <br><br>
                El sistema detectará automáticamente si hay puntos temporales sobrepuestos y generará una alerta.
            </p>
            """, unsafe_allow_html=True)

        with c_func:
            archivos_union = st.file_uploader(
                "Seleccionar archivos CSV:",
                type=['csv'],
                accept_multiple_files=True,
                key="union_archivos",
                label_visibility="collapsed"
            )
            
            c_input, c_btn = st.columns([2, 1])
            with c_input:
                 nombre_archivo_union = st.text_input("Nombre saliente:", value="archivos_unidos", key="nombre_union")
            with c_btn:
                 st.write("") # Spacer
                 st.write("")
                 btn_unir = st.button("🔗 Unir Ahora", key="btn_unir", type="primary", use_container_width=True)

            if btn_unir:
                if archivos_union and len(archivos_union) > 1:
                    with st.spinner("Uniendo archivos..."):
                        contenido_unido, puntos_sobrepuestos = unir_archivos_incertidumbre(
                            archivos_union, nombre_archivo_union
                        )
                        
                        if contenido_unido:
                            st.session_state.archivos_unidos = {
                                'contenido': contenido_unido,
                                'nombre': nombre_archivo_union,
                                'puntos_sobrepuestos': puntos_sobrepuestos
                            }
                            
                            st.success(f"✅ {len(archivos_union)} archivos unidos correctamente")
                            
                            if puntos_sobrepuestos:
                                st.warning(f"⚠️ Se detectaron {len(puntos_sobrepuestos)} puntos sobrepuestos")
                                with st.expander("Ver puntos sobrepuestos"):
                                    for punto in puntos_sobrepuestos:
                                        st.write(f"Y={punto[0]}, Z={punto[1]}, Tiempo={punto[2]}s")
                            
                            # Botón de descarga
                            st.download_button(
                                label="📥 Descargar CSV Unido",
                                data=contenido_unido.encode('utf-8-sig'),
                                file_name=f"{nombre_archivo_union}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("  Seleccione min. 2 archivos")

    st.markdown("---")

    # --- HERRAMIENTA 2: Extraer matriz de presiones ---
    with st.container():
        st.markdown("""
        <div class="section-card" style="border-left: 5px solid #f59e0b; margin-bottom: 10px;">
            <h3 style="color: #f59e0b; margin: 0;">📊 02. MATRIZ DE PRESIONES</h3>
        </div>
        """, unsafe_allow_html=True)
        
        c_desc, c_func = st.columns([1, 2])
        
        with c_desc:
            st.markdown("""
            <p style="color: #ccc; font-size: 0.95rem;">
                Extrae una matriz estructurada (Filas=Y, Columnas=Z) de presiones a partir de datos crudos.
                <br><br>
                Ideal para análisis numérico posterior o verificación manual de campos.
            </p>
            """, unsafe_allow_html=True)
            
        with c_func:
            archivo_matriz = st.file_uploader(
                "Cargar archivo de incertidumbre:",
                type=['csv'],
                key="archivo_matriz",
                 label_visibility="collapsed"
            )
            
            c_input, c_btn = st.columns([2, 1])
            with c_input:
                nombre_matriz = st.text_input("Nombre matriz:", value="matriz_presiones", key="nombre_matriz")
            with c_btn:
                st.write("")
                st.write("")
                btn_matriz = st.button("📊 Extraer", key="btn_matriz", type="primary", use_container_width=True)

            # Configuración de sensores para esta herramienta
            with st.expander("Configuración de Sensores y Atmósfera (Avanzado)"):
                configuracion_matriz = mostrar_configuracion_sensores("herramienta2")
                upl_inf_vtk = st.file_uploader("🔗 'Valores en el infinito.txt' para normalización VTK:", type=['txt', 'csv'], key="upl_inf_vtk")

            if btn_matriz:
                if archivo_matriz:
                    with st.spinner("Procesando..."):
                        matriz = extraer_matriz_presiones_completa(archivo_matriz, configuracion_matriz, upl_inf_vtk)

                        if matriz is not None and not matriz.empty:
                            st.session_state.matriz_presiones = {
                                'matriz': matriz,
                                'nombre': nombre_matriz
                            }

                            st.success("✅ Matriz extraída")
                            st.dataframe(matriz.head(), use_container_width=True)

                            # Botón de descarga
                            df_matriz = pd.DataFrame(matriz, columns=["Y", "Z", "Presion"])
                            csv_matriz = df_matriz.to_csv(sep=';', decimal=',', index=False)
                            st.download_button(
                                label="📥 Descargar Matriz CSV",
                                data=csv_matriz.encode('utf-8-sig'),
                                file_name=f"{nombre_matriz}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("  Falta archivo")

    st.markdown("---")

    # --- HERRAMIENTA 3: Generador VTK ---
    with st.container():
        st.markdown("""
        <div class="section-card" style="border-left: 5px solid #10b981; margin-bottom: 10px;">
            <h3 style="color: #10b981; margin: 0;">🎯 03. GENERADOR VTK (CFD)</h3>
            <p style="color:#aaa; margin: 6px 0 0 0; font-size:0.9rem;">
                Convierte datos de presión en archivos <b>.VTK</b> compatibles con ParaView / Salome.
                Elegí el tipo de archivo (2D, 3D o 4D) y la fuente de datos.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ─── VTK 2D ─────────────────────────────────────────────────────
        with st.expander("🗺 VTK 2D — Plano de Presión  |  Plano YZ (X fijo). Presión como color.", expanded=True):

            fuente_2d = st.radio(
                "Fuente de datos:",
                ["📂 Drive 2D (base de datos)", "💾 Memoria (sesión actual)", " Subir CSV nuevo"],
                key="fuente_vtk2d", horizontal=True
            )

            df_vtk2d = None
            fname_2d_drive = None          # nombre original del archivo en Drive

            if fuente_2d == "📂 Drive 2D (base de datos)":
                archivos_2d = auth.get_user_files_2d(st.session_state.username)
                if archivos_2d:
                    dict_2d = {f"{a[1]} [{a[2][:10] if a[2] else ''}]": a for a in archivos_2d}
                    sel_2d = st.selectbox("Seleccionar archivo 2D:", list(dict_2d.keys()), key="sel_drive2d_vtk")
                    if sel_2d:
                        fid_2d = dict_2d[sel_2d][0]
                        fname_2d_drive = dict_2d[sel_2d][1]          # ej: 2D-X500-OAO10-T5s.csv
                        raw_2d = auth.download_file_2d(fid_2d)
                        if raw_2d:
                            import io
                            df_vtk2d = pd.read_csv(io.BytesIO(raw_2d), sep=';', decimal=',')
                            st.success(f"✅ Cargado desde Drive: **{fname_2d_drive}**")
                else:
                    st.info("No hay archivos 2D en Drive. Guardá desde BETZ 2D → Paso 4.")

            elif fuente_2d == "💾 Memoria (sesión actual)":
                mat_disp = st.session_state.get('matriz_presiones')
                if mat_disp:
                    st.success(f"✅ Usando: {mat_disp['nombre']}")
                    df_vtk2d = mat_disp['matriz']
                else:
                    st.warning("No hay matriz en sesión. Usá Herramienta 02 o cargá desde Drive.")

            else:
                csv_new = st.file_uploader("CSV Matriz (sep=;, dec=,):", type=['csv'], key="up_vtk2d")
                if csv_new:
                    try:
                        df_vtk2d = pd.read_csv(csv_new, sep=';', decimal=',')
                    except Exception as e:
                        st.error(f"Error leyendo CSV: {e}")

            if df_vtk2d is not None:
                x_vtk2d = st.number_input(" Posición X [mm]:", value=0.0, step=10.0, key="x_vtk2d")
                res_vtk2d = st.slider("Suavizado:", 1, 5, 2, key="res_vtk2d")

                # Nombre auto: si viene de Drive reemplazamos prefijo, sino usamos X
                if fname_2d_drive:
                    stem_2d = os.path.splitext(fname_2d_drive)[0]      # 2D-X500-OAO10-T5s
                    nombre_auto_vtk2d = "VTK-" + stem_2d[stem_2d.index("-")+1:] if "-" in stem_2d else f"VTK-{stem_2d}"
                else:
                    nombre_auto_vtk2d = f"VTK-X{int(x_vtk2d)}-2D"

                c2d_chk, c2d_nom = st.columns([0.18, 0.82])
                if c2d_chk.checkbox("Nombre libre", key="chk_vtk2d"):
                    nombre_vtk2d = c2d_nom.text_input("Nombre:", placeholder=nombre_auto_vtk2d, key="nom_vtk2d")
                    if not nombre_vtk2d: nombre_vtk2d = nombre_auto_vtk2d
                else:
                    nombre_vtk2d = nombre_auto_vtk2d
                    c2d_nom.code(f"{nombre_vtk2d}.vtk")

                if st.button("🗺 Generar VTK 2D", key="btn_gen_vtk2d", type="primary"):
                    resultado_2d = crear_vtk_plano_presion_2d(df_vtk2d, nombre_vtk2d, x_vtk2d)
                    if resultado_2d:
                        vtk_path_2d, vtk_bytes_2d = resultado_2d
                        c_dl1, c_dl2 = st.columns(2)
                        with c_dl1:
                            st.download_button("📥 Descargar VTK 2D", vtk_bytes_2d,
                                               file_name=os.path.basename(vtk_path_2d),
                                               mime="application/octet-stream", key="dl_vtk2d")
                        with c_dl2:
                            if st.button(" Guardar en Drive", key="save_vtk2d_drive"):
                                if auth.save_vtk_plano(st.session_state.username,
                                                       os.path.basename(vtk_path_2d), vtk_bytes_2d):
                                    st.success("✅ Subido → HERRAMIENTAS/ARCHIVOS VTK/PLANOS DE PRESION")
                                else:
                                    st.error("Error al subir a Drive")
                    else:
                        st.error("  No se pudo generar el VTK.")

        # ─── VTK 3D ─────────────────────────────────────────────────────
        with st.expander("🕸 VTK 3D — Malla Delaunay  |  Triangulación 3D fiel a los datos. Ideal para CFD.", expanded=True):

            fuente_3d = st.radio(
                "Fuente de datos:",
                ["📂 Drive 3D (base de datos)", "💾 Memoria (sesión actual)", " Subir CSV nuevo"],
                key="fuente_vtk3d", horizontal=True
            )

            df_vtk3d = None
            fname_3d_drive = None

            if fuente_3d == "📂 Drive 3D (base de datos)":
                archivos_3d = auth.get_user_surfaces(st.session_state.username)
                if archivos_3d:
                    dict_3d = {f"{s[1]} [{s[3][:10] if s[3] else ''}]": s for s in archivos_3d}
                    sel_3d = st.selectbox("Seleccionar plano 3D:", list(dict_3d.keys()), key="sel_drive3d_vtk")
                    if sel_3d:
                        import json as _json
                        fname_3d_drive = dict_3d[sel_3d][1]          # ej: 3D-X500-OAO10
                        data_str_3d = dict_3d[sel_3d][4]
                        df_vtk3d = pd.DataFrame(_json.loads(data_str_3d))
                        st.success(f"✅ Cargado: **{fname_3d_drive}**")
                else:
                    st.info("No hay planos 3D en Drive. Guardá desde BETZ 3D → Paso 5.")

            elif fuente_3d == "💾 Memoria (sesión actual)":
                mat_disp3d = st.session_state.get('matriz_presiones')
                if mat_disp3d:
                    st.success(f"✅ Usando: {mat_disp3d['nombre']}")
                    df_vtk3d = mat_disp3d['matriz']
                else:
                    st.warning("No hay matriz en sesión.")

            else:
                csv_new3d = st.file_uploader("CSV Matriz:", type=['csv'], key="up_vtk3d")
                if csv_new3d:
                    try:
                        df_vtk3d = pd.read_csv(csv_new3d, sep=';', decimal=',')
                    except Exception as e:
                        st.error(f"Error: {e}")

            if df_vtk3d is not None:
                x_vtk3d = st.number_input(" Posición X [mm]:", value=0.0, step=10.0, key="x_vtk3d")

                if fname_3d_drive:
                    stem_3d = os.path.splitext(fname_3d_drive)[0]
                    nombre_auto_vtk3d = "VTK-" + stem_3d[stem_3d.index("-")+1:] if "-" in stem_3d else f"VTK-{stem_3d}"
                else:
                    nombre_auto_vtk3d = f"VTK-X{int(x_vtk3d)}-3D"

                c3d_chk, c3d_nom = st.columns([0.18, 0.82])
                if c3d_chk.checkbox("Nombre libre", key="chk_vtk3d"):
                    nombre_vtk3d = c3d_nom.text_input("Nombre:", placeholder=nombre_auto_vtk3d, key="nom_vtk3d")
                    if not nombre_vtk3d: nombre_vtk3d = nombre_auto_vtk3d
                else:
                    nombre_vtk3d = nombre_auto_vtk3d
                    c3d_nom.code(f"{nombre_vtk3d}.vtk")

                if st.button("🕸 Generar VTK 3D Delaunay", key="btn_gen_vtk3d", type="primary"):
                    res_3d = crear_vtk_superficie_3d_delaunay(df_vtk3d, nombre_vtk3d, x_vtk3d)
                    if res_3d:
                        with open(res_3d, "rb") as f3d:
                            vtk_bytes_3d = f3d.read()
                        c_dl3, c_dl4 = st.columns(2)
                        with c_dl3:
                            st.download_button("📥 Descargar VTK 3D", vtk_bytes_3d,
                                               file_name=f"{nombre_vtk3d}.vtk",
                                               mime="application/octet-stream", key="dl_vtk3d")
                        with c_dl4:
                            if st.button(" Guardar en Drive", key="save_vtk3d_drive"):
                                if auth.save_vtk_superficie(st.session_state.username,
                                                            f"{nombre_vtk3d}.vtk", vtk_bytes_3d):
                                    st.success("✅ Subido → HERRAMIENTAS/ARCHIVOS VTK/SUPERFICIES 3D")
                                else:
                                    st.error("Error al subir a Drive")
                    else:
                        st.error("  No se pudo generar el VTK.")

        # ─── VTK 4D ─────────────────────────────────────────────────────
        with st.expander("🌌 VTK 4D — Multi-plano  |  Genera un VTK Delaunay por cada plano 4D, en su estación X.", expanded=True):

            archivos_4d_vtk = auth.get_user_surfaces_4d(st.session_state.username)
            if not archivos_4d_vtk:
                st.info("No hay planos 4D en Drive. Guardá desde BETZ 4D → Paso 1.")
            else:
                dict_4d_vtk = {f"{s[1]} (X={s[2]}mm) [{s[3][:10] if s[3] else ''}]": s for s in archivos_4d_vtk}
                sels_4d = st.multiselect("Seleccionar planos 4D:", list(dict_4d_vtk.keys()), key="sels_4d_vtk")

                if sels_4d:
                    rename_4d = st.checkbox("Personalizar nombres de salida", key="chk_rename_4d")

                    if rename_4d:
                        custom_names_4d = {}
                        for lab in sels_4d:
                            s4 = dict_4d_vtk[lab]
                            stem_4d = s4[1]
                            auto_4d = "VTK-" + stem_4d[stem_4d.index("-")+1:] if "-" in stem_4d else f"VTK-{stem_4d}"
                            custom_names_4d[lab] = st.text_input(
                                f"  Nombre para {s4[1]}:", value=auto_4d, key=f"nom4d_{s4[1]}"
                            )
                    else:
                        # Mostrar los nombres automáticos como preview
                        for lab in sels_4d:
                            s4 = dict_4d_vtk[lab]
                            stem_4d = s4[1]
                            auto_4d = "VTK-" + stem_4d[stem_4d.index("-")+1:] if "-" in stem_4d else f"VTK-{stem_4d}"
                            st.code(f"{auto_4d}.vtk", language=None)

                    if st.button("🌌 Generar VTK por cada plano", key="btn_gen_vtk4d", type="primary"):
                        import json as _json4
                        for lab in sels_4d:
                            s4 = dict_4d_vtk[lab]
                            df_s4 = pd.DataFrame(_json4.loads(s4[4]))
                            x_s4  = s4[2]
                            stem_4d = s4[1]
                            auto_4d = "VTK-" + stem_4d[stem_4d.index("-")+1:] if "-" in stem_4d else f"VTK-{stem_4d}"
                            nom_s4 = custom_names_4d.get(lab, auto_4d) if rename_4d else auto_4d
                            with st.spinner(f"Generando {nom_s4}.vtk..."):
                                res_s4 = crear_vtk_superficie_3d_delaunay(df_s4, nom_s4, x_s4)
                            if res_s4:
                                with open(res_s4, "rb") as f4:
                                    bytes_s4 = f4.read()
                                st.download_button(
                                    f"📥 {nom_s4}.vtk",
                                    bytes_s4,
                                    file_name=f"{nom_s4}.vtk",
                                    mime="application/octet-stream",
                                    key=f"dl_vtk4d_{s4[1]}"
                                )
                                st.success(f"✅ {nom_s4}.vtk generado")
                            else:
                                st.error(f"Error generando VTK para {s4[1]}")










elif st.session_state.seccion_actual == 'configuracion':
    # Hero Title for Config
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(180deg, #000 0%, #111 100%); margin-bottom: 3rem;">
        <h1 style="font-size: 3.5rem; margin-bottom: 1rem; letter-spacing: 4px; color: #fff; text-transform: uppercase;">Configuración</h1>
        <p style="color: #666; font-size: 1.2rem; max-width: 600px; margin: 0 auto;">Estado del sistema y gestión de parámetros operativos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔌 Estado del Sistema")
    
    # Check Database
    db_ok = os.path.exists("users.db")
    db_size = os.path.getsize("users.db") / 1024 if db_ok else 0
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="section-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 10px;">💾</div>
            <h4 style="margin:0; color:white;">Base de Datos</h4>
            <div style="font-size: 1.2rem; font-weight:bold; color: {'#4ade80' if db_ok else '#f87171'}; margin: 10px 0;">
                {' ONLINE' if db_ok else ' OFFLINE'}
            </div>
            <p style="color: grey; font-size: 0.8rem; margin:0;">users.db ({db_size:.1f} KB)</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="section-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 10px;">👤</div>
            <h4 style="margin:0; color:white;">Sesión Activa</h4>
            <div style="font-size: 1.2rem; font-weight:bold; color: #60a5fa; margin: 10px 0;">
                {st.session_state.username}
            </div>
            <p style="color: grey; font-size: 0.8rem; margin:0;">Privilegios: {'Admin' if st.session_state.username=='admin' else 'Usuario'}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="section-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🚀</div>
            <h4 style="margin:0; color:white;">Versión App</h4>
            <div style="font-size: 1.2rem; font-weight:bold; color: #a78bfa; margin: 10px 0;">
                v2.1.0
            </div>
            <p style="color: grey; font-size: 0.8rem; margin:0;">Build: {datetime.now().strftime('%Y.%m.%d')}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗃 Explorador de Archivos en Drive")
    st.info("Navegá entre carpetas, renombrá o eliminá archivos de tu cuenta en Google Drive.")

    import drive_api as _dapi

    # --- Inicializar session state del explorador ---
    if 'drive_folder_path' not in st.session_state:
        st.session_state.drive_folder_path = []   # Lista de (id, nombre)
    if 'drive_current_folder_id' not in st.session_state:
        st.session_state.drive_current_folder_id = None
    if 'drive_rename_file_id' not in st.session_state:
        st.session_state.drive_rename_file_id = None
    if 'drive_confirm_delete_id' not in st.session_state:
        st.session_state.drive_confirm_delete_id = None

    # --- Obtener la carpeta raíz del usuario la primera vez ---
    if st.session_state.drive_current_folder_id is None:
        with st.spinner("Conectando con Google Drive..."):
            user_root_id = _dapi.get_user_root(st.session_state.username)
        if user_root_id:
            st.session_state.drive_current_folder_id = user_root_id
            st.session_state.drive_folder_path = [(user_root_id, f" {st.session_state.username}")]
        else:
            st.error("  No se pudo conectar con Google Drive. Verificá las credenciales.")
            user_root_id = None

    current_folder_id = st.session_state.drive_current_folder_id

    if current_folder_id:
        # --- BREADCRUMB ---
        breadcrumb_cols = st.columns(len(st.session_state.drive_folder_path) * 2)
        for i, (fid, fname) in enumerate(st.session_state.drive_folder_path):
            with breadcrumb_cols[i * 2]:
                is_last = (i == len(st.session_state.drive_folder_path) - 1)
                if is_last:
                    st.markdown(f"<span style='color:#60a5fa; font-weight:bold;'>{fname}</span>", unsafe_allow_html=True)
                else:
                    if st.button(fname, key=f"bread_{fid}"):
                        # Navegar hacia atrás a esta carpeta
                        idx = next((j for j, (x, _) in enumerate(st.session_state.drive_folder_path) if x == fid), None)
                        if idx is not None:
                            st.session_state.drive_folder_path = st.session_state.drive_folder_path[:idx + 1]
                            st.session_state.drive_current_folder_id = fid
                            st.session_state.drive_rename_file_id = None
                            st.session_state.drive_confirm_delete_id = None
                            st.rerun()
            if i < len(st.session_state.drive_folder_path) - 1:
                with breadcrumb_cols[i * 2 + 1]:
                    st.markdown("<span style='color:#555;'> › </span>", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#222; margin: 0.5rem 0;'>", unsafe_allow_html=True)

        # --- CARGAR CONTENIDO DE LA CARPETA ACTUAL ---
        with st.spinner("Cargando contenido..."):
            contenido = _dapi.list_folder_contents(current_folder_id)

        FOLDER_MIME = 'application/vnd.google-apps.folder'
        carpetas = [f for f in contenido if f.get('mimeType') == FOLDER_MIME]
        archivos = [f for f in contenido if f.get('mimeType') != FOLDER_MIME]

        if not contenido:
            st.markdown("<p style='color:#666; font-style:italic;'>Esta carpeta está vacía.</p>", unsafe_allow_html=True)
        else:
            # --- MOSTRAR CARPETAS ---
            for carpeta in carpetas:
                c_icon, c_name, c_btn = st.columns([0.05, 0.75, 0.2])
                with c_icon:
                    st.markdown(" 📁")
                with c_name:
                    st.markdown(f"<span style='color:#fbbf24;'>{carpeta['name']}</span>", unsafe_allow_html=True)
                with c_btn:
                    if st.button("Abrir →", key=f"open_{carpeta['id']}"):
                        st.session_state.drive_folder_path.append((carpeta['id'], f" {carpeta['name']}"))
                        st.session_state.drive_current_folder_id = carpeta['id']
                        st.session_state.drive_rename_file_id = None
                        st.session_state.drive_confirm_delete_id = None
                        st.rerun()

            if carpetas and archivos:
                st.markdown("<div style='border-top: 1px solid #222; margin: 0.3rem 0;'></div>", unsafe_allow_html=True)

            # --- MOSTRAR ARCHIVOS ---
            for archivo in archivos:
                fid  = archivo['id']
                fname = archivo['name']
                created = archivo.get('createdTime', '')[:10] if archivo.get('createdTime') else ''

                is_renaming = (st.session_state.drive_rename_file_id == fid)
                is_confirming_delete = (st.session_state.drive_confirm_delete_id == fid)

                if is_renaming:
                    # --- MODO RENOMBRAR ---
                    r_col1, r_col2, r_col3 = st.columns([0.6, 0.2, 0.2])
                    with r_col1:
                        nuevo_nombre = st.text_input("Nuevo nombre:", value=fname, key=f"inp_rename_{fid}", label_visibility="collapsed")
                    with r_col2:
                        if st.button("✅ Guardar", key=f"confirm_rename_{fid}"):
                            with st.spinner("Renombrando..."):
                                ok = _dapi.rename_file(fid, nuevo_nombre)
                            if ok:
                                st.success(f"✅ Renombrado a '{nuevo_nombre}'")
                            else:
                                st.error("  Error al renombrar.")
                            st.session_state.drive_rename_file_id = None
                            st.rerun()
                    with r_col3:
                        if st.button("✖ Cancelar", key=f"cancel_rename_{fid}"):
                            st.session_state.drive_rename_file_id = None
                            st.rerun()

                elif is_confirming_delete:
                    # --- MODO CONFIRMACIÓN BORRADO ---
                    st.warning(f"⚠️ ¿Seguro que querés eliminar **{fname}**? Esta acción es irreversible.")
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        if st.button("🗑️ Sí, eliminar", type="primary", key=f"confirm_del_{fid}"):
                            with st.spinner("Eliminando..."):
                                ok = _dapi.delete_file(fid)
                            if ok:
                                st.success(f"✅ '{fname}' eliminado.")
                            else:
                                st.error("  Error al eliminar.")
                            st.session_state.drive_confirm_delete_id = None
                            st.rerun()
                    with d_col2:
                        if st.button("Cancelar", key=f"cancel_del_{fid}"):
                            st.session_state.drive_confirm_delete_id = None
                            st.rerun()

                else:
                    # --- MODO NORMAL ---
                    f_col1, f_col2, f_col3, f_col4 = st.columns([0.05, 0.65, 0.15, 0.15])
                    with f_col1:
                        st.markdown("📄")
                    with f_col2:
                        st.markdown(f"<span style='color:#e5e7eb;'>{fname}</span>"
                                    f"<br><span style='color:#555; font-size:0.75rem;'>{created}</span>",
                                    unsafe_allow_html=True)
                    with f_col3:
                        if st.button(" Renombrar", key=f"ren_{fid}"):
                            st.session_state.drive_rename_file_id = fid
                            st.session_state.drive_confirm_delete_id = None
                            st.rerun()
                    with f_col4:
                        if st.button("🗑️ Eliminar", key=f"del_{fid}"):
                            st.session_state.drive_confirm_delete_id = fid
                            st.session_state.drive_rename_file_id = None
                            st.rerun()

    st.markdown("---")
    st.markdown("### 👥 Gestión de Usuarios")
    
    if st.session_state.username == 'admin':
        st.success("✅ Acceso de Administrador - Panel de Gestión de Usuarios")
        
        with st.expander("➕ Crear Nuevo Usuario", expanded=False):
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                new_username = st.text_input("Nombre de Usuario", key="admin_new_user")
            with col_u2:
                new_password = st.text_input("Contraseña", type="password", key="admin_new_pass")
            
            if st.button("Crear Usuario", type="primary"):
                if not new_username or not new_password:
                    st.error("Complete todos los campos")
                elif len(new_password) < 4:
                    st.error("La contraseña debe tener al menos 4 caracteres")
                else:
                    if auth.create_user(new_username, new_password):
                        st.success(f"✅ Usuario '{new_username}' creado exitosamente")
                    else:
                        st.error(f"  El usuario '{new_username}' ya existe")
        
        st.info("💡 Los usuarios creados podrán acceder inmediatamente con sus credenciales.")
    else:
        st.warning("⚠️ Solo el administrador puede gestionar usuarios. Contacte al admin para solicitar acceso.")

elif st.session_state.seccion_actual == 'smn_2d':
    from codigo_fuente.Ensayo_SMN_2D import show_smn_2d
    show_smn_2d()
elif st.session_state.seccion_actual == 'smn_3d':
    from codigo_fuente.Ensayo_SMN_3D import show_smn_3d
    show_smn_3d()
elif st.session_state.seccion_actual == 'smn_4d':
    from codigo_fuente.Ensayo_SMN_4D import show_smn_4d
    show_smn_4d()


# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p><strong>Laboratorio de Aerodinámica y Fluidos - UTN HAEDO</strong></p>
    <p>Sistema de Análisis de Datos Aerodinámicos • Versión 1.43 - Con Herramientas de Procesamiento</p>
    <p><small>Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</small></p>
</div>
""", unsafe_allow_html=True)
