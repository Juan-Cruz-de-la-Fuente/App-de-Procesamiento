# -*- coding: utf-8 -*-
import streamlit as st
import os
import base64
from codigo_fuente.Estilos import apply_styles
from codigo_fuente import Auth_Manager as auth
from codigo_fuente import Drive_Connection as drive_api

# Configuración de la página
st.set_page_config(
    page_title="Laboratorio de Aerodinámica y Fluidos - UTN HAEDO",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Aplicar estilos globales
apply_styles()

# --- INITIALIZATION ---
if 'seccion_actual' not in st.session_state:
    st.session_state.seccion_actual = 'inicio'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

# Variables de estado de datos
states = [
    'archivos_cargados', 'datos_procesados', 'configuracion_inicial',
    'sub_archivos_generados', 'datos_3d_filtrados', 'sub_archivos_3d',
    'configuracion_3d', 'sub_archivos_3d_generados', 'diferencias_guardadas'
]
for state in states:
    if state not in st.session_state:
        st.session_state[state] = {} if 'config' in state or 'datos' in state or 'sub' in state or 'dif' in state else []

def login_page():
    import os
    import base64
    import mimetypes

    if 'login_images_b64' not in st.session_state or not st.session_state.login_images_b64:
        from pathlib import Path
        import base64
        
        # Ruta absoluta basada en la ubicación de este archivo
        base_path = Path(__file__).parent
        folder_portada = base_path / "codigo_fuente" / "Imagenes de portada"
        
        img_list = []
        if folder_portada.exists():
            valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
            try:
                archivos = [f for f in folder_portada.iterdir() if f.suffix.lower() in valid_exts]
                for f_path in archivos:
                    try:
                        ext = f_path.suffix.lower()[1:]
                        if ext == 'jpg': ext = 'jpeg'
                        with open(f_path, 'rb') as img_f:
                            b64 = base64.b64encode(img_f.read()).decode()
                            img_list.append(f"data:image/{ext};base64,{b64}")
                    except: continue
            except: pass
            
            import random
            random.shuffle(img_list)
        
        if img_list:
            st.session_state.login_images_b64 = img_list[:12]
        else:
            st.session_state.login_images_b64 = []
        
    img_b64_list = st.session_state.login_images_b64
                
    if not img_b64_list:
        fallback_url = 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop'
        carousel_html = f'<div style="position: absolute; top:0; left:0; right:0; bottom:0; background-size: cover; background-position: center; background-image: linear-gradient(to bottom, rgba(0,0,0,0.2), rgba(0,0,0,0.8)), url(\'{fallback_url}\'); z-index: 0; opacity: 1;"></div>'
        carousel_css = ""
    else:
        num_imgs = len(img_b64_list)
        if num_imgs == 1:
            carousel_html = f'<div style="position: absolute; top:0; left:0; right:0; bottom:0; background-size: cover; background-position: center; background-image: linear-gradient(to bottom, rgba(0,0,0,0.2), rgba(0,0,0,0.8)), url(\'{img_b64_list[0]}\'); z-index: 0; opacity: 1;"></div>'
            carousel_css = ""
        else:
            time_per_slide = 5
            total_time = num_imgs * time_per_slide
            # Porcentaje de tiempo que cada imagen es visible (incluyendo fade)
            p_visible = 100.0 / num_imgs
            p_fade = p_visible * 0.2
            
            carousel_css = f"""
            <style>
            @keyframes crossFade {{
                0% {{ opacity: 0; transform: scale(1.05); }}
                {p_fade}% {{ opacity: 1; transform: scale(1.0); }}
                {p_visible - p_fade}% {{ opacity: 1; transform: scale(1.0); }}
                {p_visible}% {{ opacity: 0; transform: scale(1.05); }}
                100% {{ opacity: 0; }}
            }}
            """
            carousel_html = ""
            for i, b64_full in enumerate(img_b64_list):
                delay = -(i * time_per_slide)
                carousel_css += f"""
                .login-bg-{i} {{
                    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                    background-image: linear-gradient(to bottom, rgba(0,0,0,0.2), rgba(0,0,0,0.8)), url('{b64_full}');
                    background-size: cover; background-position: center;
                    animation: crossFade {total_time}s infinite linear;
                    animation-delay: {delay}s;
                    opacity: 0; z-index: 0;
                }}
                """
                carousel_html += f'<div class="login-bg-{i}"></div>\n'
            carousel_css += "</style>\n"

    st.markdown(carousel_css + f"""
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
        with st.container(border=True):
            st.markdown("### Acceso al Sistema")
            user = st.text_input("Usuario")
            pw = st.text_input("Contraseña", type="password")
            if st.button("INGRESAR", type="primary", use_container_width=True):
                if auth.verify_user(user, pw):
                    st.session_state.logged_in = True
                    st.session_state.username = user
                    drive_api.init_user_folders(user)
                    st.rerun()
                else:
                    if auth.check_db_status():
                        st.error("⚠️ **Error de Sincronización:** La base de datos está vacía. Esto indica que la conexión con Google Drive falló al iniciar y no pudo descargarse `users.db`. Por favor, asegúrate de configurar correctamente el secreto `google_token_json` en las configuraciones avanzadas de tu aplicación de Streamlit.")
                    else:
                        st.error("Credenciales incorrectas")

# --- MAIN LOGIC ---
def render_navbar():
    if st.session_state.get('logged_in'):
        with st.container():
            c_sl, c1, c2, c3, c4, c5, c_sr = st.columns([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])
            
            with c1:
                if st.button("🚀 INICIO", use_container_width=True, 
                             type="primary" if st.session_state.seccion_actual == 'inicio' else "secondary"):
                    st.session_state.seccion_actual = 'inicio'
                    st.rerun()
            
            with c2:
                with st.popover("🌌 ENSAYO ESTELA", use_container_width=True):
                    if st.button("📈 Vis. Estela 1D", use_container_width=True):
                        st.session_state.seccion_actual = 'betz_1d'
                        st.rerun()
                    if st.button("📈 Vis. Estela 2D", use_container_width=True):
                        st.session_state.seccion_actual = 'betz_2d'
                        st.rerun()
                    if st.button("🌪️ Vis. Estela 3D", use_container_width=True):
                        st.session_state.seccion_actual = 'betz_3d'
                        st.rerun()
                    if st.button("🌌 Vis. Estela 4D", use_container_width=True):
                        st.session_state.seccion_actual = 'betz_4d'
                        st.rerun()
                    if st.button("🌀 Análisis de Vórtices", use_container_width=True):
                        st.session_state.seccion_actual = 'vortices'
                        st.rerun()
                    if st.button("🎬 Animación 4D", use_container_width=True):
                        st.session_state.seccion_actual = 'animacion'
                        st.rerun()
                    if st.button("📊 INTERPOLACIÓN", use_container_width=True):
                        st.session_state.seccion_actual = 'interpolacion'
                        st.rerun()
                    if st.button("🔧 HERRAMIENTAS", use_container_width=True):
                        st.session_state.seccion_actual = 'herramientas'
                        st.rerun()

            with c3:
                if st.button("📦 MODELOS", use_container_width=True, 
                             type="primary" if st.session_state.seccion_actual == 'modelos' else "secondary"):
                    st.session_state.seccion_actual = 'modelos'
                    st.rerun()

            with c4:
                if st.button("⚙️ CONFIG", use_container_width=True, 
                             type="primary" if st.session_state.seccion_actual == 'configuracion' else "secondary"):
                    st.session_state.seccion_actual = 'configuracion'
                    st.rerun()

            with c5:
                if st.button(f"👤 SALIR ({st.session_state.username})", use_container_width=True):
                    st.session_state.logged_in = False
                    st.session_state.username = None
                    st.rerun()

            st.markdown("<hr style='border-top: 1px solid #333; margin-top: 10px;'>", unsafe_allow_html=True)

if not st.session_state.logged_in:
    login_page()
else:
    render_navbar()
    
    # Navegación de Secciones

    if st.session_state.seccion_actual == 'inicio':
        from codigo_fuente.Seccion_Inicio import show_inicio
        show_inicio()
    elif st.session_state.seccion_actual == 'betz_1d':
        from codigo_fuente.Ensayo_Estela_1D import show_1d
        show_1d()
    elif st.session_state.seccion_actual == 'betz_2d':
        from codigo_fuente.Ensayo_Estela_2D import show_2d
        show_2d()
    elif st.session_state.seccion_actual == 'betz_3d':
        from codigo_fuente.Ensayo_Estela_3D import show_3d
        show_3d()
    elif st.session_state.seccion_actual == 'betz_4d':
        from codigo_fuente.Ensayo_Estela_4D import show_4d
        show_4d()
    elif st.session_state.seccion_actual == 'vortices':
        from codigo_fuente.Ensayo_Estela_Analisis_Vortices import show_vortices
        show_vortices()
    elif st.session_state.seccion_actual == 'animacion':
        from codigo_fuente.Ensayo_Estela_Animacion_4D import show_animacion
        show_animacion()
    elif st.session_state.seccion_actual == 'interpolacion':
        from codigo_fuente.Ensayo_Estela_Interpolacion import show_interpolacion
        show_interpolacion()
    elif st.session_state.seccion_actual == 'modelos':
        from codigo_fuente.Ensayo_Estela_Modelos_3D import show_modelos
        show_modelos()
    elif st.session_state.seccion_actual == 'herramientas':
        from codigo_fuente.Herramientas import show_herramientas
        show_herramientas()
    elif st.session_state.seccion_actual == 'configuracion':
        from codigo_fuente.Configuracion import show_configuracion
        show_configuracion()
