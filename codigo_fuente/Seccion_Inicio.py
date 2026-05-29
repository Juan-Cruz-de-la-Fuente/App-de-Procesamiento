import streamlit as st


import base64
from codigo_fuente.Estilos import apply_styles

def show_inicio():
    apply_styles()
    
    import os
    import base64
    import mimetypes

    if 'inicio_images_b64' not in st.session_state or not st.session_state.inicio_images_b64:
        from pathlib import Path
        import base64
        
        # Ruta absoluta: Seccion_Inicio.py está en codigo_fuente/, volvemos uno atrás para llegar a Aplicacion/
        base_path = Path(__file__).parent.parent
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
            st.session_state.inicio_images_b64 = img_list[:12]
        else:
            st.session_state.inicio_images_b64 = []

    img_b64_list = st.session_state.inicio_images_b64
                
    if not img_b64_list:
        fallback_url = 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop'
        carousel_html = f'<div style="position: absolute; top:0; left:0; right:0; bottom:0; background-size: cover; background-position: center; background-image: linear-gradient(to bottom, rgba(0,0,0,0.5), rgba(0,0,0,0.9)), url(\'{fallback_url}\'); z-index: 0; opacity: 1;"></div>'
        carousel_css = ""
    else:
        num_imgs = len(img_b64_list)
        if num_imgs == 1:
            carousel_html = f'<div style="position: absolute; top:0; left:0; right:0; bottom:0; background-size: cover; background-position: center; background-image: linear-gradient(to bottom, rgba(0,0,0,0.5), rgba(0,0,0,0.9)), url(\'{img_b64_list[0]}\'); z-index: 0; opacity: 1;"></div>'
            carousel_css = ""
        else:
            time_per_slide = 5
            total_time = num_imgs * time_per_slide
            p_visible = 100.0 / num_imgs
            p_fade = p_visible * 0.2
            
            carousel_css = f"""
            <style>
            @keyframes crossFadeInicio {{
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
                .inicio-bg-{i} {{
                    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                    background-image: linear-gradient(to bottom, rgba(0,0,0,0.5), rgba(0,0,0,0.9)), url('{b64_full}');
                    background-size: cover; background-position: center;
                    animation: crossFadeInicio {total_time}s infinite linear;
                    animation-delay: {delay}s;
                    opacity: 0; z-index: 0;
                }}
                """
                carousel_html += f'<div class="inicio-bg-{i}"></div>\n'
            carousel_css += "</style>\n"

    st.markdown(carousel_css + f"""
<div style="position: relative; text-align: center; padding: 5rem 2rem; border-radius: 8px; overflow: hidden; margin-bottom: 3rem; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
{carousel_html}
<div style="position: relative; z-index: 10;">
<h1 style="font-size: 4rem; margin-bottom: 1rem; letter-spacing: 8px; color: #fff; text-transform: uppercase; text-shadow: 0 0 20px rgba(255,255,255,0.2);">
AeroAnalysis Pro
</h1>
<p style="color: rgba(255,255,255,0.9); font-size: 1.5rem; letter-spacing: 2px; text-transform: uppercase; text-shadow: 0 4px 10px rgba(0,0,0,0.8);">
Laboratorio de Aerodinámica y Fluidos - UTN HAEDO
</p>
<div style="width: 100px; height: 3px; background: #60a5fa; margin: 2rem auto; box-shadow: 0 0 10px #60a5fa;"></div>
</div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    
    features = [
        ("📈 ANÁLISIS 1D/2D", "Procesamiento de perfiles de estela y cálculo de pérdida de cantidad de movimiento.", "betz_1d"),
        ("🌪️ SUPERFICIES 3D", "Generación de mallas Delaunay y visualización volumétrica de campos de presión.", "betz_3d"),
        ("🌀 VÓRTICES", "Detección automática de núcleos y análisis cinemático de estructuras vorticosas.", "vortices")
    ]
    
    for i, (title, desc, section) in enumerate(features):
        with [c1, c2, c3][i]:
            st.markdown(f"""
<div class="section-card" style="height: 250px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; transition: all 0.3s ease;">
<h3 style="color: #60a5fa; margin-bottom: 1rem;">{title}</h3>
<p style="color: #ccc; font-size: 0.9rem;">{desc}</p>
</div>
""", unsafe_allow_html=True)
            if st.button(f"Explorar {title.split()[-1]}", key=f"btn_feat_{i}", use_container_width=True):
                st.session_state.seccion_actual = section
                st.rerun()

    st.markdown("""
<div style="margin-top: 4rem; padding: 2rem; background: rgba(255,255,255,0.03); border-radius: 15px; border: 1px solid rgba(255,255,255,0.05);">
<h4 style="color: #fff; margin-bottom: 1rem;">🚀 Acceso Rápido</h4>
<p style="color: #888; font-size: 0.9rem;">
Bienvenido al sistema avanzado de procesamiento de datos aerodinámicos. Utilice la barra de navegación superior 
o los botones de acceso rápido para comenzar su análisis. Todos los datos se sincronizan automáticamente con 
su cuenta de Google Drive para mayor seguridad y accesibilidad.
</p>
</div>
""", unsafe_allow_html=True)
