import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import re
import io
from scipy.spatial import Delaunay
from codigo_fuente.Calculations_Core import calcular_variable_atmosferica, rotate_points, procesar_promedios, obtener_numero_sensor_desde_columna, calcular_altura_absoluta_z
from codigo_fuente.Graficos_Comunes import mostrar_configuracion_sensores
from codigo_fuente import Auth_Manager as auth

# Helper local: aplicar alpha/beta + traslación al modelo
def _aplicar_pose_modelo_4d(obj_base, alpha_deg, beta_deg, dx, dy, dz, cg):
    x = np.array(obj_base['x'], dtype=float) - cg['x']
    y = np.array(obj_base['y'], dtype=float) - cg['y']
    z = np.array(obj_base['z'], dtype=float) - cg['z']
    x, y, z = rotate_points(x, y, z, 0.0, float(alpha_deg), float(-beta_deg))
    x = x + cg['x'] + dx
    y = y + cg['y'] + dy
    z = z + cg['z'] + dz
    return x, y, z

def _extraer_aoa_4d(nombre):
    m = re.search(r'OAO(neg)?(\d+(?:[.,]\d+)?)', str(nombre), re.IGNORECASE)
    if m:
        return (-1 if m.group(1) else 1) * float(str(m.group(2)).replace(',', '.'))
    return None

def show_4d():
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

    if 'configuracion_4d_local' not in st.session_state: st.session_state.configuracion_4d_local = None
    if 'archivos_4d_memoria' not in st.session_state: st.session_state.archivos_4d_memoria = {}
    if 'planos_seleccionados_4d' not in st.session_state: st.session_state.planos_seleccionados_4d = []

    # --- PASO 1 (Expander) ---
    with st.expander("🛠️ PASO 1: Carga y Procesamiento de Archivos Crudos (Nuevos)", expanded=True):
        st.markdown("Configure el peine y cargue archivos CSV para procesar y guardar en la nube.")

        conf_cols = st.columns([1.5, 1])
        with conf_cols[0]:
            conf = mostrar_configuracion_sensores("4d_local")
            if st.button("💾 CONFIRMAR CONFIGURACIÓN PARA PROCESAR", use_container_width=True):
                st.session_state.configuracion_4d_local = conf
                st.success("✅ Configuración de procesamiento lista.")

        st.markdown("---")
        up_4d = st.file_uploader("Arrastre archivos CSV 4D aquí", type=['csv'], accept_multiple_files=True, key="up_4d")
        inf_4d = st.file_uploader("🔗 Archivo Infinito (Opcional)", type=['txt', 'csv'], key="inf_4d")
        
        if up_4d and st.session_state.configuracion_4d_local:
            for f in up_4d:
                name = f.name.replace('.csv', '').replace('incertidumbre_', '')
                if name not in st.session_state.archivos_4d_memoria:
                    with st.spinner(f"🔨 Procesando {name}..."):
                        datos = procesar_promedios(f, st.session_state.configuracion_4d_local['orden'], inf_4d)
                        if datos is not None:
                            st.session_state.archivos_4d_memoria[name] = datos
            st.success(f"✅ {len(st.session_state.archivos_4d_memoria)} archivos en memoria.")

        st.markdown("#### Guardar en Base de Datos de Drive (4D)")
        
        opciones_archivos = list(st.session_state.archivos_4d_memoria.keys()) if st.session_state.archivos_4d_memoria else ["No hay archivos cargados"]
        arc_sel = st.selectbox("Seleccionar Archivo en Memoria:", opciones_archivos)
        
        tiempos = [0]
        if st.session_state.archivos_4d_memoria and arc_sel in st.session_state.archivos_4d_memoria:
            df_arc = st.session_state.archivos_4d_memoria[arc_sel]
            tiempos = sorted(df_arc['Tiempo_s'].dropna().unique())
            
        t_sel = st.selectbox("Tiempo [s]:", tiempos)
        
        c1, c2 = st.columns(2)
        aoa = c1.number_input("AOA [°]:", value=0.0)
        x_pos = c2.number_input("Posición X [mm]:", value=0.0)
        nombre_auto = f"4D-X{int(x_pos)}-OAO{str(aoa).replace('-','neg')}-T{int(t_sel)}s"
        nombre_final = st.text_input("Nombre del archivo a guardar:", value=nombre_auto)
        
        if st.button("🚀 SUBIR PLANO A DRIVE (4D)", use_container_width=True, type="primary", disabled=not st.session_state.archivos_4d_memoria):
            if st.session_state.archivos_4d_memoria and arc_sel in st.session_state.archivos_4d_memoria:
                df_arc = st.session_state.archivos_4d_memoria[arc_sel]
                df_run = df_arc[df_arc['Tiempo_s'] == t_sel].copy()
                res = []
                for _, row in df_run.iterrows():
                    y_t = row.get('Pos_Y_Traverser')
                    z_b = row.get('Pos_Z_Base')
                    for col in df_run.columns:
                        num = obtener_numero_sensor_desde_columna(col)
                        if num is not None:
                            val = row[col]
                            if pd.isna(val): continue
                            z_r = calcular_altura_absoluta_z(num, z_b, st.session_state.configuracion_4d_local['distancia_toma_12'], st.session_state.configuracion_4d_local['distancia_entre_tomas'], 12, st.session_state.configuracion_4d_local['orden'])
                            res.append({'Y': y_t, 'Z': z_r, 'Presion': val})
                
                df_plane_save = pd.DataFrame(res)
                if not df_plane_save.empty:
                    json_4d = df_plane_save.to_json(orient='records')
                    if auth.save_surface_data_4d(st.session_state.username, nombre_final, x_pos, json_4d):
                        st.success(f"✅ Guardado en Drive: {nombre_final}")
                    else:
                        st.error("Error al guardar.")

    st.markdown("---")

    # --- PASO 2 (Sin Expander) ---
    st.markdown("### 📥 PASO 2: Selección de Planos para Análisis")
    modo_carga = st.radio("Cargar planos desde:", ["🗄️ Base de Datos (Drive)", "🧠 Memoria de Sesión"], horizontal=True)

    if modo_carga == "🗄️ Base de Datos (Drive)":
        try:
            planos_drv = auth.get_user_surfaces_4d(st.session_state.username)
        except:
            planos_drv = []

        if not planos_drv:
            st.info("No hay planos 4D en la Base de Datos.")
        else:
            dict_drv = {f"{p[1]} (X={p[2]})": p for p in planos_drv}
            planos_sel_labels = st.multiselect("Seleccionar Planos de Drive:", list(dict_drv.keys()))
            if st.button("📥 Cargar Planos de Drive al Visualizador"):
                with st.spinner("Descargando planos seleccionados..."):
                    loaded = []
                    for lbl in planos_sel_labels:
                        p_info = list(dict_drv[lbl])
                        if not p_info[4]:
                            p_info[4] = auth.get_surface_data_string(p_info[0])
                        loaded.append(tuple(p_info))
                    st.session_state.planos_seleccionados_4d = loaded
                st.success(f"✅ {len(planos_sel_labels)} planos cargados y listos para visualizar.")
    else:
        if not st.session_state.archivos_4d_memoria:
            st.warning("⚠️ No hay archivos en la memoria de sesión. Procese archivos en el Paso 1 primero.")
        else:
            st.info("Funcionalidad de carga múltiple desde memoria en desarrollo. Por favor guarde en Drive y cargue desde allí para el análisis 4D multicapa.")

    st.markdown("---")

    # --- PASO 3 (Sin Expander) ---
    st.markdown("### 🎨 PASO 3: Opciones de Visualización")
    col_var, col_scale, col_bg, col_opt = st.columns(4)
    with col_var:
        var_sel = st.selectbox("Variable a graficar:", ["Presión Total [Actual]", " ρ_∞", "V_∞", "P_∞"])
    with col_scale:
        scale = st.slider("Relieve (Presión -> Z en gráfico):", 0.0, 5.0, 1.0)
    with col_bg:
        vis_bg = st.selectbox("Fondo:", ["Oscuro (Negro)", "Claro (Blanco)"], index=0, key="vis_bg_4d")
    with col_opt:
        st.write("Opciones de Ejes y Modelo")
        show_model = st.checkbox("Mostrar Modelo 3D de Referencia", value=True)
        vis_ejes = st.checkbox("Mostrar Ejes 3D", value=True, key="vis_ejes_4d")

    st.markdown("---")

    # --- PASO 4 (Sin Expander) ---
    st.markdown("### 📈 PASO 4: Visualización 4D Interactiva")
    
    if not st.session_state.planos_seleccionados_4d:
        st.warning("⚠️ No hay planos cargados. Seleccione y cargue planos en el Paso 2.")
    else:
        fig = go.Figure()
        for p_info in st.session_state.planos_seleccionados_4d:
            df = pd.read_json(io.StringIO(p_info[4]))
            df['Presion'] = calcular_variable_atmosferica(df, var_sel)
            x_base = p_info[2]
            df_clean = df.dropna(subset=['Y', 'Z', 'Presion']).drop_duplicates(subset=['Y', 'Z'])
            if len(df_clean) < 3: continue
            tri = Delaunay(df_clean[['Y', 'Z']].values)
            p_ref = df_clean['Presion'].max()
            x_def = x_base - ((df_clean['Presion'] - p_ref) * scale)
            
            fig.add_trace(go.Mesh3d(
                x=x_def, y=df_clean['Y'], z=df_clean['Z'],
                i=tri.simplices[:,0], j=tri.simplices[:,1], k=tri.simplices[:,2],
                intensity=df_clean['Presion'], colorscale='Jet',
                name=f"X={x_base}", hovertemplate=f"X: {x_base:.2f}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<br>{var_sel}: %{{intensity:.2f}}<extra></extra>"
            ))

        if show_model and 'objeto_referencia_4d' in st.session_state:
            obj_base = st.session_state.objeto_referencia_base if 'objeto_referencia_base' in st.session_state else st.session_state.objeto_referencia_4d
            cg = st.session_state.get('modelo_cg', {'x': 0.0, 'y': 0.0, 'z': 0.0})
            
            # Detectar AOA automáticamente de los planos graficados
            alpha_auto = 0.0
            if st.session_state.planos_seleccionados_4d:
                p_name = st.session_state.planos_seleccionados_4d[0][1]
                aoa_val = _extraer_aoa_4d(p_name)
                if aoa_val is not None:
                    alpha_auto = aoa_val
                    
            # Aplicar rotación
            xm, ym, zm = _aplicar_pose_modelo_4d(obj_base, alpha_auto, 0.0, 0.0, 0.0, 0.0, cg)
            obj_ref = st.session_state.objeto_referencia_4d
            
            # Mesh o Scatter
            if obj_ref.get('type') == 'scatter':
                fig.add_trace(go.Scatter3d(
                    x=xm, y=ym, z=zm,
                    mode='markers', marker=dict(size=2, color='#888', opacity=0.5),
                    name="Modelo"
                ))
            else:
                fig.add_trace(go.Mesh3d(
                    x=xm, y=ym, z=zm,
                    i=obj_ref['i'], j=obj_ref['j'], k=obj_ref['k'],
                    color='#5588cc', opacity=0.3, name=f"Modelo (α={alpha_auto:.1f}°)",
                    alphahull=0, showscale=False,
                    lighting=dict(ambient=0.4, diffuse=0.8)
                ))

        # Configurar colores de fondo
        bg_color = '#0e1117' if "Oscuro" in vis_bg else '#ffffff'
        font_color = 'white' if "Oscuro" in vis_bg else 'black'
        
        # Configurar visibilidad de ejes
        axis_props = dict(
            showgrid=vis_ejes, zeroline=vis_ejes, showticklabels=vis_ejes,
            showaxeslabels=vis_ejes, showbackground=False
        )

        # Rotar la cámara 180° sobre el eje Z (cambiando los signos de x e y en el eye de la cámara)
        camera_dict = dict(eye=dict(x=-1.25, y=-1.25, z=1.25))
        
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                camera=camera_dict,
                xaxis=dict(title="X (Estación)" if vis_ejes else "", autorange="reversed", **axis_props),
                yaxis=dict(title="Y (Envergadura)" if vis_ejes else "", **axis_props),
                zaxis=dict(title="Z (Altura)" if vis_ejes else "", **axis_props)
            ),
            height=800,
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=font_color)
        )
        st.plotly_chart(fig, use_container_width=True)

