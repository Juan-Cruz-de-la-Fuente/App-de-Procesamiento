# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from scipy.spatial import Delaunay
from codigo_fuente import Auth_Manager as auth

def show_smn_3d():
    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🧪 ENSAYO-SMN 3D
            </h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Superficie Tridimensional Delaunay de Campos de Flujo
            </h2>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 2px solid #333; margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)

    # --- SESSION STATE INITIALIZATION ---
    if 'smn_archivos_memoria' not in st.session_state:
        st.session_state.smn_archivos_memoria = {}
    if 'smn_surf_seleccionada' not in st.session_state:
        st.session_state.smn_surf_seleccionada = pd.DataFrame()
    if 'smn_surf_nombre' not in st.session_state:
        st.session_state.smn_surf_nombre = ""
    if 'smn_v_inf' not in st.session_state: st.session_state.smn_v_inf = 17.5
    if 'smn_rho_inf' not in st.session_state: st.session_state.smn_rho_inf = 1.2
    if 'smn_p_inf' not in st.session_state: st.session_state.smn_p_inf = -94.0 
    if 'smn_t_inf' not in st.session_state: st.session_state.smn_t_inf = 15.0

    # --- CONFIGURACIÓN DE CONDICIONES ATMOSFÉRICAS DEL INFINITO ---
    st.markdown("### 🌐 Condiciones del Infinito y Referencia")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        st.session_state.smn_v_inf = c1.number_input("Velocidad Infinito V_∞ [m/s]:", value=st.session_state.smn_v_inf, format="%.2f", key="smn_3d_v_inf_input")
        st.session_state.smn_rho_inf = c2.number_input("Densidad Infinito ρ_∞ [kg/m³]:", value=st.session_state.smn_rho_inf, format="%.4f", key="smn_3d_rho_inf_input")
        st.session_state.smn_p_inf = c3.number_input("Presión de Referencia P_∞ [Pa]:", value=st.session_state.smn_p_inf, format="%.1f", key="smn_3d_p_inf_input")
        st.session_state.smn_t_inf = c4.number_input("Temperatura T_∞ [°C]:", value=st.session_state.smn_t_inf, format="%.1f", key="smn_3d_t_inf_input")

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📥 Cargar y Guardar Superficie 3D")
    st.caption("Procesá un CSV de sonda multiagujero para crear una superficie Delaunay 3D interactiva.")
    
    up_smn_3d = st.file_uploader("Subir archivo de ensayo SMN (.csv)", type=['csv'], key="up_smn_3d")
    
    if up_smn_3d:
        try:
            df_raw = pd.read_csv(up_smn_3d, sep=';', decimal=',')
            if 'Posicion Sonda X[mm]' not in df_raw.columns:
                df_raw = pd.read_csv(up_smn_3d, sep=',', decimal='.')
            
            required = ['Posicion Sonda X[mm]', 'Posicion Sonda Y[mm]']
            if not all(col in df_raw.columns for col in required):
                st.error("❌ El archivo CSV no contiene columnas válidas de posición.")
            else:
                st.success(f"✅ Archivo leído correctamente: {len(df_raw)} puntos.")
                df_proc = pd.DataFrame()
                df_proc['Y'] = df_raw['Posicion Sonda X[mm]'].astype(float)
                df_proc['Z'] = df_raw['Posicion Sonda Y[mm]'].astype(float)
                
                var_mappings = {
                    'Presion_Est': 'Presion estatica [Pa]',
                    'Presion_Tot': 'Presion total [Pa]',
                    'Vel_Tot': 'Velocidad [m/seg]',
                    'Vx': 'Velocidad X [m/seg]',
                    'Vy': 'Velocidad Y [m/seg]',
                    'Vz': 'Velocidad Z [m/seg]',
                    'Alfa': 'Alfa []',
                    'Beta': 'Beta []'
                }
                for k, col in var_mappings.items():
                    found_col = next((c for c in df_raw.columns if c.replace(' ', '').lower() == col.replace(' ', '').lower() or k.lower() in c.lower()), None)
                    if found_col is not None:
                        df_proc[k] = df_raw[found_col].astype(float)
                    else:
                        df_proc[k] = 0.0
                
                q_inf = 0.5 * st.session_state.smn_rho_inf * (st.session_state.smn_v_inf ** 2)
                if q_inf != 0:
                    df_proc['Cp_Est'] = (df_proc['Presion_Est'] - st.session_state.smn_p_inf) / q_inf
                    df_proc['Cp_Tot'] = (df_proc['Presion_Tot'] - st.session_state.smn_p_inf) / q_inf
                else:
                    df_proc['Cp_Est'] = 0.0
                    df_proc['Cp_Tot'] = 0.0
                    
                name_mem = up_smn_3d.name.replace('.csv', '')
                st.session_state.smn_archivos_memoria[name_mem] = df_proc
        except Exception as e:
            st.error(f"Error procesando CSV: {e}")
            
    op_smn = list(st.session_state.smn_archivos_memoria.keys()) if st.session_state.smn_archivos_memoria else ["No hay archivos"]
    sel_smn_3d = st.selectbox("Seleccionar Archivo a Guardar en Drive (3D):", op_smn, key="sel_smn_3d_save")
    smn_x_3d = st.number_input("Posición del plano X [mm]:", value=150.0, step=10.0, key="smn_x_3d")
    smn_aoa_3d = st.number_input("Ángulo AOA [°]:", value=0.0, step=1.0, key="smn_aoa_3d")
    
    nombre_auto_3d = f"SMN-3D-X{int(smn_x_3d)}-OAO{str(smn_aoa_3d).replace('-','neg')}-T0s"
    nombre_final_3d = st.text_input("Nombre de la Superficie en Drive:", value=nombre_auto_3d, key="name_final_3d")
    
    if st.button("🚀 SUBIR SUPERFICIE A DRIVE (SMN 3D)", use_container_width=True, type="primary", disabled=not st.session_state.smn_archivos_memoria):
        if sel_smn_3d in st.session_state.smn_archivos_memoria:
            df_to_save = st.session_state.smn_archivos_memoria[sel_smn_3d].copy()
            df_to_save['Pos_X'] = smn_x_3d
            df_to_save['AOA'] = smn_aoa_3d
            json_data = df_to_save.to_json(orient='records')
            if auth.save_surface_data(st.session_state.username, nombre_final_3d, json_data):
                st.success(f"✅ Guardado en Drive: {nombre_final_3d}")
            else:
                st.error("Error al guardar en Drive.")
                
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Cargar superficies de SMN desde Drive o Memoria
    st.markdown("### 📥 PASO 2: Selección de Superficie 3D")
    modo_carga_3d = st.radio("Carga desde:", ["🗄️ Base de Datos (Drive)", "🧠 Memoria de Sesión"], horizontal=True, key="modo_carga_smn_3d")
    
    df_active_3d = pd.DataFrame()
    if modo_carga_3d == "🗄️ Base de Datos (Drive)":
        try:
            drv_surfs = auth.get_user_surfaces(st.session_state.username)
            drv_surfs_smn = [s for s in drv_surfs if s[1].startswith("SMN-3D-")]
        except:
            drv_surfs_smn = []
            
        if not drv_surfs_smn:
            st.info("No hay superficies SMN guardadas en Drive.")
        else:
            dict_drv = {s[1]: s for s in drv_surfs_smn}
            sel_drv_3d = st.selectbox("Seleccionar Superficie en Drive:", ["-- Seleccionar --"] + list(dict_drv.keys()), key="sel_drv_smn_3d")
            if sel_drv_3d != "-- Seleccionar --":
                if 'last_drv_smn_3d' not in st.session_state or st.session_state.last_drv_smn_3d != sel_drv_3d:
                    with st.spinner("Descargando superficie SMN..."):
                        s_data = list(dict_drv[sel_drv_3d])
                        if not s_data[4]:
                            s_data[4] = auth.get_surface_data_string(s_data[0])
                        df_active_3d = pd.read_json(io.StringIO(s_data[4]))
                        st.session_state.smn_surf_seleccionada = df_active_3d
                        st.session_state.smn_surf_nombre = sel_drv_3d
                        st.session_state.last_drv_smn_3d = sel_drv_3d
                        st.rerun()
    else:
        if not st.session_state.smn_archivos_memoria:
            st.warning("⚠️ No hay superficies SMN en la memoria de sesión.")
        else:
            sel_mem_3d = st.selectbox("Seleccionar Superficie en Memoria:", list(st.session_state.smn_archivos_memoria.keys()), key="sel_mem_smn_3d")
            if st.button("📥 Cargar Superficie de Memoria al Visualizador", use_container_width=True, key="btn_load_smn_3d"):
                st.session_state.smn_surf_seleccionada = st.session_state.smn_archivos_memoria[sel_mem_3d]
                st.session_state.smn_surf_nombre = sel_mem_3d
                st.success("✅ Superficie cargada.")
                st.rerun()
                
    # Graficación 3D
    if 'smn_surf_seleccionada' in st.session_state and st.session_state.smn_surf_seleccionada is not None and not st.session_state.smn_surf_seleccionada.empty:
        df_s = st.session_state.smn_surf_seleccionada.copy()
        st.markdown("### 🎨 Opciones y Superficie Delaunay 3D")
        
        c_opt3, c_opt4 = st.columns(2)
        var_options_3d = {
            'Presión Estática [Pa]': 'Presion_Est',
            'Presión Total [Pa]': 'Presion_Tot',
            'Velocidad Total [m/s]': 'Vel_Tot',
            'Velocidad inducida Vx [m/s]': 'Vx',
            'Velocidad inducida Vy [m/s]': 'Vy',
            'Velocidad inducida Vz [m/s]': 'Vz',
            'Coeficiente de Presión Cp Est.': 'Cp_Est',
            'Coeficiente de Presión Cp Tot.': 'Cp_Tot',
            'Ángulo Alfa [°]': 'Alfa',
            'Ángulo Beta [°]': 'Beta'
        }
        var_options_3d = {k: v for k, v in var_options_3d.items() if v in df_s.columns}
        
        var_sel_3d = c_opt3.selectbox("Variable a representar como Altura (Z):", list(var_options_3d.keys()), key="var_sel_smn_3d")
        show_pts_3d = c_opt4.checkbox("Mostrar marcas de puntos medidos", value=True, key="show_pts_smn_3d")
        
        col_var_3d = var_options_3d[var_sel_3d]
        df_clean = df_s.dropna(subset=['Y', 'Z', col_var_3d]).drop_duplicates(subset=['Y', 'Z'])
        
        if len(df_clean) >= 4:
            py = df_clean['Y'].values
            pz = df_clean['Z'].values
            pv = df_clean[col_var_3d].values
            
            tri = Delaunay(np.vstack([py, pz]).T)
            
            fig = go.Figure()
            fig.add_trace(go.Mesh3d(
                x=py, y=pz, z=pv,
                i=tri.simplices[:, 0], j=tri.simplices[:, 1], k=tri.simplices[:, 2],
                intensity=pv, colorscale='Turbo', colorbar_title=var_sel_3d,
                name='Superficie Delaunay',
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5, roughness=0.5)
            ))
            
            if show_pts_3d:
                fig.add_trace(go.Scatter3d(
                    x=py, y=pz, z=pv,
                    mode='markers',
                    marker=dict(size=3, color='red', opacity=0.8),
                    name='Puntos medidos'
                ))
                
            fig.update_layout(
                title=f"Visualización Tridimensional: {st.session_state.get('smn_surf_nombre', 'Plano SMN')}",
                scene=dict(
                    xaxis_title="Y (Envergadura) [mm]",
                    yaxis_title="Z (Altura) [mm]",
                    zaxis_title=var_sel_3d,
                    aspectmode='data'
                ),
                width=1200,
                height=800,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Se necesitan al menos 4 puntos para triangular la superficie 3D.")
