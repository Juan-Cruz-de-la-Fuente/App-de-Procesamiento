# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from scipy.interpolate import griddata
from codigo_fuente import Auth_Manager as auth

def show_smn_2d():
    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🧪 ENSAYO-SMN 2D
            </h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Mapeo Bidimensional de Campos de Flujo y Coeficientes
            </h2>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 2px solid #333; margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)

    # --- SESSION STATE INITIALIZATION ---
    if 'smn_archivos_memoria' not in st.session_state:
        st.session_state.smn_archivos_memoria = {}
    if 'smn_matriz_seleccionada' not in st.session_state:
        st.session_state.smn_matriz_seleccionada = pd.DataFrame()
    if 'smn_v_inf' not in st.session_state: st.session_state.smn_v_inf = 17.5
    if 'smn_rho_inf' not in st.session_state: st.session_state.smn_rho_inf = 1.2
    if 'smn_p_inf' not in st.session_state: st.session_state.smn_p_inf = -94.0 
    if 'smn_t_inf' not in st.session_state: st.session_state.smn_t_inf = 15.0

    # --- CONFIGURACIÓN DE CONDICIONES ATMOSFÉRICAS DEL INFINITO ---
    st.markdown("### 🌐 Condiciones del Infinito y Referencia")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        st.session_state.smn_v_inf = c1.number_input("Velocidad Infinito V_∞ [m/s]:", value=st.session_state.smn_v_inf, format="%.2f", key="smn_2d_v_inf_input")
        st.session_state.smn_rho_inf = c2.number_input("Densidad Infinito ρ_∞ [kg/m³]:", value=st.session_state.smn_rho_inf, format="%.4f", key="smn_2d_rho_inf_input")
        st.session_state.smn_p_inf = c3.number_input("Presión de Referencia P_∞ [Pa]:", value=st.session_state.smn_p_inf, format="%.1f", key="smn_2d_p_inf_input")
        st.session_state.smn_t_inf = c4.number_input("Temperatura T_∞ [°C]:", value=st.session_state.smn_t_inf, format="%.1f", key="smn_2d_t_inf_input")

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📥 Cargar y Guardar Plano 2D")
    st.caption("Subí un archivo CSV de sonda multiagujero para procesar, calcular coeficientes y mapear.")
    
    # 1. Cargador de Archivo Crudo al inicio
    up_smn_2d = st.file_uploader("Subir archivo de ensayo SMN (.csv) - Ej: datos_fluido_0AOA", type=['csv'], key="up_smn_2d")
    
    if up_smn_2d:
        try:
            # Intentar leer en formato semicolon / coma
            df_raw = pd.read_csv(up_smn_2d, sep=';', decimal=',')
            if 'Posicion Sonda X[mm]' not in df_raw.columns:
                df_raw = pd.read_csv(up_smn_2d, sep=',', decimal='.')
            
            # Verificar columnas requeridas
            required = ['Posicion Sonda X[mm]', 'Posicion Sonda Y[mm]']
            if not all(col in df_raw.columns for col in required):
                st.error("❌ El archivo CSV no contiene columnas válidas de posición ('Posicion Sonda X[mm]', 'Posicion Sonda Y[mm]').")
            else:
                st.success(f"✅ Archivo leído correctamente: {len(df_raw)} puntos de control.")
                
                # Convertir columnas a variables estándar de graficación
                df_proc = pd.DataFrame()
                df_proc['Y'] = df_raw['Posicion Sonda X[mm]'].astype(float)
                df_proc['Z'] = df_raw['Posicion Sonda Y[mm]'].astype(float)
                
                # Variables físicas mapeadas
                var_mappings = {
                    'Presion_Est': 'Presion estatica [Pa]',
                    'Presion_Tot': 'Presion total [Pa]',
                    'Vel_Tot': 'Velocidad [m/seg]',
                    'Vx': 'Velocidad X [m/seg]',
                    'Vy': 'Velocidad Y [m/seg]',
                    'Vz': 'Velocidad Z [m/seg]',
                    'Alfa': 'Alfa []',
                    'Beta': 'Beta []',
                    'Cp_Alfa': 'Cp Alfa []',
                    'Cp_Beta': 'Cp Beta []'
                }
                
                for k, col in var_mappings.items():
                    # Buscar coincidencia exacta o parecida por acentos / caracteres raros
                    found_col = next((c for c in df_raw.columns if c.replace(' ', '').lower() == col.replace(' ', '').lower() or k.lower() in c.lower()), None)
                    if found_col is None and '' in col:
                        found_col = next((c for c in df_raw.columns if 'alfa' in c.lower() or 'beta' in c.lower()), None)
                    if found_col is not None:
                        df_proc[k] = df_raw[found_col].astype(float)
                    else:
                        df_proc[k] = 0.0
                
                # Calcular coeficientes adimensionales
                q_inf = 0.5 * st.session_state.smn_rho_inf * (st.session_state.smn_v_inf ** 2)
                if q_inf != 0:
                    df_proc['Cp_Est'] = (df_proc['Presion_Est'] - st.session_state.smn_p_inf) / q_inf
                    df_proc['Cp_Tot'] = (df_proc['Presion_Tot'] - st.session_state.smn_p_inf) / q_inf
                else:
                    df_proc['Cp_Est'] = 0.0
                    df_proc['Cp_Tot'] = 0.0
                    
                # Guardar archivo procesado en memoria de sesión
                name_mem = up_smn_2d.name.replace('.csv', '')
                st.session_state.smn_archivos_memoria[name_mem] = df_proc
        except Exception as e:
            st.error(f"Error procesando CSV: {e}")
            
    # 2. Guardado en Google Drive
    op_smn = list(st.session_state.smn_archivos_memoria.keys()) if st.session_state.smn_archivos_memoria else ["No hay archivos"]
    sel_smn = st.selectbox("Seleccionar Archivo a Guardar en Drive:", op_smn, key="sel_smn_2d_save")
    
    c_p1, c_p2 = st.columns(2)
    smn_x = c_p1.number_input("Posición del plano X [mm]:", value=150.0, step=10.0, key="smn_x_2d")
    smn_aoa = c_p2.number_input("Ángulo AOA [°]:", value=0.0, step=1.0, key="smn_aoa_2d")
    
    nombre_auto_2d = f"SMN-2D-X{int(smn_x)}-OAO{str(smn_aoa).replace('-','neg')}-T0s"
    nombre_final_2d = st.text_input("Nombre del archivo en Drive:", value=nombre_auto_2d, key="name_final_2d")
    
    if st.button("🚀 SUBIR MATRIZ A DRIVE (SMN 2D)", use_container_width=True, type="primary", disabled=not st.session_state.smn_archivos_memoria):
        if sel_smn in st.session_state.smn_archivos_memoria:
            df_to_save = st.session_state.smn_archivos_memoria[sel_smn].copy()
            df_to_save['Pos_X'] = smn_x
            df_to_save['AOA'] = smn_aoa
            csv_bytes = df_to_save.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')
            if auth.save_csv_2d(st.session_state.username, f"{nombre_final_2d}.csv", csv_bytes):
                st.success(f"✅ Guardado en Drive: {nombre_final_2d}.csv")
            else:
                st.error("Error al guardar en Drive.")
                
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 3. Carga y Visualización
    st.markdown("### 📥 PASO 2: Selección de Plano para Visualización 2D")
    modo_carga_2d = st.radio("Carga desde:", ["🗄️ Base de Datos (Drive)", "🧠 Memoria de Sesión"], horizontal=True, key="modo_carga_smn_2d")
    
    df_active = pd.DataFrame()
    if modo_carga_2d == "🗄️ Base de Datos (Drive)":
        try:
            drv_files = auth.get_user_files_2d(st.session_state.username)
            # Filtrar solo archivos de SMN
            drv_files_smn = [f for f in drv_files if f[1].startswith("SMN-2D-")]
        except:
            drv_files_smn = []
            
        if not drv_files_smn:
            st.info("No hay planos SMN guardados en tu Google Drive.")
        else:
            dict_drv = {f[1]: f for f in drv_files_smn}
            sel_drv = st.selectbox("Seleccionar Plano SMN en Drive:", ["-- Seleccionar --"] + list(dict_drv.keys()), key="sel_drv_smn_2d")
            if sel_drv != "-- Seleccionar --":
                if 'last_drv_smn_2d' not in st.session_state or st.session_state.last_drv_smn_2d != sel_drv:
                    with st.spinner("Descargando plano SMN..."):
                        raw = auth.download_file_2d(dict_drv[sel_drv][0])
                        if raw:
                            df_active = pd.read_csv(io.BytesIO(raw), sep=';', decimal=',')
                            if 'Y' not in df_active.columns:
                                df_active = pd.read_csv(io.BytesIO(raw), sep=',', decimal='.')
                            st.session_state.smn_matriz_seleccionada = df_active
                            st.session_state.last_drv_smn_2d = sel_drv
                            st.rerun()
    else:
        if not st.session_state.smn_archivos_memoria:
            st.warning("⚠️ No hay planos SMN en la memoria de la sesión.")
        else:
            sel_mem = st.selectbox("Seleccionar Plano en Memoria:", list(st.session_state.smn_archivos_memoria.keys()), key="sel_mem_smn_2d")
            if st.button("📥 Cargar Plano de Memoria al Visualizador", use_container_width=True):
                st.session_state.smn_matriz_seleccionada = st.session_state.smn_archivos_memoria[sel_mem]
                st.success("✅ Plano cargado correctamente.")
                st.rerun()
                
    # 4. Graficación 2D
    if not st.session_state.smn_matriz_seleccionada.empty:
        df_m = st.session_state.smn_matriz_seleccionada.copy()
        st.markdown("### 🎨 Opciones y Gráfico 2D")
        
        c_opt1, c_opt2 = st.columns(2)
        var_options = {
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
        var_options = {k: v for k, v in var_options.items() if v in df_m.columns}
        
        var_sel = c_opt1.selectbox("Variable a graficar (2D):", list(var_options.keys()), key="var_sel_smn_2d")
        render_type = c_opt2.selectbox("Tipo de Renderizado (2D):", ["Contour Suavizado", "Mapa de Calor"], key="render_type_smn_2d")
        
        col_var_name = var_options[var_sel]
        df_clean = df_m.dropna(subset=['Y', 'Z', col_var_name])
        y, z, v = df_clean['Y'].values, df_clean['Z'].values, df_clean[col_var_name].values
        
        if len(df_clean) >= 3:
            grid_y = np.linspace(y.min(), y.max(), 150)
            grid_z = np.linspace(z.min(), z.max(), 150)
            Gy, Gz = np.meshgrid(grid_y, grid_z)
            Gv = griddata((y, z), v, (Gy, Gz), method='cubic')
            
            fig = go.Figure()
            if render_type == "Contour Suavizado":
                fig.add_trace(go.Contour(x=grid_y, y=grid_z, z=Gv, colorscale='Turbo', colorbar=dict(title=var_sel)))
            else:
                fig.add_trace(go.Heatmap(x=grid_y, y=grid_z, z=Gv, colorscale='Turbo', colorbar=dict(title=var_sel)))
            
            fig.add_trace(go.Scatter(x=y, y=z, mode='markers', marker=dict(size=4, color='white', opacity=0.6), name='Puntos medidos'))
            
            fig.update_layout(
                title=f"Mapeo 2D SMN: {var_sel}",
                xaxis_title="Y (Envergadura) [mm]",
                yaxis_title="Z (Altura) [mm]",
                height=700,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )
            fig.update_xaxes(scaleanchor="y", scaleratio=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No hay suficientes puntos de medición para realizar la interpolación 2D.")
