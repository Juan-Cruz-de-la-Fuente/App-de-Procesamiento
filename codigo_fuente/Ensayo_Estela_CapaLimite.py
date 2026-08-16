import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import re
from scipy.interpolate import griddata
from codigo_fuente.Calculations_Core import (
    procesar_promedios,
    obtener_numero_sensor_desde_columna,
    calcular_altura_absoluta_z
)
from codigo_fuente import Auth_Manager as auth
from codigo_fuente.Graficos_Comunes import mostrar_configuracion_sensores

def _calcular_valores_infinito(txt_bytes, timestamp_str):
    try:
        df_inf = pd.read_csv(io.BytesIO(txt_bytes), sep=';', skip_blank_lines=True)
        df_inf.columns = [str(c).strip() for c in df_inf.columns]
        
        if len(df_inf.columns) > 2:
            first_col = df_inf.columns[0]
            df_inf["ts_clean"] = df_inf[first_col].astype(str).str.split(',').str[0].str.strip()
            df_inf["dt_val"] = pd.to_datetime(df_inf["ts_clean"], format='%d%m%y%H%M%S', errors='coerce')
            
            mask_failed = df_inf["dt_val"].isna()
            if mask_failed.any():
                df_inf.loc[mask_failed, "dt_val"] = pd.to_datetime(
                    df_inf.loc[mask_failed, "ts_clean"], format='%y%m%d%H%M%S', errors='coerce'
                )
            df_inf = df_inf.dropna(subset=["dt_val"])
            
            if df_inf.empty: return None
                
            ts_clean = str(timestamp_str).split(',')[0].strip()
            target_dt = pd.to_datetime(ts_clean, format='%d%m%y%H%M%S', errors='coerce')
            if pd.isna(target_dt):
                target_dt = pd.to_datetime(ts_clean, format='%y%m%d%H%M%S', errors='coerce')
            
            if pd.isna(target_dt): return None
                
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
            
            return float(rho)
    except Exception as e:
        st.warning(f"Error al vincular valores en el infinito: {e}")
    return None

def show_capa_limite():
    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🌬️ CAPA LÍMITE - Análisis 2D
            </h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Mapeo de Velocidades en el Plano Transversal (Ps = max(P))
            </h2>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 2px solid #333; margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)

    if 'configuracion_cl_local' not in st.session_state: st.session_state.configuracion_cl_local = None
    if "archivos_cl_memoria" not in st.session_state: st.session_state.archivos_cl_memoria = {}
    if "matriz_seleccionada_cl" not in st.session_state: st.session_state.matriz_seleccionada_cl = pd.DataFrame()
    if "cl_rho_inf" not in st.session_state: st.session_state.cl_rho_inf = 1.225

    # Configuración Global
    col_dens, _ = st.columns([1, 2])
    densidad = col_dens.number_input("Densidad del aire (ρ) [kg/m³]:", value=st.session_state.cl_rho_inf, step=0.01, format="%.4f")

    # --- CARGA Y CONFIGURACIÓN ---
    with st.expander("📥 CARGA Y CONFIGURACIÓN DE MATRICES NUEVAS", expanded=True):
        st.markdown("### 📥 PROCESAMIENTO DE ARCHIVOS CRUDOS")
        conf = mostrar_configuracion_sensores("cl_local")
        if st.button("💾 CONFIRMAR CONFIGURACIÓN PARA PROCESAR", use_container_width=True, key="btn_conf_cl"):
            st.session_state.configuracion_cl_local = conf
            st.success("✅ Configuración de procesamiento lista.")

        st.markdown("---")
        
        c_up1, c_up2 = st.columns(2)
        with c_up1:
            up_cl = st.file_uploader("Arrastre sus archivos CSV aquí", type=['csv'], accept_multiple_files=True, key="up_cl")
        with c_up2:
            up_inf_cl = st.file_uploader("Subir archivo Valores en el infinito (.txt)", type=['txt'], key="up_inf_cl")
        
        if up_cl and up_inf_cl:
            # Tomamos el timestamp del primer archivo subido como referencia para vincular el infinito
            ts_m = re.search(r'(\d{10,14})', up_cl[0].name)
            if ts_m:
                timestamp_detectado = ts_m.group(1)
                rho_calc = _calcular_valores_infinito(up_inf_cl.getvalue(), timestamp_detectado)
                if rho_calc:
                    if st.session_state.cl_rho_inf != rho_calc:
                        st.session_state.cl_rho_inf = rho_calc
                        st.success(f"✅ Densidad del aire actualizada automáticamente: ρ = {rho_calc:.4f} kg/m³")
                        st.rerun()

        if up_cl and st.session_state.configuracion_cl_local:
            for f in up_cl:
                name = f.name.replace('.csv', '').replace('incertidumbre_', '')
                if name not in st.session_state.archivos_cl_memoria:
                    with st.spinner(f"🔨 Procesando {name}..."):
                        datos = procesar_promedios(f, st.session_state.configuracion_cl_local['orden'])
                        if datos is not None:
                            st.session_state.archivos_cl_memoria[name] = datos
            st.success(f"✅ {len(st.session_state.archivos_cl_memoria)} matrices en memoria.")

        st.markdown("#### 🚀 Guardar Matriz 2D en Drive")
        opciones_cl = list(st.session_state.archivos_cl_memoria.keys()) if st.session_state.archivos_cl_memoria else ["No hay archivos"]
        arc_sel = st.selectbox("Seleccionar Archivo para guardar:", opciones_cl, key="sel_save_cl")
        
        tiempos = [0]
        if st.session_state.archivos_cl_memoria and arc_sel in st.session_state.archivos_cl_memoria:
            df_arc = st.session_state.archivos_cl_memoria[arc_sel]
            if 'Tiempo_s' in df_arc.columns:
                tiempos = sorted(df_arc['Tiempo_s'].dropna().unique())
                
        t_sel = st.selectbox("Tiempo [s]:", tiempos, key="t_sel_save_cl")
        
        c1_s, c2_s = st.columns(2)
        x_pos = c1_s.number_input("Posición X [mm]:", value=0.0, key="x_pos_cl_save")
        aoa = c2_s.number_input("AOA [°]:", value=0.0, key="aoa_cl_save")
        
        nombre_auto_cl = f"CL-2D-X{int(x_pos)}-OAO{str(aoa).replace('-','neg')}-T{int(t_sel)}s.csv"
        
        if 'last_nombre_auto_cl' not in st.session_state:
            st.session_state.last_nombre_auto_cl = nombre_auto_cl
            st.session_state.nombre_final_cl_save = nombre_auto_cl
            
        if st.session_state.last_nombre_auto_cl != nombre_auto_cl:
            st.session_state.nombre_final_cl_save = nombre_auto_cl
            st.session_state.last_nombre_auto_cl = nombre_auto_cl
            
        nombre_final_cl = st.text_input("Nombre del archivo a guardar:", key="nombre_final_cl_save")
        
        if st.button("🚀 SUBIR MATRIZ A DRIVE", use_container_width=True, type="primary", disabled=not st.session_state.archivos_cl_memoria, key="btn_save_cl"):
            if st.session_state.archivos_cl_memoria and arc_sel in st.session_state.archivos_cl_memoria:
                df_arc = st.session_state.archivos_cl_memoria[arc_sel]
                df_run = df_arc[df_arc['Tiempo_s'] == t_sel].copy() if 'Tiempo_s' in df_arc.columns else df_arc.copy()
                
                # Transformar el dataframe en una grilla Y,Z
                res = []
                for _, row in df_run.iterrows():
                    y_t = row.get('Pos_Y_Traverser', 0)
                    z_b = row.get('Pos_Z_Base', 0)
                    for col in df_run.columns:
                        num = obtener_numero_sensor_desde_columna(col)
                        if num is not None:
                            val = row[col]
                            if pd.isna(val): continue
                            z_r = calcular_altura_absoluta_z(num, z_b, st.session_state.configuracion_cl_local['distancia_toma_12'], st.session_state.configuracion_cl_local['distancia_entre_tomas'], 12, st.session_state.configuracion_cl_local['orden'])
                            res.append({'Y': y_t, 'Z': z_r, 'Presion': val})
                
                df_matriz_save = pd.DataFrame(res)
                if not df_matriz_save.empty:
                    df_matriz_save['Presion'] = pd.to_numeric(df_matriz_save['Presion'], errors='coerce')
                    # Encontrar Presión Estática (Máximo valor global de presión en la matriz)
                    P_s = df_matriz_save['Presion'].max()
                    
                    # Calcular Velocidad para cada punto
                    df_matriz_save['Velocidad'] = np.sqrt(2 * (df_matriz_save['Presion'] - P_s).abs() / densidad)
                    
                    csv_b = df_matriz_save.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')
                    # Usamos save_csv_2d por ser un plano bidimensional, aunque esté en CL.
                    # Se guardará en la carpeta 2D de Drive
                    if auth.save_csv_2d(st.session_state.username, nombre_final_cl, csv_b):
                        st.success(f"✅ Matriz 2D guardada en Drive: {nombre_final_cl}")
                    else:
                        st.error("Error al guardar en Drive.")

    st.markdown("---")

    # --- PASO 2: Selección ---
    st.markdown("### 📥 PASO 2: Selección de Matrices para Análisis")
    modo_carga = st.radio("Cargar matrices desde:", ["🗄️ Base de Datos (Drive)", "🧠 Memoria de Sesión"], horizontal=True, key="modo_carga_cl")
    
    if modo_carga == "🗄️ Base de Datos (Drive)":
        try:
            archivos_drv = auth.get_user_files_2d(st.session_state.username)
        except:
            archivos_drv = []

        if not archivos_drv:
            st.info("No se encontraron matrices guardadas en Drive.")
        else:
            # Filtramos solo los archivos que empiecen con CL
            archivos_cl = [a for a in archivos_drv if a[1].startswith("CL-")]
            if not archivos_cl:
                st.info("No se encontraron matrices de Capa Límite (CL-...) guardadas en Drive.")
            else:
                dict_drv = {f"{a[1]} [{a[2][:10] if a[2] else ''}]": a for a in archivos_cl}
                sel_drv = st.selectbox("Seleccionar Matriz de Drive:", ["-- Seleccionar --"] + list(dict_drv.keys()), key="sel_perfiles_cl_ui")
                if sel_drv != "-- Seleccionar --":
                    if 'last_sel_drv_cl' not in st.session_state or st.session_state.last_sel_drv_cl != sel_drv:
                        with st.spinner("Descargando matriz..."):
                            raw = auth.download_file_2d(dict_drv[sel_drv][0])
                            if raw:
                                df_m = pd.read_csv(io.BytesIO(raw), sep=';', decimal=',')
                                if 'Y' not in df_m.columns:
                                    df_m = pd.read_csv(io.BytesIO(raw), sep=',', decimal='.')
                                st.session_state.matriz_seleccionada_cl = df_m
                                st.session_state.last_sel_drv_cl = sel_drv
                        st.success(f"✅ Matriz cargada y lista para visualizar.")
                        st.rerun()
    else:
        if not st.session_state.archivos_cl_memoria:
            st.warning("⚠️ No hay matrices en la memoria de sesión. Procese archivos en el Paso 1.")
        else:
            arc_mem_sel = st.selectbox("Seleccionar Matriz en Memoria:", list(st.session_state.archivos_cl_memoria.keys()), key="sel_mem_cl_ui")
            df_arc = st.session_state.archivos_cl_memoria[arc_mem_sel]
            tiempos = sorted(df_arc['Tiempo_s'].dropna().unique()) if 'Tiempo_s' in df_arc.columns else [0]
            t_sel_mem = st.selectbox("Tiempo [s]:", tiempos, key="t_sel_mem_cl_ui") if len(tiempos) > 1 else tiempos[0]
            
            if st.button("📥 Cargar Matriz al Visualizador", use_container_width=True, key="btn_load_mem_cl"):
                if not st.session_state.configuracion_cl_local:
                    st.error("⚠️ Falta confirmar la configuración del peine en el Paso 1.")
                else:
                    conf = st.session_state.configuracion_cl_local
                    df_run = df_arc[df_arc['Tiempo_s'] == t_sel_mem] if 'Tiempo_s' in df_arc.columns else df_arc.copy()
                    res = []
                    for _, row in df_run.iterrows():
                        y_t = row.get('Pos_Y_Traverser', 0)
                        z_b = row.get('Pos_Z_Base', 0)
                        for col in df_run.columns:
                            num = obtener_numero_sensor_desde_columna(col)
                            if num is not None:
                                val = row[col]
                                if pd.isna(val): continue
                                z_r = calcular_altura_absoluta_z(num, z_b, conf['distancia_toma_12'], conf['distancia_entre_tomas'], 12, conf['orden'])
                                res.append({'Y': y_t, 'Z': z_r, 'Presion': val})
                    
                    df_matriz_mem = pd.DataFrame(res)
                    if not df_matriz_mem.empty:
                        df_matriz_mem['Presion'] = pd.to_numeric(df_matriz_mem['Presion'], errors='coerce')
                        # Encontrar Presión Estática (Máximo valor global)
                        P_s = df_matriz_mem['Presion'].max()
                        # Calcular Velocidad para cada punto
                        df_matriz_mem['Velocidad'] = np.sqrt(2 * (df_matriz_mem['Presion'] - P_s).abs() / densidad)
                        
                        st.session_state.matriz_seleccionada_cl = df_matriz_mem
                        st.success(f"✅ Matriz de memoria ({arc_mem_sel}) cargada y procesada.")
                        st.rerun()
                    else:
                        st.error("No se pudieron procesar los puntos de la matriz.")

    st.markdown("---")
    
    # --- PASO 3: Visualización 2D ---
    st.markdown("### 🎨 PASO 3: Visualización 2D Interactiva")
    
    if st.session_state.matriz_seleccionada_cl.empty:
        st.warning("⚠️ Seleccione y cargue una matriz en el Paso 2 para visualizar.")
    else:
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            var_sel = st.selectbox("Variable a graficar:", ["Velocidad [m/s]", "Presión Medida [Pa]"])
        with c_opt2:
            render_type = st.selectbox("Tipo de Renderizado:", ["Contour Suavizado", "Mapa de Calor"])
            
        df_m = st.session_state.matriz_seleccionada_cl
        
        if var_sel == "Velocidad [m/s]":
            val_col = 'Velocidad'
        else:
            val_col = 'Presion'
            
        y, z, v = df_m['Y'].values, df_m['Z'].values, df_m[val_col].values
        
        # En caso de matriz degenerada (una sola línea vertical) fallará griddata, lo protegemos
        if len(np.unique(y)) > 1 and len(np.unique(z)) > 1:
            grid_y = np.linspace(y.min(), y.max(), 150)
            grid_z = np.linspace(z.min(), z.max(), 150)
            Gy, Gz = np.meshgrid(grid_y, grid_z)
            Gv = griddata((y, z), v, (Gy, Gz), method='cubic')
            
            fig = go.Figure()
            if render_type == "Contour Suavizado":
                fig.add_trace(go.Contour(x=grid_y, y=grid_z, z=Gv, colorscale='Jet', colorbar=dict(title=var_sel)))
            else:
                fig.add_trace(go.Heatmap(x=grid_y, y=grid_z, z=Gv, colorscale='Jet', colorbar=dict(title=var_sel)))
            
            fig.add_trace(go.Scatter(x=y, y=z, mode='markers', marker=dict(size=3, color='white', opacity=0.3), name='Puntos medidos'))
                
            fig.update_layout(title=f"Mapeo 2D Capa Límite: {var_sel}", xaxis_title="Y [mm]", yaxis_title="Z [mm]", height=700, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            fig.update_xaxes(scaleanchor="y", scaleratio=1)
            st.plotly_chart(fig, use_container_width=True, config={'modeBarButtonsToAdd': ['drawline', 'eraseshape']})
        else:
            # Fallback a scatter plot si es 1D
            st.warning("La matriz cargada solo contiene puntos en una dimensión (Y constante). Mostrando gráfico Scatter 1D.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=v, y=z, mode='lines+markers', name=var_sel))
            fig.update_layout(title=f"Perfil Capa Límite: {var_sel}", xaxis_title=var_sel, yaxis_title="Z [mm]", height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)
