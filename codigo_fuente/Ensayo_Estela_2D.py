import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
from scipy.interpolate import griddata
from codigo_fuente.Calculations_Core import (
    procesar_promedios,
    obtener_numero_sensor_desde_columna,
    calcular_altura_absoluta_z,
    calcular_variable_atmosferica
)
from codigo_fuente import Auth_Manager as auth

def show_2d():
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

    if 'configuracion_2d_local' not in st.session_state: st.session_state.configuracion_2d_local = None
    if 'archivos_2d_memoria' not in st.session_state: st.session_state.archivos_2d_memoria = {}
    if 'matriz_seleccionada_2d' not in st.session_state: st.session_state.matriz_seleccionada_2d = pd.DataFrame()

    from codigo_fuente.Graficos_Comunes import mostrar_configuracion_sensores

    # --- CARGA Y CONFIGURACIÓN (EXPANDER ÚNICO) ---
    with st.expander("📥 CARGA Y CONFIGURACIÓN DE MATRICES NUEVAS", expanded=True):
        st.markdown("""
        <div class="section-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: white;">📥 PROCESAMIENTO DE ARCHIVOS CRUDOS</h3>
            <p style="color: #bbb; margin-bottom: 20px;">Configure el peine y cargue archivos CSV para procesar y guardar en la nube.</p>
        </div>
        """, unsafe_allow_html=True)

        conf = mostrar_configuracion_sensores("2d_local")
        if st.button("💾 CONFIRMAR CONFIGURACIÓN PARA PROCESAR", use_container_width=True):
            st.session_state.configuracion_2d_local = conf
            st.success("✅ Configuración de procesamiento lista.")

        st.markdown("---")
        up_2d = st.file_uploader("Arrastre archivos CSV 2D aquí", type=['csv'], accept_multiple_files=True, key="up_2d")
        inf_2d = st.file_uploader("🔗 Archivo Infinito (Opcional)", type=['txt', 'csv'], key="inf_2d")
        
        if up_2d and st.session_state.configuracion_2d_local:
            for f in up_2d:
                name = f.name.replace('.csv', '').replace('incertidumbre_', '')
                if name not in st.session_state.archivos_2d_memoria:
                    with st.spinner(f"🔨 Procesando {name}..."):
                        datos = procesar_promedios(f, st.session_state.configuracion_2d_local['orden'], inf_2d)
                        if datos is not None:
                            st.session_state.archivos_2d_memoria[name] = datos
            st.success(f"✅ {len(st.session_state.archivos_2d_memoria)} archivos en memoria.")

        st.markdown("#### Guardar en Drive (2D)")
        op_2d = list(st.session_state.archivos_2d_memoria.keys()) if st.session_state.archivos_2d_memoria else ["No hay archivos"]
        arc_sel = st.selectbox("Seleccionar Archivo:", op_2d)
        
        tiempos = [0]
        if st.session_state.archivos_2d_memoria and arc_sel in st.session_state.archivos_2d_memoria:
            df_arc = st.session_state.archivos_2d_memoria[arc_sel]
            tiempos = sorted(df_arc['Tiempo_s'].dropna().unique())
            
        t_sel = st.selectbox("Tiempo [s]:", tiempos)
        
        c1, c2 = st.columns(2)
        aoa = c1.number_input("AOA [°]:", value=0.0)
        x_pos = c2.number_input("Posición X [mm]:", value=0.0)
        nombre_auto = f"2D-X{int(x_pos)}-OAO{str(aoa).replace('-','neg')}-T{int(t_sel)}s.csv"
        nombre_final = st.text_input("Nombre del archivo a guardar:", value=nombre_auto)
        
        if st.button("🚀 SUBIR MATRIZ A DRIVE (2D)", use_container_width=True, type="primary", disabled=not st.session_state.archivos_2d_memoria):
            if st.session_state.archivos_2d_memoria and arc_sel in st.session_state.archivos_2d_memoria:
                df_arc = st.session_state.archivos_2d_memoria[arc_sel]
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
                            z_r = calcular_altura_absoluta_z(num, z_b, st.session_state.configuracion_2d_local['distancia_toma_12'], st.session_state.configuracion_2d_local['distancia_entre_tomas'], 12, st.session_state.configuracion_2d_local['orden'])
                            res.append({'Y': y_t, 'Z': z_r, 'Presion': val, 'rho_inf': row.get('rho_inf', 1.225), 'V_inf': row.get('V_inf', 0.0), 'P_inf': row.get('P_inf', 101325.0)})
                
                df_matriz_save = pd.DataFrame(res)
                if not df_matriz_save.empty:
                    csv_b = df_matriz_save.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')
                    if auth.save_csv_2d(st.session_state.username, nombre_final, csv_b):
                        st.success(f"✅ Guardado en Drive: {nombre_final}")
                    else:
                        st.error("Error al guardar.")

    st.markdown("---")

    # --- PASO 2 (Sin Expander) ---
    st.markdown("### 📥 PASO 2: Selección de Matrices para Análisis")
    modo_carga = st.radio("Cargar matrices desde:", ["🗄️ Base de Datos (Drive)", "🧠 Memoria de Sesión"], horizontal=True, key="modo_carga_2d")

    if modo_carga == "🗄️ Base de Datos (Drive)":
        try:
            archivos_drv = auth.get_user_files_2d(st.session_state.username)
        except:
            archivos_drv = []

        if not archivos_drv:
            st.info("No hay matrices en Drive.")
        else:
            dict_drv = {f"{a[1]} [{a[2][:10] if a[2] else ''}]": a for a in archivos_drv}
            sel_drv = st.selectbox("Seleccionar Matriz de Drive:", ["-- Seleccionar --"] + list(dict_drv.keys()), key="sel_drv_2d_ui")
            if sel_drv != "-- Seleccionar --":
                if 'last_sel_drv_2d' not in st.session_state or st.session_state.last_sel_drv_2d != sel_drv:
                    with st.spinner("Descargando matriz seleccionada..."):
                        raw = auth.download_file_2d(dict_drv[sel_drv][0])
                        if raw:
                            df_m = pd.read_csv(io.BytesIO(raw), sep=';', decimal=',')
                            if 'Y' not in df_m.columns:
                                df_m = pd.read_csv(io.BytesIO(raw), sep=',', decimal='.')
                            st.session_state.matriz_seleccionada_2d = df_m
                            st.session_state.last_sel_drv_2d = sel_drv
                    st.success(f"✅ Matriz cargada y lista para visualizar.")
                    st.rerun()
    else:
        if not st.session_state.archivos_2d_memoria:
            st.warning("⚠️ No hay matrices en la memoria de sesión. Procese archivos en el Paso 1 primero.")
        else:
            arc_mem_sel = st.selectbox("Seleccionar Matriz en Memoria:", list(st.session_state.archivos_2d_memoria.keys()))
            if st.button("📥 Cargar Matriz al Visualizador", use_container_width=True):
                st.session_state.matriz_seleccionada_2d = st.session_state.archivos_2d_memoria[arc_mem_sel]
                st.success("✅ Matriz de memoria cargada.")

    st.markdown("---")

    # --- PASO 3 (Sin Expander) ---
    st.markdown("### 🎨 PASO 3: Opciones de Visualización")
    c_opt1, c_opt2, c_opt3 = st.columns(3)
    with c_opt1:
        var_sel = st.selectbox("Variable a graficar:", ["Presión Total [Actual]", " ρ_∞", "V_∞", "P_∞"])
    with c_opt2:
        render_type = st.selectbox("Tipo de Renderizado:", ["Contour Suavizado", "Mapa de Calor"])
    with c_opt3:
        cuerda = st.number_input("Cuerda de Referencia [mm]:", value=300.0)

    st.markdown("---")

    # --- PASO 4 (Sin Expander) ---
    st.markdown("### 📈 PASO 4: Visualización 2D Interactiva")
    
    if st.session_state.matriz_seleccionada_2d.empty:
        st.warning("⚠️ Seleccione y cargue una matriz en el Paso 2 para visualizar.")
    else:
        df_m = st.session_state.matriz_seleccionada_2d
        df_m['Val_Plot'] = calcular_variable_atmosferica(df_m, var_sel)
        y, z, v = df_m['Y'].values, df_m['Z'].values, df_m['Val_Plot'].values
        
        grid_y = np.linspace(y.min(), y.max(), 150)
        grid_z = np.linspace(z.min(), z.max(), 150)
        Gy, Gz = np.meshgrid(grid_y, grid_z)
        Gv = griddata((y, z), v, (Gy, Gz), method='cubic')
        
        fig = go.Figure()
        if render_type == "Contour Suavizado":
            fig.add_trace(go.Contour(x=grid_y, y=grid_z, z=Gv, colorscale='Jet', colorbar=dict(title=var_sel)))
        else:
            fig.add_trace(go.Heatmap(x=grid_y, y=grid_z, z=Gv, colorscale='Jet', colorbar=dict(title=var_sel)))
            
        fig.update_layout(title=f"Mapeo 2D: {var_sel}", xaxis_title="Y [mm]", yaxis_title="Z [mm]", height=700, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True, config={'modeBarButtonsToAdd': ['drawline', 'eraseshape']})

    st.markdown("---")
    
    # --- PASO 5 (Sin Expander) ---
    st.markdown("### 📊 PASO 5: Estabilidad del Ensayo (Condiciones Atmosféricas)")
    if st.session_state.archivos_2d_memoria:
        data_inf = []
        for n, df in st.session_state.archivos_2d_memoria.items():
            if 'rho_inf' in df.columns:
                d_inf = df[['Timestamp', 'rho_inf', 'V_inf', 'P_inf']].copy()
                d_inf['Origen'] = n
                data_inf.append(d_inf)
        if data_inf:
            df_inf = pd.concat(data_inf).drop_duplicates()
            st.line_chart(df_inf.set_index('Timestamp')[['V_inf']])
        else:
            st.info("No hay datos de variables en el infinito en las matrices procesadas.")
    else:
        st.info("Procese archivos en el Paso 1 que incluyan el Archivo Infinito para ver la estabilidad atmosférica.")

