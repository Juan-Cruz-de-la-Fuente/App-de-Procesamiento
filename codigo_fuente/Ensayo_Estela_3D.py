import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import io
from datetime import datetime
from codigo_fuente.Calculations_Core import (
    procesar_promedios,
    obtener_numero_sensor_desde_columna,
    calcular_altura_absoluta_z,
    crear_sub_archivos_3d_por_tiempo_y_posicion
)
from codigo_fuente.Graficos_Comunes import (
    crear_superficie_delaunay_3d,
    crear_superficie_diferencia_delaunay_3d
)
from codigo_fuente import Auth_Manager as auth

def show_3d():
    st.markdown("# 🌪️ VISUALIZACIÓN DE ESTELA 3D - Análisis Tridimensional")
    st.markdown("Análisis 3D con superficie interactiva de presiones")

    if 'configuracion_3d_local' not in st.session_state: st.session_state.configuracion_3d_local = None
    if 'archivos_3d_memoria' not in st.session_state: st.session_state.archivos_3d_memoria = {}
    if 'superficie_seleccionada_3d' not in st.session_state: st.session_state.superficie_seleccionada_3d = None

    from codigo_fuente.Graficos_Comunes import mostrar_configuracion_sensores, crear_superficie_delaunay_3d, crear_superficie_diferencia_delaunay_3d

    # --- CARGA Y CONFIGURACIÓN (EXPANDER ÚNICO) ---
    with st.expander("📥 CARGA Y CONFIGURACIÓN DE SUPERFICIES NUEVAS", expanded=True):
        st.markdown("""
        <div class="section-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: white;">📥 PROCESAMIENTO DE ARCHIVOS CRUDOS</h3>
            <p style="color: #bbb; margin-bottom: 20px;">Configure el peine y cargue archivos CSV para procesar y guardar en la nube.</p>
        </div>
        """, unsafe_allow_html=True)

        conf = mostrar_configuracion_sensores("3d_local")
        if st.button("💾 CONFIRMAR CONFIGURACIÓN PARA PROCESAR", use_container_width=True):
            st.session_state.configuracion_3d_local = conf
            st.success("✅ Configuración de procesamiento lista.")

        st.markdown("---")
        up_3d = st.file_uploader("Arrastre archivos CSV 3D aquí", type=['csv'], accept_multiple_files=True, key="up_3d")
        inf_3d = st.file_uploader("🔗 Archivo Infinito (Opcional)", type=['txt', 'csv'], key="inf_3d")
        
        if up_3d and st.session_state.configuracion_3d_local:
            for f in up_3d:
                name = f.name.replace('.csv', '').replace('incertidumbre_', '')
                if name not in st.session_state.archivos_3d_memoria:
                    with st.spinner(f"🔨 Procesando {name}..."):
                        datos = procesar_promedios(f, st.session_state.configuracion_3d_local['orden'], inf_3d)
                        if datos is not None:
                            st.session_state.archivos_3d_memoria[name] = datos
            st.success(f"✅ {len(st.session_state.archivos_3d_memoria)} archivos en memoria.")

        st.markdown("#### Guardar en Drive (3D)")
        op_3d = list(st.session_state.archivos_3d_memoria.keys()) if st.session_state.archivos_3d_memoria else ["No hay archivos"]
        arc_sel = st.selectbox("Seleccionar Archivo:", op_3d)
        
        tiempos = [0]
        if st.session_state.archivos_3d_memoria and arc_sel in st.session_state.archivos_3d_memoria:
            df_arc = st.session_state.archivos_3d_memoria[arc_sel]
            tiempos = sorted(df_arc['Tiempo_s'].dropna().unique())
            
        t_sel = st.selectbox("Tiempo [s]:", tiempos)
        
        c1, c2 = st.columns(2)
        aoa = c1.number_input("AOA [°]:", value=0.0)
        x_pos = c2.number_input("Posición X [mm]:", value=0.0)
        nombre_auto = f"3D-X{int(x_pos)}-OAO{str(aoa).replace('-','neg')}-T{int(t_sel)}s"
        nombre_final = st.text_input("Nombre del archivo a guardar:", value=nombre_auto)
        
        if st.button("🚀 SUBIR SUPERFICIE A DRIVE (3D)", use_container_width=True, type="primary", disabled=not st.session_state.archivos_3d_memoria):
            if st.session_state.archivos_3d_memoria and arc_sel in st.session_state.archivos_3d_memoria:
                df_arc = st.session_state.archivos_3d_memoria[arc_sel]
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
                            z_r = calcular_altura_absoluta_z(num, z_b, st.session_state.configuracion_3d_local['distancia_toma_12'], st.session_state.configuracion_3d_local['distancia_entre_tomas'], 12, st.session_state.configuracion_3d_local['orden'])
                            res.append({'Y': y_t, 'Z': z_r, 'Presion': val})
                
                df_surf_save = pd.DataFrame(res)
                if not df_surf_save.empty:
                    json_s = df_surf_save.to_json(orient='records')
                    if auth.save_surface_data(st.session_state.username, nombre_final, json_s):
                        st.success(f"✅ Guardado en Drive: {nombre_final}")
                    else:
                        st.error("Error al guardar.")

    st.markdown("---")

    # --- PASO 2 (Sin Expander) ---
    st.markdown("### 📥 PASO 2: Selección de Superficies para Análisis")
    modo_carga = st.radio("Cargar superficies desde:", ["🗄️ Base de Datos (Drive)", "🧠 Memoria de Sesión"], horizontal=True, key="modo_carga_3d")

    if modo_carga == "🗄️ Base de Datos (Drive)":
        try:
            surfaces_drv = auth.get_user_surfaces(st.session_state.username)
        except:
            surfaces_drv = []

        if not surfaces_drv:
            st.info("No hay superficies en Drive.")
        else:
            dict_drv = {f"{s[1]} ({s[3][:10] if s[3] else ''})": s for s in surfaces_drv}
            sel_drv = st.selectbox("Seleccionar Superficie de Drive:", ["-- Seleccionar --"] + list(dict_drv.keys()))
            if sel_drv != "-- Seleccionar --":
                if st.button("📥 Cargar Superficie al Visualizador", use_container_width=True):
                    with st.spinner("Descargando superficie seleccionada..."):
                        s_data = list(dict_drv[sel_drv])
                        if not s_data[4]:
                            s_data[4] = auth.get_surface_data_string(s_data[0])
                        st.session_state.superficie_seleccionada_3d = {
                            'id': s_data[0],
                            'nombre': s_data[1],
                            'datos': pd.read_json(io.StringIO(s_data[4]))
                        }
                    st.success(f"✅ Superficie cargada y lista para visualizar.")
    else:
        if not st.session_state.archivos_3d_memoria:
            st.warning("⚠️ No hay superficies en la memoria de sesión. Procese archivos en el Paso 1 primero.")
        else:
            arc_mem_sel = st.selectbox("Seleccionar Superficie en Memoria:", list(st.session_state.archivos_3d_memoria.keys()))
            if st.button("📥 Cargar Superficie al Visualizador", use_container_width=True):
                df_arc = st.session_state.archivos_3d_memoria[arc_mem_sel]
                tiempos = sorted(df_arc['Tiempo_s'].dropna().unique())
                t_sel = st.selectbox("Confirmar Tiempo [s] a cargar:", tiempos)
                
                # We need to process it to Z coordinates like in Paso 1 to plot it directly
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
                            z_r = calcular_altura_absoluta_z(num, z_b, st.session_state.configuracion_3d_local['distancia_toma_12'], st.session_state.configuracion_3d_local['distancia_entre_tomas'], 12, st.session_state.configuracion_3d_local['orden'])
                            res.append({'Y': y_t, 'Z': z_r, 'Presion': val})
                
                if res:
                    st.session_state.superficie_seleccionada_3d = {
                        'id': None,
                        'nombre': f"{arc_mem_sel}_T{int(t_sel)}",
                        'datos': pd.DataFrame(res)
                    }
                    st.success("✅ Superficie de memoria procesada y cargada.")
                else:
                    st.error("No se pudieron procesar los datos para visualizar.")

    st.markdown("---")

    # --- PASO 3 (Sin Expander) ---
    st.markdown("### 🎨 PASO 3: Opciones de Visualización")
    c_opt1, c_opt2 = st.columns(2)
    with c_opt1:
        var_3d = st.selectbox("Variable a graficar:", ["Presión Total", "rho_inf", "V_inf"], key="var_3d_viz")
    with c_opt2:
        st.write("Visibilidad")
        show_pts = st.checkbox("Mostrar puntos de interpolación", value=True)

    st.markdown("---")

    # --- PASO 4 (Sin Expander) ---
    st.markdown("### 📈 PASO 4: Visualización 3D Interactiva")
    
    if not st.session_state.superficie_seleccionada_3d:
        st.warning("⚠️ Seleccione y cargue una superficie en el Paso 2 para visualizar.")
    else:
        s_curr = st.session_state.superficie_seleccionada_3d
        mock_conf = {'distancia_toma_12': 0, 'distancia_entre_tomas': 0, 'orden': 'asc'}
        fig = crear_superficie_delaunay_3d(s_curr['datos'], mock_conf, s_curr['nombre'], mostrar_puntos=show_pts, variable=var_3d)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- PASO 5 (Sin Expander) ---
    st.markdown("### ➖ PASO 5: Comparar con otra superficie (Diferencia)")
    
    try:
        surfaces_drv = auth.get_user_surfaces(st.session_state.username)
    except:
        surfaces_drv = []

    if st.session_state.superficie_seleccionada_3d and len(surfaces_drv) > 1:
        opts_diff = [s[1] for s in surfaces_drv if s[1] != st.session_state.superficie_seleccionada_3d['nombre']]
        sel_b = st.selectbox("Restar superficie B (Desde Drive):", opts_diff)
        if st.button("Calcular Diferencia A - B", use_container_width=True):
            s_b_info = next(s for s in surfaces_drv if s[1] == sel_b)
            df_b = pd.read_json(io.StringIO(s_b_info[4]))
            mock_conf = {'distancia_toma_12': 0, 'distancia_entre_tomas': 0, 'orden': 'asc'}
            fig_diff = crear_superficie_diferencia_delaunay_3d(st.session_state.superficie_seleccionada_3d['datos'], df_b, st.session_state.superficie_seleccionada_3d['nombre'], sel_b, mock_conf)
            if fig_diff:
                st.plotly_chart(fig_diff, use_container_width=True)
    else:
        st.info("Cargue una superficie principal y asegúrese de tener al menos otra guardada en Drive para comparar.")
