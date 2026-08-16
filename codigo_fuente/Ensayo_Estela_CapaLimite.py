import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import io
from codigo_fuente.Calculations_Core import (
    procesar_promedios, 
    crear_archivos_individuales_por_tiempo_y_posicion, 
    extraer_tiempo_y_coordenadas_YZ,
    extraer_datos_para_grafico
)
from codigo_fuente import Auth_Manager as auth
from codigo_fuente.Graficos_Comunes import mostrar_configuracion_sensores

def show_capa_limite():
    st.markdown("# 🌬️ CAPA LÍMITE - Análisis de Velocidades")
    st.markdown("Análisis de perfiles de presión para obtener velocidad asumiendo $P_s = \max(P_t)$")

    if 'configuracion_cl_local' not in st.session_state: st.session_state.configuracion_cl_local = None
    if "datos_procesados_cl" not in st.session_state: st.session_state.datos_procesados_cl = {}
    if "sub_archivos_cl_memoria" not in st.session_state: st.session_state.sub_archivos_cl_memoria = {}
    if "perfiles_seleccionados_cl" not in st.session_state: st.session_state.perfiles_seleccionados_cl = []

    # Configuración Global
    col_dens, _ = st.columns([1, 2])
    densidad = col_dens.number_input("Densidad del aire (ρ) [kg/m³]:", value=1.225, step=0.01)

    # --- CARGA Y CONFIGURACIÓN ---
    with st.expander("📥 CARGA Y CONFIGURACIÓN DE PERFILES", expanded=False):
        st.markdown("### 📥 PROCESAMIENTO DE ARCHIVOS CRUDOS")
        conf = mostrar_configuracion_sensores("cl_local")
        if st.button("💾 CONFIRMAR CONFIGURACIÓN PARA PROCESAR", use_container_width=True, key="btn_conf_cl"):
            st.session_state.configuracion_cl_local = conf
            st.success("✅ Configuración de procesamiento lista.")

        st.markdown("---")
        up_cl = st.file_uploader("Arrastre sus archivos CSV aquí", type=['csv'], accept_multiple_files=True, key="up_cl")
        
        if up_cl and st.session_state.configuracion_cl_local:
            for f in up_cl:
                if f.name not in st.session_state.datos_procesados_cl:
                    with st.spinner(f"🔨 Procesando {f.name}..."):
                        datos = procesar_promedios(f, st.session_state.configuracion_cl_local['orden'])
                        if datos is not None:
                            st.session_state.datos_procesados_cl[f.name] = datos
                            subs = crear_archivos_individuales_por_tiempo_y_posicion(datos, f.name)
                            st.session_state.sub_archivos_cl_memoria.update(subs)
                            st.session_state.perfiles_seleccionados_cl = [{'nombre': f"[Archivo Completo] {f.name}", 'datos': datos}]
            st.success(f"✅ {len(st.session_state.datos_procesados_cl)} archivos en memoria.")

        st.markdown("#### 🚀 Subir a Drive (Capa Límite)")
        opciones_cl = [f"[Archivo Completo] {k}" for k in st.session_state.datos_procesados_cl.keys()] + list(st.session_state.sub_archivos_cl_memoria.keys())
        if not opciones_cl: opciones_cl = ["No hay archivos cargados"]
            
        sel_save = st.selectbox("Seleccionar Archivo para guardar:", opciones_cl, key="sel_save_cl")
        
        df_target = None
        if st.session_state.datos_procesados_cl or st.session_state.sub_archivos_cl_memoria:
            if sel_save.startswith("[Archivo Completo] "):
                real_k = sel_save.replace("[Archivo Completo] ", "")
                df_target = st.session_state.datos_procesados_cl.get(real_k)
            elif sel_save in st.session_state.sub_archivos_cl_memoria:
                sub = st.session_state.sub_archivos_cl_memoria[sel_save]
                df_target = sub['datos']
                
        nombre_final_cl = st.text_input("Nombre del archivo a guardar:", value="Capa_Limite_Guardado.csv", key="nombre_final_cl_save")
        
        if st.button("🚀 SUBIR COMPLETO A DRIVE", use_container_width=True, type="primary", disabled=df_target is None, key="btn_save_cl"):
            if df_target is not None:
                csv_b = df_target.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')
                if auth.save_csv_1d(st.session_state.username, nombre_final_cl, csv_b):
                    st.success(f"✅ Archivo completo guardado en Drive: {nombre_final_cl}")
                else:
                    st.error("Error al guardar en Drive.")

    st.markdown("---")

    # --- PASO 2: Selección ---
    st.markdown("### 📥 PASO 2: Selección de Perfiles")
    modo_carga = st.radio("Cargar perfiles desde:", ["🧠 Memoria de Sesión", "🗄️ Base de Datos (Drive)"], horizontal=True, key="modo_carga_cl")
    
    if modo_carga == "🧠 Memoria de Sesión":
        opciones_mem = [f"[Archivo Completo] {k}" for k in st.session_state.datos_procesados_cl.keys()] + list(st.session_state.sub_archivos_cl_memoria.keys())
        if not opciones_mem:
            st.warning("⚠️ No hay archivos en la memoria de sesión.")
        else:
            default_mem = [f"[Archivo Completo] {k}" for k in st.session_state.datos_procesados_cl.keys()]
            sel_labels_mem = st.multiselect("Seleccionar Perfiles de Memoria de Sesión:", opciones_mem, default=default_mem if default_mem else None, key="sel_perfiles_cl_mem_ui")
            
            if 'last_sel_perfiles_cl_mem' not in st.session_state: st.session_state.last_sel_perfiles_cl_mem = []
                
            if sel_labels_mem != st.session_state.last_sel_perfiles_cl_mem:
                st.session_state.perfiles_seleccionados_cl = []
                for label in sel_labels_mem:
                    if label.startswith("[Archivo Completo] "):
                        real_k = label.replace("[Archivo Completo] ", "")
                        df = st.session_state.datos_procesados_cl[real_k].copy()
                        st.session_state.perfiles_seleccionados_cl.append({'nombre': label, 'datos': df})
                    elif label in st.session_state.sub_archivos_cl_memoria:
                        sub = st.session_state.sub_archivos_cl_memoria[label]
                        st.session_state.perfiles_seleccionados_cl.append({'nombre': label, 'datos': sub['datos'].copy()})
                st.session_state.last_sel_perfiles_cl_mem = sel_labels_mem
                st.rerun()
    else:
        try:
            files_drv = auth.get_user_files_1d(st.session_state.username)
        except:
            files_drv = []

        if not files_drv:
            st.info("No se encontraron perfiles guardados en Drive.")
        else:
            sel_labels = st.multiselect("Seleccionar Perfiles de Drive:", files_drv, key="sel_perfiles_cl_ui")
            if 'last_sel_perfiles_cl' not in st.session_state: st.session_state.last_sel_perfiles_cl = []
                
            if sel_labels != st.session_state.last_sel_perfiles_cl:
                st.session_state.perfiles_seleccionados_cl = []
                with st.spinner("Descargando perfiles..."):
                    for label in sel_labels:
                        csv_content = auth.get_csv_content_1d(st.session_state.username, label)
                        if csv_content:
                            df = pd.read_csv(io.StringIO(csv_content), sep=';', decimal=',')
                            st.session_state.perfiles_seleccionados_cl.append({'nombre': label, 'datos': df})
                    st.session_state.last_sel_perfiles_cl = sel_labels
                st.rerun()

    st.markdown("---")

    # --- PASO 3: Configuración Vis ---
    st.markdown("### 🛠️ PASO 3: Configuración del Peine para Gráfico")
    conf_vis = mostrar_configuracion_sensores("cl_vis")

    # --- PASO 4: Gráfico ---
    st.markdown("### 📈 PASO 4: Visualización de Velocidad")
    
    if not st.session_state.perfiles_seleccionados_cl:
        st.warning("⚠️ Seleccione y cargue perfiles.")
    else:
        fig = go.Figure()

        for perf in st.session_state.perfiles_seleccionados_cl:
            df_perf = perf['datos']
            for i_row, row in df_perf.iterrows():
                # Extraer presiones y alturas
                z, p = extraer_datos_para_grafico({'datos': df_perf}, conf_vis, fila_index=i_row)
                if z and p:
                    # Encontrar Presión Estática (Máximo valor de presión / el menos negativo)
                    P_s = max(p)
                    
                    # Calcular Velocidad: V = sqrt(2 * |P - P_s| / rho)
                    velocidades = [np.sqrt(2 * abs(val - P_s) / densidad) for val in p]
                    
                    sub_label = f"Fila {i_row+1} ({perf['nombre']}) [Ps = {P_s:.2f}]"
                    fig.add_trace(go.Scatter(x=velocidades, y=z, mode='lines+markers', name=sub_label))

        fig.update_layout(
            title="Perfil de Velocidades en la Capa Límite",
            xaxis_title="Velocidad [m/s]", 
            yaxis_title="Altura Z [mm]", 
            height=600, 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)
