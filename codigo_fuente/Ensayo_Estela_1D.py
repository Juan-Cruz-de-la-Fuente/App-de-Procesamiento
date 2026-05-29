import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import io
import zipfile
import random
from datetime import datetime
from codigo_fuente.Calculations_Core import (
    procesar_promedios, 
    crear_archivos_individuales_por_tiempo_y_posicion, 
    extraer_datos_para_grafico, 
    calcular_area_bajo_curva, 
    extraer_nombre_base_archivo, 
    calcular_posiciones_sensores
)
from codigo_fuente import Auth_Manager as auth

def crear_grafico_diferencia_areas(sub_archivo_a, sub_archivo_b, configuracion):
    """Crear gráfico mostrando la diferencia como UNA sola área"""
    z_a, presion_a = extraer_datos_para_grafico(sub_archivo_a, configuracion)
    z_b, presion_b = extraer_datos_para_grafico(sub_archivo_b, configuracion)
    
    if not z_a or not z_b or not presion_a or not presion_b:
        return None, 0
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=presion_a, y=z_a,
        mode='lines',
        name=f"{sub_archivo_a['archivo_fuente']} T{sub_archivo_a['tiempo']}s",
        line=dict(color='#08596C', width=2, dash='dot'),
        opacity=0.6,
        hovertemplate='<b>%{fullData.name}</b><br>Presión: %{x:.3f} Pa<br>Altura: %{y:.1f} mm<br><extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=presion_b, y=z_b,
        mode='lines',
        name=f"{sub_archivo_b['archivo_fuente']} T{sub_archivo_b['tiempo']}s",
        line=dict(color='#E74C3C', width=2, dash='dot'),
        opacity=0.6,
        hovertemplate='<b>%{fullData.name}</b><br>Presión: %{x:.3f} Pa<br>Altura: %{y:.1f} mm<br><extra></extra>'
    ))
    
    z_min = max(min(z_a), min(z_b))
    z_max = min(max(z_a), max(z_b))
    
    if z_max <= z_min:
        return fig, 0
        
    z_comun = np.linspace(z_min, z_max, 200)
    p_a_interp = np.interp(z_comun, z_a, presion_a)
    p_b_interp = np.interp(z_comun, z_b, presion_b)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([p_a_interp, p_b_interp[::-1]]),
        y=np.concatenate([z_comun, z_comun[::-1]]),
        fill='toself',
        fillcolor='rgba(155, 89, 182, 0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False,
        name='Diferencia'
    ))
    
    area_a = calcular_area_bajo_curva(z_a, presion_a)
    area_b = calcular_area_bajo_curva(z_b, presion_b)
    diferencia_area = area_a - area_b
    
    fig.update_layout(
        title=f"Diferencia de Perfiles: {sub_archivo_a['archivo_fuente']} vs {sub_archivo_b['archivo_fuente']}",
        xaxis_title="Presión [Pa]",
        yaxis_title="Altura Z [mm]",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=600
    )
    
    return fig, diferencia_area

def show_1d():
    st.markdown("# 📊 VISUALIZACIÓN DE ESTELA 1D - Análisis Unidimensional")
    st.markdown("Análisis de perfiles de presión concatenados con extracción automática de tiempo y coordenadas")

    if 'configuracion_1d_local' not in st.session_state: st.session_state.configuracion_1d_local = None
    if "datos_procesados_1d" not in st.session_state: st.session_state.datos_procesados_1d = {}
    if "sub_archivos_1d_memoria" not in st.session_state: st.session_state.sub_archivos_1d_memoria = {}
    if "perfiles_seleccionados_1d" not in st.session_state: st.session_state.perfiles_seleccionados_1d = []

    from codigo_fuente.Graficos_Comunes import mostrar_configuracion_sensores

    # --- CARGA Y CONFIGURACIÓN (EXPANDER ÚNICO) ---
    with st.expander("📥 CARGA Y CONFIGURACIÓN DE PERFILES NUEVOS", expanded=False):
        st.markdown("""
        <div class="section-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: white;">📥 PROCESAMIENTO DE ARCHIVOS CRUDOS</h3>
            <p style="color: #bbb; margin-bottom: 20px;">Configure el peine y cargue archivos CSV para procesar y guardar en la nube.</p>
        </div>
        """, unsafe_allow_html=True)

        conf = mostrar_configuracion_sensores("1d_local")
        if st.button("💾 CONFIRMAR CONFIGURACIÓN PARA PROCESAR", use_container_width=True):
            st.session_state.configuracion_1d_local = conf
            st.success("✅ Configuración de procesamiento lista.")

        st.markdown("---")
        up_1d = st.file_uploader("Arrastre sus archivos CSV 1D aquí", type=['csv'], accept_multiple_files=True, key="up_1d")
        
        if up_1d and st.session_state.configuracion_1d_local:
            for f in up_1d:
                if f.name not in st.session_state.datos_procesados_1d:
                    with st.spinner(f"🔨 Procesando {f.name}..."):
                        datos = procesar_promedios(f, st.session_state.configuracion_1d_local['orden'])
                        if datos is not None:
                            st.session_state.datos_procesados_1d[f.name] = datos
                            subs = crear_archivos_individuales_por_tiempo_y_posicion(datos, f.name)
                            st.session_state.sub_archivos_1d_memoria.update(subs)
            st.success(f"✅ {len(st.session_state.datos_procesados_1d)} archivos en memoria.")

        st.markdown("#### 🚀 Subir a Drive (1D)")
        opciones_1d = list(st.session_state.sub_archivos_1d_memoria.keys()) if st.session_state.sub_archivos_1d_memoria else ["No hay archivos"]
        sel_save = st.selectbox("Seleccionar sub-archivo:", opciones_1d)
        
        nombre_base_1d = "archivo_1d.csv"
        if st.session_state.sub_archivos_1d_memoria and sel_save in st.session_state.sub_archivos_1d_memoria:
            sub = st.session_state.sub_archivos_1d_memoria[sel_save]
            nombre_base_1d = sub.get('nombre_archivo', f"{sel_save}.csv")
            
        nombre_final_1d = st.text_input("Nombre del archivo a guardar:", value=nombre_base_1d)
        
        if st.button("🚀 SUBIR A DRIVE", use_container_width=True, type="primary", disabled=not st.session_state.sub_archivos_1d_memoria):
            if st.session_state.sub_archivos_1d_memoria and sel_save in st.session_state.sub_archivos_1d_memoria:
                sub = st.session_state.sub_archivos_1d_memoria[sel_save]
                csv_b = sub['datos'].to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')
                if auth.save_csv_1d(st.session_state.username, nombre_final_1d, csv_b):
                    st.success(f"✅ Guardado: {nombre_final_1d}")
                else:
                    st.error("Error al guardar.")

    st.markdown("---")

    # --- PASO 2 (Sin Expander) ---
    st.markdown("### 📥 PASO 2: Selección de Perfiles para Análisis")
    modo_carga = st.radio("Cargar perfiles desde:", ["🗄️ Base de Datos (Drive)", "🧠 Memoria de Sesión"], horizontal=True, key="modo_carga_1d")
    
    if modo_carga == "🗄️ Base de Datos (Drive)":
        try:
            files_drv = auth.get_user_files_1d(st.session_state.username)
        except:
            files_drv = []

        if not files_drv:
            st.info("No hay archivos en Drive.")
        else:
            sel_labels = st.multiselect("Seleccionar Perfiles de Drive:", files_drv)
            if st.button("📥 Cargar Perfiles al Visualizador", use_container_width=True):
                st.session_state.perfiles_seleccionados_1d = []
                for label in sel_labels:
                    csv_content = auth.get_csv_content_1d(st.session_state.username, label)
                    if csv_content:
                        df = pd.read_csv(io.StringIO(csv_content), sep=';', decimal=',')
                        st.session_state.perfiles_seleccionados_1d.append({'nombre': label, 'datos': df})
                st.success(f"✅ {len(st.session_state.perfiles_seleccionados_1d)} perfiles cargados y listos para visualizar.")
    else:
        if not st.session_state.sub_archivos_1d_memoria:
            st.warning("⚠️ No hay sub-archivos en la memoria de sesión. Procese archivos en el Paso 1 primero.")
        else:
            st.info("Funcionalidad de carga múltiple desde memoria en desarrollo. Por favor guarde en Drive y cargue desde allí para el análisis comparativo.")

    st.markdown("---")

    # --- PASO 3 (Sin Expander) ---
    st.markdown("### 🛠️ PASO 3: Configuración de Visualización")
    conf_vis = mostrar_configuracion_sensores("1d_vis")

    st.markdown("---")

    # --- PASO 4 (Sin Expander) ---
    st.markdown("### 📈 PASO 4: Visualización y Análisis de Perfiles")
    
    if not st.session_state.perfiles_seleccionados_1d:
        st.warning("⚠️ Seleccione y cargue perfiles en el Paso 2 para ver el gráfico.")
    else:
        fig = go.Figure()
        for perf in st.session_state.perfiles_seleccionados_1d:
            z, p = extraer_datos_para_grafico({'datos': perf['datos'], 'archivo_fuente': perf['nombre'], 'tiempo': 'N/A'}, conf_vis)
            if z and p:
                fig.add_trace(go.Scatter(x=p, y=z, mode='lines+markers', name=perf['nombre']))
        
        fig.update_layout(xaxis_title="Presión [Pa]", yaxis_title="Altura Z [mm]", height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)
        
        if len(st.session_state.perfiles_seleccionados_1d) >= 2:
            st.markdown("### 📊 Comparativa de Áreas")
            p1 = st.session_state.perfiles_seleccionados_1d[0]
            p2 = st.session_state.perfiles_seleccionados_1d[1]
            fig_diff, area = crear_grafico_diferencia_areas({'datos': p1['datos'], 'archivo_fuente': p1['nombre'], 'tiempo': 'N/A'}, 
                                                           {'datos': p2['datos'], 'archivo_fuente': p2['nombre'], 'tiempo': 'N/A'}, conf_vis)
            if fig_diff:
                st.plotly_chart(fig_diff, use_container_width=True)
                st.metric("Diferencia de Área (A-B)", f"{area:.4f}")
