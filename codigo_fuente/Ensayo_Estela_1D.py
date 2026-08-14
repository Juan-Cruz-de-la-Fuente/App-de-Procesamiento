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
                            st.session_state.perfiles_seleccionados_1d = [{'nombre': f"[Archivo Completo] {f.name}", 'datos': datos}]
            st.success(f"✅ {len(st.session_state.datos_procesados_1d)} archivos en memoria.")

        st.markdown("#### 🚀 Subir a Drive (1D)")
        opciones_1d = [f"[Archivo Completo] {k}" for k in st.session_state.datos_procesados_1d.keys()] + list(st.session_state.sub_archivos_1d_memoria.keys())
        if not opciones_1d:
            opciones_1d = ["No hay archivos cargados"]
            
        sel_save = st.selectbox("Seleccionar Archivo para guardar:", opciones_1d)
        
        tiempos = [0]
        df_target = None
        if st.session_state.datos_procesados_1d or st.session_state.sub_archivos_1d_memoria:
            if sel_save.startswith("[Archivo Completo] "):
                real_k = sel_save.replace("[Archivo Completo] ", "")
                df_target = st.session_state.datos_procesados_1d.get(real_k)
            elif sel_save in st.session_state.sub_archivos_1d_memoria:
                sub = st.session_state.sub_archivos_1d_memoria[sel_save]
                df_target = sub['datos']

            if df_target is not None and 'Tiempo_s' in df_target.columns:
                tiempos = sorted(df_target['Tiempo_s'].dropna().unique())
                
        t_sel = st.selectbox("Tiempo [s]:", tiempos, key="t_sel_save_1d")
        
        c1, c2 = st.columns(2)
        x_pos = c1.number_input("Posición X [mm]:", value=0.0, key="x_pos_1d_save")
        aoa = c2.number_input("AOA [°]:", value=0.0, key="aoa_1d_save")
        
        nombre_auto_1d = f"1D-X{int(x_pos)}-OAO{str(aoa).replace('-','neg')}-T{int(t_sel)}s.csv"
        nombre_final_1d = st.text_input("Nombre del archivo a guardar:", value=nombre_auto_1d, key="nombre_final_1d_save")
        
        if st.button("🚀 SUBIR COMPLETO A DRIVE (1D)", use_container_width=True, type="primary", disabled=df_target is None):
            if df_target is not None:
                csv_b = df_target.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')
                if auth.save_csv_1d(st.session_state.username, nombre_final_1d, csv_b):
                    st.success(f"✅ Archivo completo guardado en Drive: {nombre_final_1d}")
                else:
                    st.error("Error al guardar en Drive.")

    st.markdown("---")

    # --- PASO 2 (Sin Expander) ---
    st.markdown("### 📥 PASO 2: Selección de Perfiles para Análisis")
    modo_carga = st.radio("Cargar perfiles desde:", ["🧠 Memoria de Sesión", "🗄️ Base de Datos (Drive)"], horizontal=True, key="modo_carga_1d")
    
    if modo_carga == "🧠 Memoria de Sesión":
        opciones_mem = [f"[Archivo Completo] {k}" for k in st.session_state.datos_procesados_1d.keys()] + list(st.session_state.sub_archivos_1d_memoria.keys())
        if not opciones_mem:
            st.warning("⚠️ No hay archivos en la memoria de sesión. Procese archivos en el Paso 1 primero.")
        else:
            default_mem = [f"[Archivo Completo] {k}" for k in st.session_state.datos_procesados_1d.keys()]
            sel_labels_mem = st.multiselect("Seleccionar Perfiles de Memoria de Sesión:", opciones_mem, default=default_mem if default_mem else None, key="sel_perfiles_1d_mem_ui")
            
            if 'last_sel_perfiles_1d_mem' not in st.session_state:
                st.session_state.last_sel_perfiles_1d_mem = []
                
            if sel_labels_mem != st.session_state.last_sel_perfiles_1d_mem:
                st.session_state.perfiles_seleccionados_1d = []
                for label in sel_labels_mem:
                    if label.startswith("[Archivo Completo] "):
                        real_k = label.replace("[Archivo Completo] ", "")
                        df = st.session_state.datos_procesados_1d[real_k].copy()
                        st.session_state.perfiles_seleccionados_1d.append({'nombre': label, 'datos': df})
                    elif label in st.session_state.sub_archivos_1d_memoria:
                        sub = st.session_state.sub_archivos_1d_memoria[label]
                        st.session_state.perfiles_seleccionados_1d.append({'nombre': label, 'datos': sub['datos'].copy()})
                st.session_state.last_sel_perfiles_1d_mem = sel_labels_mem
                st.success(f"✅ {len(st.session_state.perfiles_seleccionados_1d)} perfiles cargados desde Memoria.")
                st.rerun()
    else:
        try:
            files_drv = auth.get_user_files_1d(st.session_state.username)
        except:
            files_drv = []

        if not files_drv:
            st.info("No se encontraron perfiles guardados en Drive.")
        else:
            sel_labels = st.multiselect("Seleccionar Perfiles de Drive:", files_drv, key="sel_perfiles_1d_ui")
            
            # Carga automática reactiva
            if 'last_sel_perfiles_1d' not in st.session_state:
                st.session_state.last_sel_perfiles_1d = []
                
            if sel_labels != st.session_state.last_sel_perfiles_1d:
                st.session_state.perfiles_seleccionados_1d = []
                with st.spinner("Descargando perfiles seleccionados..."):
                    for label in sel_labels:
                        csv_content = auth.get_csv_content_1d(st.session_state.username, label)
                        if csv_content:
                            df = pd.read_csv(io.StringIO(csv_content), sep=';', decimal=',')
                            # Asegurar columnas Pos_Y_Traverser y Pos_Z_Base
                            if 'Pos_Y_Traverser' not in df.columns and 'Archivo' in df.columns:
                                coords = df['Archivo'].apply(extraer_tiempo_y_coordenadas_YZ)
                                df['Tiempo_s'] = [c[0] for c in coords]
                                df['Pos_Y_Traverser'] = [c[1] for c in coords]
                                df['Pos_Z_Base'] = [c[2] for c in coords]
                            elif 'Pos_Y_Traverser' not in df.columns:
                                t_v, y_v, z_v = extraer_tiempo_y_coordenadas_YZ(label)
                                df['Pos_Y_Traverser'] = y_v if y_v is not None else 0
                                df['Pos_Z_Base'] = z_v if z_v is not None else 0

                            st.session_state.perfiles_seleccionados_1d.append({'nombre': label, 'datos': df})
                    st.session_state.last_sel_perfiles_1d = sel_labels
                st.success(f"✅ {len(st.session_state.perfiles_seleccionados_1d)} perfiles cargados desde Drive.")
                st.rerun()

    st.markdown("---")

    # --- PASO 3 (Sin Expander) ---
    st.markdown("### 🛠️ PASO 3: Configuración de Visualización")
    conf_vis = mostrar_configuracion_sensores("1d_vis")

    # Detección de secciones Y disponibles
    y_secciones_disponibles = []
    has_pos_y = False
    if st.session_state.perfiles_seleccionados_1d:
        for perf in st.session_state.perfiles_seleccionados_1d:
            if 'datos' in perf and 'Pos_Y_Traverser' in perf['datos'].columns:
                y_vals = perf['datos']['Pos_Y_Traverser'].dropna().unique()
                for y_v in y_vals:
                    if y_v not in y_secciones_disponibles:
                        y_secciones_disponibles.append(y_v)
        y_secciones_disponibles = sorted(y_secciones_disponibles)
        if len(y_secciones_disponibles) > 0:
            has_pos_y = True

    filas_opciones = []
    sel_y_seccion = "Todas las Secciones Y"

    def _format_y_val(val):
        try:
            v_num = float(val)
            return f"{int(v_num)}" if v_num.is_integer() else f"{v_num:.1f}"
        except:
            return str(val)

    if has_pos_y:
        col_y1, col_y2 = st.columns([1, 1])
        with col_y1:
            opciones_y = ["Todas las Secciones Y"] + [f"Y = {_format_y_val(y)}" for y in y_secciones_disponibles]
            sel_y_seccion = st.selectbox("🎯 Filtrar por Sección Y (Plano XY):", opciones_y, key="sel_y_seccion_1d")
    else:
        # Fallback por mediciones/filas si Pos_Y_Traverser no diferenció distintas secciones
        if st.session_state.perfiles_seleccionados_1d:
            for perf in st.session_state.perfiles_seleccionados_1d:
                df_p = perf['datos']
                for i_row, row in df_p.iterrows():
                    arc_name = row.get('Archivo', f"Fila_{i_row+1}")
                    sub_label = f"Y = {_format_y_val(row.get('Pos_Y_Traverser', i_row))}"
                    if len(st.session_state.perfiles_seleccionados_1d) > 1:
                        sub_label += f" ({perf['nombre']})"
                    filas_opciones.append({'label': sub_label, 'perf_nombre': perf['nombre'], 'row_idx': i_row})

            col_y1, col_y2 = st.columns([1, 1])
            with col_y1:
                opts_f = ["Todas las Mediciones / Filas"] + [f['label'] for f in filas_opciones]
                sel_y_seccion = st.selectbox("🎯 Filtrar por Medición / Sección Y (Plano XY):", opts_f, key="sel_y_seccion_1d_fallback")

    st.markdown("---")

    # --- PASO 4 (Sin Expander) ---
    st.markdown("### 📈 PASO 4: Visualización y Análisis de Perfiles")
    
    if not st.session_state.perfiles_seleccionados_1d:
        st.warning("⚠️ Seleccione y cargue perfiles en el Paso 2 para ver el gráfico.")
    else:
        fig = go.Figure()
        trazas_creadas = []
        num_perfiles = len(st.session_state.perfiles_seleccionados_1d)

        if has_pos_y:
            y_target = None
            if sel_y_seccion != "Todas las Secciones Y":
                try:
                    y_target = float(sel_y_seccion.replace("Y = ", "").strip())
                except:
                    y_target = None

            for perf in st.session_state.perfiles_seleccionados_1d:
                df_perf = perf['datos']
                y_vals_in_perf = df_perf['Pos_Y_Traverser'].dropna().unique() if 'Pos_Y_Traverser' in df_perf.columns else [None]
                
                if y_target is not None:
                    z, p = extraer_datos_para_grafico({'datos': df_perf}, conf_vis, y_filtro=y_target)
                    if z and p:
                        label_trace = f"Y = {_format_y_val(y_target)}"
                        if num_perfiles > 1:
                            label_trace += f" ({perf['nombre']})"
                        fig.add_trace(go.Scatter(x=p, y=z, mode='lines+markers', name=label_trace))
                        trazas_creadas.append({'nombre': label_trace, 'z': z, 'p': p, 'sub': {'datos': df_perf, 'archivo_fuente': perf['nombre'], 'tiempo': 'N/A'}})
                else:
                    for y_val in y_vals_in_perf:
                        z, p = extraer_datos_para_grafico({'datos': df_perf}, conf_vis, y_filtro=y_val)
                        if z and p:
                            tag_y = f"Y = {_format_y_val(y_val)}" if y_val is not None else perf['nombre']
                            label_trace = tag_y if num_perfiles == 1 else f"{tag_y} ({perf['nombre']})"
                            fig.add_trace(go.Scatter(x=p, y=z, mode='lines+markers', name=label_trace))
                            trazas_creadas.append({'nombre': label_trace, 'z': z, 'p': p, 'sub': {'datos': df_perf, 'archivo_fuente': perf['nombre'], 'tiempo': 'N/A'}})
        else:
            for perf in st.session_state.perfiles_seleccionados_1d:
                df_perf = perf['datos']
                for i_row, row in df_perf.iterrows():
                    row_y = row.get('Pos_Y_Traverser', i_row)
                    sub_label = f"Y = {_format_y_val(row_y)}"
                    if num_perfiles > 1:
                        sub_label += f" ({perf['nombre']})"
                    if sel_y_seccion == "Todas las Mediciones / Filas" or sel_y_seccion == sub_label:
                        z, p = extraer_datos_para_grafico({'datos': df_perf}, conf_vis, fila_index=i_row)
                        if z and p:
                            fig.add_trace(go.Scatter(x=p, y=z, mode='lines+markers', name=sub_label))
                            trazas_creadas.append({'nombre': sub_label, 'z': z, 'p': p, 'sub': {'datos': df_perf.iloc[[i_row]], 'archivo_fuente': perf['nombre'], 'tiempo': 'N/A'}})

        fig.update_layout(
            title="Perfil de Presiones a lo largo del Eje Z",
            xaxis_title="Presión [Pa]", 
            yaxis_title="Altura Z [mm]", 
            height=600, 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if len(trazas_creadas) >= 2:
            st.markdown("### 📊 Comparativa de Áreas")
            t1 = trazas_creadas[0]
            t2 = trazas_creadas[1]
            fig_diff, area = crear_grafico_diferencia_areas(t1['sub'], t2['sub'], conf_vis)
            if fig_diff:
                st.plotly_chart(fig_diff, use_container_width=True)
                st.metric("Diferencia de Área (A-B)", f"{area:.4f}")

