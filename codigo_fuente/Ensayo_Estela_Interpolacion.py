import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import io
import itertools
import random
from scipy.interpolate import griddata

from codigo_fuente.Calculations_Core import (
    procesar_promedios,
    obtener_numero_sensor_desde_columna,
    calcular_altura_absoluta_z
)
from codigo_fuente import Auth_Manager as auth
from codigo_fuente import Drive_Connection as _dapi
from codigo_fuente.Graficos_Comunes import mostrar_configuracion_sensores

def show_interpolacion():
    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            📊 ANÁLISIS DE INTERPOLACIÖN
            </h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Convergencia Angular de Planos de Estela
            </h2>
        </div>
    """, unsafe_allow_html=True)

    if 'results_interp' not in st.session_state: st.session_state.results_interp = None
    if 'interp_planos_memoria' not in st.session_state: st.session_state.interp_planos_memoria = []
    
    # --- PASO 1 (Sin Expander) ---
    st.markdown("### 📥 PASO 1: Origen de Datos (Carga de Planos)")
    modo_carga = st.radio(
        "Seleccionar origen:",
        ["🗄️ Base de Datos", "📂 Google Drive"],
        horizontal=True, key="modo_carga_interp_v2"
    )

    if modo_carga == "🗄️ Base de Datos":
        try:
            mis_planos_db = auth.get_user_surfaces_4d(st.session_state.username)
        except:
            mis_planos_db = []
            
        if not mis_planos_db:
            st.info("No hay planos guardados en la base de datos 4D.")
        else:
            estaciones_db = sorted(list(set(p[2] for p in mis_planos_db)))
            x_est_db = st.selectbox("Estación (X) [mm]:", estaciones_db, key="x_db_interp")
            planos_db_filtrados = [p for p in mis_planos_db if p[2] == x_est_db]
            
            if st.button("📥 Cargar Estación desde DB", use_container_width=True):
                nuevos_planos = []
                for p in planos_db_filtrados:
                    m = re.search(r'OAO(neg)?(\d+(?:[.,]\d+)?)', str(p[1]), re.IGNORECASE)
                    aoa_val = ((-1 if m.group(1) else 1) * float(str(m.group(2)).replace(',', '.'))) if m else 0.0
                    json_str = p[4] if p[4] else auth.get_surface_data_string(p[0])
                    nuevos_planos.append({'name': p[1], 'aoa': aoa_val, 'json': json_str, 'x': p[2]})
                st.session_state.interp_planos_memoria = nuevos_planos
                st.success(f"✅ Cargados {len(nuevos_planos)} planos.")

    else:
        st.markdown("#### 📂 Navegador de Drive")
        if 'drive_interp_folder_id' not in st.session_state:
            st.session_state.drive_interp_folder_id = None
            st.session_state.drive_interp_path = []

        if st.session_state.drive_interp_folder_id is None:
            with st.spinner("Conectando..."):
                f4d_id = _dapi.get_folder_4d(st.session_state.username)
                if f4d_id:
                    st.session_state.drive_interp_folder_id = f4d_id
                    st.session_state.drive_interp_path = [(f4d_id, "📁 Carpeta 4D")]

        if st.session_state.drive_interp_path:
            path_str = " / ".join([p[1] for p in st.session_state.drive_interp_path])
            st.caption(f"📍 {path_str}")
        
        if len(st.session_state.drive_interp_path) > 1:
            if st.button("⬅️ Subir nivel"):
                st.session_state.drive_interp_path.pop()
                st.session_state.drive_interp_folder_id = st.session_state.drive_interp_path[-1][0]
                st.rerun()

        if st.session_state.drive_interp_folder_id:
            items = _dapi.list_folder_contents(st.session_state.drive_interp_folder_id)
            if items:
                carpetas = [i for i in items if i['mimeType'] == 'application/vnd.google-apps.folder']
                archivos_csv = [i for i in items if i['name'].lower().endswith('.csv')]
                
                if carpetas:
                    c_sel = st.selectbox("📁 Carpetas:", ["-- Seleccionar --"] + [f"{c['name']}" for c in carpetas])
                    if c_sel != "-- Seleccionar --":
                        c_id = next(c['id'] for c in carpetas if c['name'] == c_sel)
                        st.session_state.drive_interp_path.append((c_id, c_sel))
                        st.session_state.drive_interp_folder_id = c_id
                        st.rerun()

                if archivos_csv:
                    sel_files_drv = st.multiselect("📄 Archivos CSV:", [a['name'] for a in archivos_csv])
                    if sel_files_drv:
                        config_interp = mostrar_configuracion_sensores("interp_drv")
                        x_pos_drv = st.number_input("Posición X [mm]:", value=0.0)
                        if st.button("⚙️ Procesar e Importar", type="primary", use_container_width=True):
                            nuevos_planos_drv = []
                            for f_name in sel_files_drv:
                                f_id = next(a['id'] for a in archivos_csv if a['name'] == f_name)
                                content = _dapi.download_file(f_id)
                                if content:
                                    try:
                                        df_p = procesar_promedios(io.BytesIO(content), config_interp.get('orden','asc'))
                                        if df_p is not None:
                                            m = re.search(r'OAO(neg)?(\d+(?:[.,]\d+)?)', f_name, re.IGNORECASE)
                                            aoa_v = ((-1 if m.group(1) else 1) * float(str(m.group(2)).replace(',', '.'))) if m else 0.0
                                            res_p = []
                                            for _, row in df_p.iterrows():
                                                y_t, z_b = row.get('Pos_Y_Traverser'), row.get('Pos_Z_Base')
                                                for col in df_p.columns:
                                                    ns = obtener_numero_sensor_desde_columna(col)
                                                    if ns is not None:
                                                        vp = row[col]
                                                        if pd.isna(vp): continue
                                                        zr = calcular_altura_absoluta_z(ns, z_b, config_interp.get('distancia_toma_12',-120), config_interp.get('distancia_entre_tomas',10.0), 12, config_interp.get('orden','asc'))
                                                        res_p.append({'Y': y_t, 'Z': zr, 'Presion': vp})
                                            df_final = pd.DataFrame(res_p)
                                            nuevos_planos_drv.append({'name': f_name, 'aoa': aoa_v, 'x': x_pos_drv, 'json': df_final.to_json(orient='records')})
                                    except: pass
                            st.session_state.interp_planos_memoria.extend(nuevos_planos_drv)
                            st.success(f"✅ Importados {len(nuevos_planos_drv)} planos.")

    st.markdown("---")

    # --- PASO 2 (Sin Expander) ---
    st.markdown("### 📋 PASO 2: Planos en Memoria para Análisis")
    if not st.session_state.interp_planos_memoria:
        st.warning("⚠️ No hay planos cargados para el análisis. Seleccione planos en el Paso 1.")
    else:
        df_mem = pd.DataFrame([{'Nombre': p['name'], 'AOA [°]': p['aoa'], 'X [mm]': p['x']} for p in st.session_state.interp_planos_memoria])
        st.dataframe(df_mem, use_container_width=True)
        
        if st.button("🗑️ Limpiar Planos", type="secondary", use_container_width=True):
            st.session_state.interp_planos_memoria = []
            st.session_state.results_interp = None
            st.rerun()

    st.markdown("---")
    
    # --- PASO 3 (Sin Expander) ---
    st.markdown("### ⚙️ PASO 3: Análisis de Convergencia y Predicción de Ángulos")
    
    planos_ready = sorted(st.session_state.interp_planos_memoria, key=lambda x: x['aoa'])
    if len(planos_ready) < 3:
        st.warning("Se necesitan al menos 3 planos para realizar el análisis.")
    else:
        confianza_t = st.slider("Exactitud Objetivo [%]:", 50, 99, 90, key="slider_conf_interp")

        if st.button("🚀 Iniciar Análisis", type="primary", use_container_width=True):
            with st.spinner("Calculando convergencia de planos angulares..."):
                angulos = [p['aoa'] for p in planos_ready]
                grillas_full = []
                for p in planos_ready:
                    df_tmp = pd.read_json(io.StringIO(p['json']))
                    y_lin = np.linspace(df_tmp['Y'].min(), df_tmp['Y'].max(), 100)
                    z_lin = np.linspace(df_tmp['Z'].min(), df_tmp['Z'].max(), 100)
                    Ym, Zm = np.meshgrid(y_lin, z_lin)
                    g_std = griddata((df_tmp['Y'], df_tmp['Z']), df_tmp['Presion'], (Ym, Zm), method='linear')
                    grillas_full.append(g_std)

                res_list = []
                N = len(grillas_full)
                for k in range(1, N):
                    combs = list(itertools.combinations(range(N), k))
                    if len(combs) > 100: combs = random.sample(combs, 100)
                    mejor_acc_k = 0
                    mejor_cfg_k = ""
                    for comb in combs:
                        idx_k = sorted(list(comb))
                        acc_list = []
                        for i in range(N):
                            if i in idx_k: continue
                            lo = next((idx for idx in reversed(idx_k) if idx < i), None)
                            hi = next((idx for idx in idx_k if idx > i), None)
                            if lo is not None and hi is not None:
                                t = (angulos[i] - angulos[lo]) / (angulos[hi] - angulos[lo])
                                pred = (1-t)*grillas_full[lo] + t*grillas_full[hi]
                            elif lo is not None: pred = grillas_full[lo]
                            elif hi is not None: pred = grillas_full[hi]
                            else: pred = np.zeros_like(grillas_full[0])
                            mask = ~np.isnan(grillas_full[i]) & ~np.isnan(pred)
                            if np.any(mask):
                                corr = np.corrcoef(grillas_full[i][mask], pred[mask])[0, 1]
                                acc_list.append(max(0, corr))
                            else:
                                acc_list.append(0)
                        acc_m = np.mean(acc_list) if acc_list else 1.0
                        if acc_m > mejor_acc_k:
                            mejor_acc_k = acc_m
                            mejor_cfg_k = ", ".join([f"{angulos[idx]}°" for idx in idx_k])
                    res_list.append({'Planos': k, 'Exactitud [%]': round(mejor_acc_k*100, 2), 'Mejor Configuración': mejor_cfg_k})

                gap_errors = []
                for i in range(N - 1):
                    diff = np.nanstd(grillas_full[i+1] - grillas_full[i])
                    gap_errors.append({
                        'Intervalo': f"{angulos[i]}° → {angulos[i+1]}°",
                        'α_inicio': angulos[i], 'α_fin': angulos[i+1],
                        'Δα [°]': round(angulos[i+1] - angulos[i], 2),
                        'Variación (σ ΔP)': round(float(diff), 4)
                    })

                st.session_state.results_interp = {
                    'tabla': pd.DataFrame(res_list),
                    'angulos': angulos,
                    'gap_errors': pd.DataFrame(gap_errors),
                    'N': N
                }
                st.rerun()

    if st.session_state.results_interp is not None:
        res = st.session_state.results_interp
        if isinstance(res, pd.DataFrame):
            df_r = res
            gap_df = None
        else:
            df_r = res['tabla']
            gap_df = res.get('gap_errors')

        st.markdown("---")
        st.subheader("📈 Curva de Convergencia")

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=df_r['Planos'], y=df_r['Exactitud [%]'],
            mode='lines+markers', name='Exactitud',
            line=dict(color='#60a5fa', width=2),
            marker=dict(size=8)
        ))
        if 'confianza_t' in locals():
            fig_conv.add_hline(y=confianza_t, line_dash="dash", line_color="#f59e0b",
                               annotation_text=f"Objetivo: {confianza_t}%")
        fig_conv.update_layout(
            title="Exactitud vs N° de Planos Utilizados",
            xaxis_title="N° de Planos (base de la interpolación)",
            yaxis_title="Exactitud [%]",
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
            font=dict(color='white'), height=350
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        if 'confianza_t' in locals():
            opt = df_r[df_r['Exactitud [%]'] >= confianza_t].head(1)
            if not opt.empty:
                st.success(
                    f"🎯 **RESULTADO ÓPTIMO:** Con **{opt.iloc[0]['Planos']} planos** "
                    f"(configuración: {opt.iloc[0]['Mejor Configuración']}) "
                    f"se alcanza el **{opt.iloc[0]['Exactitud [%]']}%** de exactitud."
                )
            else:
                max_row = df_r.loc[df_r['Exactitud [%]'].idxmax()]
                st.warning(
                    f"⚠️ Con los planos actuales el máximo alcanzable es **{max_row['Exactitud [%]']}%** "
                    f"usando {max_row['Planos']} planos. Se necesitan más ángulos de medición."
                )

        st.dataframe(df_r, use_container_width=True)

        if gap_df is not None and not gap_df.empty:
            st.markdown("---")
            st.subheader("🔍 Análisis de Intervalos: ¿Dónde hay más cambio del fenómeno?")
            st.caption(
                "La **Variación (σ ΔP)** mide cuánto cambia el campo de presiones entre cada par de planos. "
                "Intervalos con mayor variación son los que más necesitan planos intermedios."
            )

            fig_gaps = go.Figure()
            fig_gaps.add_trace(go.Bar(
                x=gap_df['Intervalo'], y=gap_df['Variación (σ ΔP)'],
                marker_color='#f87171', name='Variación entre planos'
            ))
            fig_gaps.update_layout(
                title="Variación del campo de presiones por intervalo angular",
                xaxis_title="Intervalo [α° → α°]",
                yaxis_title="σ(ΔPresión) [Pa]",
                paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                font=dict(color='white'), height=320
            )
            st.plotly_chart(fig_gaps, use_container_width=True)

            st.markdown("#### 💡 Ángulos Recomendados para Medir")
            gaps_ordenados = gap_df.sort_values('Variación (σ ΔP)', ascending=False)
            rec_rows = []
            for _, row in gaps_ordenados.iterrows():
                alpha_mid = round((row['α_inicio'] + row['α_fin']) / 2, 2)
                rec_rows.append({
                    'Intervalo Crítico': row['Intervalo'],
                    'Δα [°]': row['Δα [°]'],
                    'Variación': row['Variación (σ ΔP)'],
                    'α Recomendado a Medir [°]': alpha_mid
                })
            df_rec = pd.DataFrame(rec_rows)
            st.dataframe(df_rec, use_container_width=True)

            if not gaps_ordenados.empty:
                top_gap = gaps_ordenados.iloc[0]
                alpha_sug = round((top_gap['α_inicio'] + top_gap['α_fin']) / 2, 2)
                st.info(
                    f"📌 **Recomendación principal:** El intervalo más crítico es "
                    f"**{top_gap['Intervalo']}** (Δα = {top_gap['Δα [°]']}°, "
                    f"variación = {top_gap['Variación (σ ΔP)']:.4f} Pa). "
                    f"Se recomienda medir un plano en **α = {alpha_sug}°** para capturar mejor el fenómeno."
                )
