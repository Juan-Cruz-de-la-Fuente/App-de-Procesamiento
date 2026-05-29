import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import io
from scipy.interpolate import griddata as _gd
from scipy.ndimage import gaussian_filter, minimum_filter

from codigo_fuente.Calculations_Core import (
    obtener_numero_sensor_desde_columna,
    calcular_altura_absoluta_z
)
from codigo_fuente import Auth_Manager as auth

# ── MOTOR DE DETECCIÓN AUTOMÁTICA ────────────────────────────────────────
def _detectar_vortices(y_data, z_data, p_data):
    """
    Detección automática de vórtices basada en física.
    Discriminadores clave:
        1. Circularidad radial  → vórtice = mínimo rodeado en TODAS las dir.
        2. Aspecto compacto     → vórtice ≠ banda elongada (estela/soporte)
        3. Profundidad mínima   → evita ruido
        4. Tamaño razonable     → evita ruido pequeño y la estela completa
        5. Simetría en Y        → los vórtices de punta de ala son simétricos
    Sin parámetros de usuario.
    """
    GRID = 200
    y_lin = np.linspace(y_data.min(), y_data.max(), GRID)
    z_lin = np.linspace(z_data.min(), z_data.max(), GRID)
    Yg, Zg = np.meshgrid(y_lin, z_lin)

    Pg = _gd((y_data, z_data), p_data, (Yg, Zg), method='linear')
    nan_m = np.isnan(Pg)
    if nan_m.any():
        Pg[nan_m] = _gd((y_data, z_data), p_data, (Yg[nan_m], Zg[nan_m]), method='nearest')

    Ps = gaussian_filter(Pg, sigma=2.0)

    p_bg    = float(np.nanpercentile(Ps, 97))
    p_floor = float(np.nanpercentile(Ps, 3))
    p_range = p_bg - p_floor
    if p_range < 1e-6:
        return [], Pg, y_lin, z_lin

    sz = max(7, GRID // 20)
    lm = (minimum_filter(Ps, size=sz) == Ps) & (Ps < p_bg - p_range * 0.05)
    cands = np.argwhere(lm)
    if len(cands) == 0:
        return [], Pg, y_lin, z_lin

    pv = Ps[cands[:, 0], cands[:, 1]]
    cands = cands[np.argsort(pv)]   # orden por profundidad

    dy = y_lin[1] - y_lin[0]
    dz = z_lin[1] - z_lin[0]
    y_min, y_max = y_lin[0], y_lin[-1]
    y_mid = (y_min + y_max) / 2.0
    y_span = y_max - y_min

    N_ANG   = 24
    MAX_R   = int(GRID * 0.35)
    REC_THR = 0.60          # recuperación del 60 % de la profundidad
    angles  = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False)
    sins    = np.sin(angles)
    coss    = np.cos(angles)

    scored = []
    for ri, ci in cands:
        y_val = float(y_lin[ci])
        z_val = float(z_lin[ri])
        
        # 1. Ignorar el centro (soporte/fuselaje)
        # El soporte suele estar en el medio. Bloqueamos el 12% central.
        if abs(y_val - y_mid) < y_span * 0.06:
            continue

        p_core = Ps[ri, ci]
        depth  = p_bg - p_core
        if depth < p_range * 0.04:
            continue

        radii     = []
        n_boundary = 0
        for k in range(N_ANG):
            found = None
            for r in range(2, MAX_R):
                ri2 = int(round(ri + r * sins[k]))
                ci2 = int(round(ci + r * coss[k]))
                if not (0 <= ri2 < GRID and 0 <= ci2 < GRID):
                    n_boundary += 1
                    break
                # Criterio de recuperación local
                if Ps[ri2, ci2] >= p_core + depth * REC_THR:
                    found = r
                    break
            if found is not None:
                radii.append(found)

        # 2. Discriminadores morfológicos estrictos
        if n_boundary > N_ANG // 3: continue  # Toca mucho borde
        if len(radii) < N_ANG * 0.70: continue # No cierra (estela abierta)

        r_arr  = np.array(radii, dtype=float)
        r_mean = r_arr.mean()
        r_std  = r_arr.std()
        cov    = r_std / (r_mean + 1e-6)
        
        # Circularidad: castigo fuerte a formas elongadas (como la estela del soporte)
        compactness = max(0.0, 1.0 - cov * 2.5)
        if compactness < 0.35:
            continue

        # Relación de aspecto (Z vs Y) para descartar bandas verticales
        semi_y_est = float(np.mean(r_arr * np.abs(np.cos(angles[:len(radii)])))) * dy
        semi_z_est = float(np.mean(r_arr * np.abs(np.sin(angles[:len(radii)])))) * dz
        aspect_ratio = semi_z_est / (semi_y_est + 1e-6)
        
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            continue # Demasiado estirado (soporte o estela laminar)

        r_real_mm = r_mean * max(dy, dz)
        dom = max(y_span, z_lin[-1] - z_lin[0])
        r_frac = r_real_mm / dom
        if r_frac < 0.005 or r_frac > 0.40:
            continue

        # Score final: prioriza profundidad, circularidad y distancia al centro
        dist_centro_norm = abs(y_val - y_mid) / (y_span / 2)
        score = (compactness * 0.4) + (min(depth / p_range * 2, 1.0) * 0.4) + (dist_centro_norm * 0.2)

        scored.append(dict(
            ri=ri, ci=ci,
            y=y_val, z=z_val,
            p_core=p_core, depth=depth,
            compactness=compactness, score=score,
            semi_y=max(semi_y_est, dy), semi_z=max(semi_z_est, dz),
            area=np.pi * semi_y_est * semi_z_est,
        ))

    if not scored:
        return [], Pg, y_lin, z_lin

    # 3. NMS y Simetría
    scored.sort(key=lambda x: -x['score'])
    min_sep = GRID * 0.08
    selected = []
    for c in scored:
        if not any(np.hypot(c['ri'] - s['ri'], c['ci'] - s['ci']) < min_sep for s in selected):
            selected.append(c)
    
    # Limitar a los 6 mejores candidatos
    selected = selected[:6]


    # Emparejamiento simétrico (bonus de puntuación)
    sym_tol_y = (y_lin[-1] - y_lin[0]) * 0.09
    sym_tol_z = (z_lin[-1] - z_lin[0]) * 0.09
    paired = set()
    for i, a in enumerate(selected):
        if i in paired:
            continue
        y_mirror = 2 * y_mid - a['y']
        for j, b in enumerate(selected):
            if j <= i or j in paired:
                continue
            if abs(b['y'] - y_mirror) < sym_tol_y and abs(b['z'] - a['z']) < sym_tol_z:
                paired.add(i); paired.add(j)
                selected[i]['has_pair'] = True
                selected[j]['has_pair'] = True
                break

    # Construir polígonos elípticos para visualización
    vortices = []
    for v in selected:
        th = np.linspace(0, 2 * np.pi, 120)
        vy = (v['y'] + v['semi_y'] * np.cos(th)).tolist()
        vz = (v['z'] + v['semi_z'] * np.sin(th)).tolist()
        vortices.append({
            'id'         : f"V{len(vortices)+1}",
            'y'          : v['y'],
            'z'          : v['z'],
            'p_min'      : v['p_core'],
            'depth'      : v['depth'],
            'semi_y'     : round(v['semi_y'], 2),
            'semi_z'     : round(v['semi_z'], 2),
            'area'       : v['area'],
            'compactness': round(v['compactness'], 3),
            'score'      : round(v['score'], 3),
            'has_pair'   : v.get('has_pair', False),
            'poly_y'     : vy,
            'poly_z'     : vz,
        })

    return vortices, Pg, y_lin, z_lin
# ─────────────────────────────────────────────────────────────────────────

def show_vortices():
    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🌀 ANÁLISIS DE VÓRTICES
            </h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Detección Automática Basada en Física
            </h2>
        </div>
    """, unsafe_allow_html=True)

    if "matriz_vortices_actual" not in st.session_state: st.session_state.matriz_vortices_actual = pd.DataFrame()

    # --- PASO 1 (Sin Expander) ---
    st.markdown("### 📥 PASO 1: Selección de Origen de Datos")
    fuente = st.radio(
        "Seleccionar fuente:",
        ["Subir CSV (Y,Z,P)", "🧠 Memoria (2D)", "☁️ Drive (2D)"],
        horizontal=True,
        key="fuente_vort_v3"
    )
    
    df_matriz = pd.DataFrame()

    if fuente == "Subir CSV (Y,Z,P)":
        up_csv = st.file_uploader("Archivo CSV (separado por ';' y decimales con ',')", type=['csv'], key="up_vort")
        if up_csv:
            try:
                df_matriz = pd.read_csv(up_csv, sep=';', decimal=',')
                if not {'Y','Z','Presion'}.issubset(df_matriz.columns):
                    df_matriz = pd.read_csv(up_csv, sep=',', decimal='.')
                if {'Y','Z','Presion'}.issubset(df_matriz.columns):
                    st.success("✅ Matriz CSV Cargada y lista para el análisis.")
            except: 
                st.error("Error al leer el archivo CSV.")

    elif fuente == "🧠 Memoria (2D)":
        if 'archivos_2d_memoria' not in st.session_state or not st.session_state.archivos_2d_memoria:
            st.warning("⚠️ No hay matrices en la memoria de sesión. Procese archivos en la sección Vis. Estela 2D primero.")
        else:
            arc_sel = st.selectbox("Seleccionar Archivo de Memoria:", list(st.session_state.archivos_2d_memoria.keys()))
            df_sel = st.session_state.archivos_2d_memoria[arc_sel]
            tiempos = sorted(df_sel['Tiempo_s'].dropna().unique())
            t_sel = st.selectbox("Confirmar Tiempo [s]:", tiempos)
            
            if st.button("📥 Cargar Matriz de Memoria", use_container_width=True):
                # Ensamblar matriz
                res = []
                # Usar config local si existe, sino valores por defecto
                conf = st.session_state.get('configuracion_2d_local', {'distancia_toma_12': -120, 'distancia_entre_tomas': 10.0, 'orden': 'asc'})
                df_run = df_sel[df_sel['Tiempo_s'] == t_sel]
                for _, row in df_run.iterrows():
                    y_t, z_b = row.get('Pos_Y_Traverser'), row.get('Pos_Z_Base')
                    for col in df_run.columns:
                        ns = obtener_numero_sensor_desde_columna(col)
                        if ns is not None:
                            val = row[col]
                            if pd.isna(val): continue
                            zr = calcular_altura_absoluta_z(ns, z_b, conf['distancia_toma_12'], conf['distancia_entre_tomas'], 12, conf['orden'])
                            res.append({'Y': y_t, 'Z': zr, 'Presion': val})
                df_matriz = pd.DataFrame(res)
                if not df_matriz.empty: 
                    st.success("✅ Matriz ensamblada y lista para el análisis.")

    else: # Drive
        try: 
            files_drv = auth.get_user_files_2d(st.session_state.username)
        except: 
            files_drv = []
            
        if not files_drv: 
            st.info("No hay matrices en Drive.")
        else:
            dict_drv = {f"{a[1]}": a for a in files_drv}
            sel_drv = st.selectbox("Seleccionar Matriz de Drive:", ["-- Seleccionar --"] + list(dict_drv.keys()))
            if sel_drv != "-- Seleccionar --":
                if st.button("📥 Descargar y Cargar", use_container_width=True):
                    raw = auth.download_file_2d(dict_drv[sel_drv][0])
                    if raw:
                        df_matriz = pd.read_csv(io.BytesIO(raw), sep=';', decimal=',')
                        if not {'Y','Z','Presion'}.issubset(df_matriz.columns):
                            df_matriz = pd.read_csv(io.BytesIO(raw), sep=',', decimal='.')
                        st.success("✅ Matriz descargada y lista para el análisis.")

    if not df_matriz.empty:
        st.session_state.matriz_vortices_actual = df_matriz

    st.markdown("---")

    # --- PASO 2 (Sin Expander) ---
    st.markdown("### 📈 PASO 2: Detección y Visualización de Vórtices")
    
    if st.session_state.matriz_vortices_actual.empty:
        st.warning("⚠️ Seleccione y cargue una matriz en el Paso 1 para iniciar la detección.")
    else:
        df_m = st.session_state.matriz_vortices_actual
        with st.spinner("🧠 El motor de IA está buscando núcleos de vórtices..."):
            vortices, P_grid, y_lin, z_lin = _detectar_vortices(df_m['Y'].values, df_m['Z'].values, df_m['Presion'].values)
        
        if P_grid is not None:
            fig = go.Figure()
            fig.add_trace(go.Contour(x=y_lin, y=z_lin, z=P_grid, colorscale='Jet', colorbar=dict(title="P [Pa]")))
            
            colors = ['#ffffff','#ffff00','#00ff88','#ff6600','#ff00ff']
            for k, v in enumerate(vortices):
                col = colors[k % len(colors)]
                fig.add_trace(go.Scatter(x=v['poly_y'], y=v['poly_z'], mode='lines', fill='toself', line=dict(color=col, width=3), name=v['id']))
                fig.add_trace(go.Scatter(x=[v['y']], y=[v['z']], mode='markers+text', marker=dict(symbol='x', size=12, color=col), text=[v['id']], textposition='top center', showlegend=False))
            
            fig.update_layout(xaxis_title="Y [mm]", yaxis_title="Z [mm]", height=700, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            fig.update_xaxes(scaleanchor="y", scaleratio=1)
            st.plotly_chart(fig, use_container_width=True)

            if vortices:
                st.markdown("### 📋 Resultados de Detección")
                df_res = pd.DataFrame([{
                    "ID": v['id'], "Y [mm]": round(v['y'],1), "Z [mm]": round(v['z'],1),
                    "ΔP [Pa]": round(v['depth'],2), "Área [mm²]": round(v['area'],1),
                    "Circ.": v['compactness']
                } for v in vortices])
                st.dataframe(df_res, use_container_width=True)
                
                # Simetría
                y_mid = (y_lin.min() + y_lin.max()) / 2.0
                area_izq = sum(v['area'] for v in vortices if v['y'] < y_mid)
                area_der = sum(v['area'] for v in vortices if v['y'] >= y_mid)
                asim = (area_der - area_izq) / (area_izq + area_der + 1e-9)
                
                col1, col2 = st.columns(2)
                col1.metric("Asimetría de Área", f"{asim:+.2f}")
                col2.metric("β Estimado", f"{(asim*15):+.1f}°")
            else:
                st.info("No se detectaron estructuras vorticosas claras.")
