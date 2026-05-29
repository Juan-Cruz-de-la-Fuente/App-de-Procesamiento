import plotly.graph_objects as go
import numpy as np
import pandas as pd
import re
from scipy.spatial import Delaunay
from codigo_fuente.Calculations_Core import obtener_numero_sensor_desde_columna, calcular_altura_absoluta_z, extraer_datos_para_grafico
import streamlit as st

def mostrar_configuracion_sensores(key_suffix):
    """UI helper to show configuration."""
    st.markdown("<div style='padding: 1rem; border: 1px solid #333; border-radius: 8px; background-color: #000;'>", unsafe_allow_html=True)
    orden_sensores = st.selectbox(
        "Orden de lectura de sensores",
        ["asc", "des"],
        format_func=lambda x: "Ascendente (Sensor 1 → 12)" if x == "asc" else "Descendente (Sensor 12 → 1)",
        key=f"orden_{key_suffix}"
    )

    sensor_referencia = st.selectbox(
        "Sensor de referencia (Toma 12)",
        [f"Sensor {i}" for i in range(1, 37)],
        index=11, key=f"sensor_ref_{key_suffix}"
    )

    c1, c2 = st.columns(2)
    with c1:
        distancia_toma_12 = st.number_input("Distancia Toma 12 [mm]", value=-120.0, step=1.0, format="%.1f", key=f"dist_12_{key_suffix}")
    with c2:
        distancia_entre_tomas = st.number_input("Sep. entre tomas [mm]", value=10.00, step=0.01, format="%.2f", key=f"dist_entre_{key_suffix}")

    st.markdown("</div>", unsafe_allow_html=True)
    
    return {
        'orden': orden_sensores,
        'sensor_referencia': sensor_referencia,
        'distancia_toma_12': distancia_toma_12,
        'distancia_entre_tomas': distancia_entre_tomas
    }
def crear_superficie_delaunay_3d(datos_completos, configuracion_3d, nombre_archivo, mostrar_puntos=True, variable='Presion Total'):
    try:
        posicion_inicial = configuracion_3d.get('distancia_toma_12', -120.0)
        distancia_entre_tomas = configuracion_3d.get('distancia_entre_tomas', 10.0)
        orden = configuracion_3d.get('orden', 'asc')
        puntos_y, puntos_z_altura, presiones_z = [], [], []
        sensor_cols = [c for c in datos_completos.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]
        n_sensores = max([obtener_numero_sensor_desde_columna(c) for c in sensor_cols], default=0)
        
        for _, fila in datos_completos.iterrows():
            y_traverser = fila.get('Pos_Y_Traverser', None) 
            z_base_ref = fila.get('Pos_Z_Base', None)
            if pd.isna(y_traverser) or pd.isna(z_base_ref): continue
            for col in sensor_cols:
                sensor_num = obtener_numero_sensor_desde_columna(col)
                if sensor_num is None: continue
                altura_sensor_real = calcular_altura_absoluta_z(sensor_num, z_base_ref, posicion_inicial, distancia_entre_tomas, n_sensores, orden)
                presion = fila.get(col, None)
                if pd.isna(presion): continue
                try:
                    p_val = float(str(presion).replace(',', '.'))
                    val_final = p_val
                    if variable == 'P_t / Rho_inf':
                        rho = float(fila.get('rho_inf', 1.225))
                        val_final = p_val / rho if rho != 0 else 0
                    elif variable == 'Velocidad Infinito':
                        val_final = float(fila.get('V_inf', 0.0))
                    elif variable == 'Presion Infinito':
                        val_final = float(fila.get('P_inf', 101325.0))
                    puntos_y.append(y_traverser)
                    puntos_z_altura.append(altura_sensor_real)
                    presiones_z.append(val_final)
                except: continue

        if len(puntos_y) < 4: return None
        tri = Delaunay(np.vstack([puntos_y, puntos_z_altura]).T)
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(
            x=puntos_y, y=puntos_z_altura, z=presiones_z,
            i=tri.simplices[:, 0], j=tri.simplices[:, 1], k=tri.simplices[:, 2],
            intensity=presiones_z, colorscale='Turbo', colorbar_title='Presión [Pa]',
            name='Superficie de presión',
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5, roughness=0.5, fresnel=0.2)
        ))
        if mostrar_puntos:
            fig.add_trace(go.Scatter3d(x=puntos_y, y=puntos_z_altura, z=presiones_z, mode='markers', marker=dict(size=3, color='red'), name='Puntos medidos'))
        fig.update_layout(title=f"Superficie 3D - {nombre_archivo}", scene=dict(xaxis_title="Y [mm]", yaxis_title="Z [mm]", zaxis_title="P [Pa]", aspectmode='data'), width=1200, height=800)
        return fig
    except: return None

def crear_superficie_diferencia_delaunay_3d(datos_a, datos_b, nombre_a, nombre_b, configuracion_3d, mostrar_puntos=True):
    try:
        def extraer_puntos(datos):
            p = {}
            s_cols = [c for c in datos.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]
            for _, fila in datos.iterrows():
                y, z_b = fila.get('Pos_Y_Traverser'), fila.get('Pos_Z_Base')
                if pd.isna(y) or pd.isna(z_b): continue
                for col in s_cols:
                    num = obtener_numero_sensor_desde_columna(col)
                    if num is None: continue
                    alt = calcular_altura_absoluta_z(num, z_b, configuracion_3d.get('distancia_toma_12',-120), configuracion_3d.get('distancia_entre_tomas',10), 12, configuracion_3d.get('orden','asc'))
                    pres = fila.get(col)
                    if pd.isna(pres): continue
                    try: p[(y, alt)] = float(str(pres).replace(',', '.'))
                    except: continue
            return p
        pts_a, pts_b = extraer_puntos(datos_a), extraer_puntos(datos_b)
        comunes = set(pts_a.keys()) & set(pts_b.keys())
        if len(comunes) < 4: return None
        py, pz, pdiff = [], [], []
        for (y, z) in comunes:
            py.append(y); pz.append(z); pdiff.append(pts_a[(y, z)] - pts_b[(y, z)])
        tri = Delaunay(np.vstack([py, pz]).T)
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=py, y=pz, z=pdiff, i=tri.simplices[:, 0], j=tri.simplices[:, 1], k=tri.simplices[:, 2], intensity=pdiff, colorscale='RdBu_r', colorbar_title='ΔP [Pa]'))
        fig.update_layout(title=f"Diferencia: {nombre_a} - {nombre_b}", scene=dict(xaxis_title="Y [mm]", yaxis_title="Z [mm]", zaxis_title="ΔP [Pa]", aspectmode='data'))
        return fig
    except: return None

def crear_grafico_diferencia_areas(sub_archivo_a, sub_archivo_b, configuracion):
    z_a, p_a = extraer_datos_para_grafico(sub_archivo_a, configuracion)
    z_b, p_b = extraer_datos_para_grafico(sub_archivo_b, configuracion)
    if not z_a or not z_b: return None, 0
    z_min, z_max = max(min(z_a), min(z_b)), min(max(z_a), max(z_b))
    z_interp = np.linspace(z_min, z_max, 100)
    p_a_i, p_b_i = np.interp(z_interp, z_a, p_a), np.interp(z_interp, z_b, p_b)
    diff = p_a_i - p_b_i
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_a_i, y=z_interp, name=sub_archivo_a['archivo_fuente'], line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=p_b_i, y=z_interp, name=sub_archivo_b['archivo_fuente'], line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=diff, y=z_interp, fill='toself', name='Diferencia', line=dict(color='green' if np.mean(diff)>0 else 'red')))
    fig.update_layout(title="Diferencia de Perfiles", xaxis_title="P [Pa]", yaxis_title="Z [mm]", height=700)
    return fig, np.trapz(np.abs(diff), z_interp)
