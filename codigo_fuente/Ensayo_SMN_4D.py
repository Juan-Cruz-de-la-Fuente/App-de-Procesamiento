# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import re
from scipy.spatial import Delaunay
from codigo_fuente import Auth_Manager as auth
from codigo_fuente.Calculations_Core import rotate_points

def _aplicar_pose_modelo_smn(obj_base, alpha_deg, beta_deg, dx, dy, dz, cg):
    x = np.array(obj_base['x'], dtype=float) - cg['x']
    y = np.array(obj_base['y'], dtype=float) - cg['y']
    z = np.array(obj_base['z'], dtype=float) - cg['z']
    x, y, z = rotate_points(x, y, z, 0.0, float(alpha_deg), float(-beta_deg))
    x = x + cg['x'] + dx
    y = y + cg['y'] + dy
    z = z + cg['z'] + dz
    return x, y, z

def _extraer_aoa_smn(nombre):
    m = re.search(r'OAO(neg)?(\d+(?:[.,]\d+)?)', str(nombre), re.IGNORECASE)
    if m:
        return (-1 if m.group(1) else 1) * float(str(m.group(2)).replace(',', '.'))
    return 0.0

def show_smn_4d():
    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🧪 ENSAYO-SMN 4D MULTICAPA
            </h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Mapeo Espacial con Plano de Estación X y Relieve Espacial
            </h2>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 2px solid #333; margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)

    # --- SESSION STATE INITIALIZATION ---
    if 'smn_archivos_memoria' not in st.session_state:
        st.session_state.smn_archivos_memoria = {}
    if 'smn_planos_seleccionados' not in st.session_state:
        st.session_state.smn_planos_seleccionados = []
    if 'smn_v_inf' not in st.session_state: st.session_state.smn_v_inf = 17.5
    if 'smn_rho_inf' not in st.session_state: st.session_state.smn_rho_inf = 1.2
    if 'smn_p_inf' not in st.session_state: st.session_state.smn_p_inf = -94.0 
    if 'smn_t_inf' not in st.session_state: st.session_state.smn_t_inf = 15.0

    # --- CONFIGURACIÓN DE CONDICIONES ATMOSFÉRICAS DEL INFINITO ---
    st.markdown("### 🌐 Condiciones del Infinito y Referencia")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        st.session_state.smn_v_inf = c1.number_input("Velocidad Infinito V_∞ [m/s]:", value=st.session_state.smn_v_inf, format="%.2f", key="smn_4d_v_inf_input")
        st.session_state.smn_rho_inf = c2.number_input("Densidad Infinito ρ_∞ [kg/m³]:", value=st.session_state.smn_rho_inf, format="%.4f", key="smn_4d_rho_inf_input")
        st.session_state.smn_p_inf = c3.number_input("Presión de Referencia P_∞ [Pa]:", value=st.session_state.smn_p_inf, format="%.1f", key="smn_4d_p_inf_input")
        st.session_state.smn_t_inf = c4.number_input("Temperatura T_∞ [°C]:", value=st.session_state.smn_t_inf, format="%.1f", key="smn_4d_t_inf_input")

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📥 Cargar y Guardar Plano en Estación X (4D)")
    st.caption("Subí un plano CSV para asociar a una estación física 'X' e integrarlo con múltiples capas espaciales y mallas 3D STL.")
    
    up_smn_4d = st.file_uploader("Subir archivo de plano SMN (.csv)", type=['csv'], key="up_smn_4d")
    
    if up_smn_4d:
        try:
            df_raw = pd.read_csv(up_smn_4d, sep=';', decimal=',')
            if 'Posicion Sonda X[mm]' not in df_raw.columns:
                df_raw = pd.read_csv(up_smn_4d, sep=',', decimal='.')
            
            required = ['Posicion Sonda X[mm]', 'Posicion Sonda Y[mm]']
            if not all(col in df_raw.columns for col in required):
                st.error("❌ El archivo CSV no contiene columnas de coordenadas válidas.")
            else:
                st.success(f"✅ Archivo leído correctamente: {len(df_raw)} puntos.")
                df_proc = pd.DataFrame()
                df_proc['Y'] = df_raw['Posicion Sonda X[mm]'].astype(float)
                df_proc['Z'] = df_raw['Posicion Sonda Y[mm]'].astype(float)
                
                var_mappings = {
                    'Presion_Est': 'Presion estatica [Pa]',
                    'Presion_Tot': 'Presion total [Pa]',
                    'Vel_Tot': 'Velocidad [m/seg]',
                    'Vx': 'Velocidad X [m/seg]',
                    'Vy': 'Velocidad Y [m/seg]',
                    'Vz': 'Velocidad Z [m/seg]',
                    'Alfa': 'Alfa []',
                    'Beta': 'Beta []'
                }
                for k, col in var_mappings.items():
                    found_col = next((c for c in df_raw.columns if c.replace(' ', '').lower() == col.replace(' ', '').lower() or k.lower() in c.lower()), None)
                    if found_col is not None:
                        df_proc[k] = df_raw[found_col].astype(float)
                    else:
                        df_proc[k] = 0.0
                
                q_inf = 0.5 * st.session_state.smn_rho_inf * (st.session_state.smn_v_inf ** 2)
                if q_inf != 0:
                    df_proc['Cp_Est'] = (df_proc['Presion_Est'] - st.session_state.smn_p_inf) / q_inf
                    df_proc['Cp_Tot'] = (df_proc['Presion_Tot'] - st.session_state.smn_p_inf) / q_inf
                else:
                    df_proc['Cp_Est'] = 0.0
                    df_proc['Cp_Tot'] = 0.0
                    
                name_mem = up_smn_4d.name.replace('.csv', '')
                st.session_state.smn_archivos_memoria[name_mem] = df_proc
        except Exception as e:
            st.error(f"Error procesando CSV: {e}")
            
    op_smn = list(st.session_state.smn_archivos_memoria.keys()) if st.session_state.smn_archivos_memoria else ["No hay archivos"]
    sel_smn_4d = st.selectbox("Seleccionar Plano a Guardar en Drive (4D):", op_smn, key="sel_smn_4d_save")
    smn_x_4d = st.number_input("Posición en Estación X [mm]:", value=150.0, step=10.0, key="smn_x_4d")
    smn_aoa_4d = st.number_input("Ángulo AOA [°]:", value=0.0, step=1.0, key="smn_aoa_4d")
    
    nombre_auto_4d = f"SMN-4D-X{int(smn_x_4d)}-OAO{str(smn_aoa_4d).replace('-','neg')}-T0s"
    nombre_final_4d = st.text_input("Nombre de la Estación 4D en Drive:", value=nombre_auto_4d, key="name_final_4d")
    
    if st.button("🚀 SUBIR PLANO A DRIVE (SMN 4D)", use_container_width=True, type="primary", disabled=not st.session_state.smn_archivos_memoria):
        if sel_smn_4d in st.session_state.smn_archivos_memoria:
            df_to_save = st.session_state.smn_archivos_memoria[sel_smn_4d].copy()
            df_to_save['Pos_X'] = smn_x_4d
            df_to_save['AOA'] = smn_aoa_4d
            json_data = df_to_save.to_json(orient='records')
            if auth.save_surface_data_4d(st.session_state.username, nombre_final_4d, smn_x_4d, json_data):
                st.success(f"✅ Guardado en Drive: {nombre_final_4d}")
            else:
                st.error("Error al guardar en Drive.")
                
    st.markdown("</div>", unsafe_allow_html=True)

    # 1. Selección de múltiples planos
    st.markdown("### 📥 PASO 2: Selección Multicapa (Múltiples Planos)")
    try:
        planos_drv = auth.get_user_surfaces_4d(st.session_state.username)
        planos_drv_smn = [p for p in planos_drv if p[1].startswith("SMN-4D-")]
    except Exception:
        planos_drv_smn = []

    if not planos_drv_smn:
        st.info("No hay planos SMN 4D registrados en la base de datos de tu Drive.")
    else:
        dict_drv = {f"{p[1]} [X={p[2]} mm]": p for p in planos_drv_smn}
        planos_sel_labels = st.multiselect("Seleccionar Planos SMN para Graficar en el Espacio:", list(dict_drv.keys()), key="sel_planos_smn_4d")
        
        if 'last_planos_smn_4d' not in st.session_state:
            st.session_state.last_planos_smn_4d = []
            
        if planos_sel_labels != st.session_state.last_planos_smn_4d:
            with st.spinner("Cargando planos espaciales..."):
                loaded = []
                for label in planos_sel_labels:
                    p_info = list(dict_drv[label])
                    if not p_info[4]:
                        p_info[4] = auth.get_surface_data_string(p_info[0])
                    loaded.append(tuple(p_info))
                st.session_state.smn_planos_seleccionados = loaded
                st.session_state.last_planos_smn_4d = planos_sel_labels
            st.rerun()

    # 2. Opciones de Visualización 4D
    st.markdown("### 🎨 PASO 3: Opciones y Ajustes de la escena")
    c_opt5, c_opt6, c_opt7 = st.columns(3)
    
    variables_smn_4d = {
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
    
    var_sel_4d = c_opt5.selectbox("Variable a graficar (4D):", list(variables_smn_4d.keys()), key="var_sel_smn_4d")
    relieve_scale = c_opt6.slider("Escala de Relieve espacial (Z -> relieve X):", 0.0, 5.0, 1.0, step=0.1, key="relieve_smn_4d")
    cuerda = c_opt7.number_input("Cuerda de Referencia [mm]:", value=300.0, step=10.0, key="cuerda_smn_4d")
    
    c_opt8, c_opt9 = st.columns(2)
    show_model_4d = c_opt8.checkbox("Mostrar Modelo 3D STL de Referencia", value=True, key="model_smn_4d")
    vis_ejes_4d = c_opt9.checkbox("Mostrar Ejes y Guías en Escena", value=True, key="ejes_smn_4d")
    
    # 3. Escena 3D Multicapa
    st.markdown("### 📈 Visualización Espacial Interactiva")
    
    if 'smn_planos_seleccionados' not in st.session_state or not st.session_state.smn_planos_seleccionados:
        st.warning("⚠️ Seleccione al menos un plano SMN en el Paso 2 para generar el gráfico.")
    else:
        fig = go.Figure()
        col_var_4d = variables_smn_4d[var_sel_4d]
        
        for p_info in st.session_state.smn_planos_seleccionados:
            df_p = pd.read_json(io.StringIO(p_info[4]))
            df_clean = df_p.dropna(subset=['Y', 'Z', col_var_4d]).drop_duplicates(subset=['Y', 'Z'])
            if len(df_clean) < 3: continue
            
            x_base = p_info[2] 
            tri = Delaunay(df_clean[['Y', 'Z']].values)
            
            val_plot = df_clean[col_var_4d].values
            val_max = val_plot.max()
            x_def = x_base - ((val_plot - val_max) * relieve_scale)
            
            fig.add_trace(go.Mesh3d(
                x=x_def, y=df_clean['Y'], z=df_clean['Z'],
                i=tri.simplices[:,0], j=tri.simplices[:,1], k=tri.simplices[:,2],
                intensity=val_plot, colorscale='Turbo',
                name=f"Estación X={x_base}",
                hovertemplate=f"Estación X: {x_base:.1f} mm<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<br>{var_sel_4d}: %{{intensity:.2f}}<extra></extra>"
            ))
        
        if show_model_4d and 'objeto_referencia_4d' in st.session_state and st.session_state.objeto_referencia_4d is not None:
            obj_ref = st.session_state.objeto_referencia_4d
            obj_base = st.session_state.objeto_referencia_base if 'objeto_referencia_base' in st.session_state else obj_ref
            cg = st.session_state.get('modelo_cg', {'x': 0.0, 'y': 0.0, 'z': 0.0})
            
            alpha_ref = 0.0
            if st.session_state.smn_planos_seleccionados:
                p_name = st.session_state.smn_planos_seleccionados[0][1]
                alpha_ref = _extraer_aoa_smn(p_name)
                
            xm, ym, zm = _aplicar_pose_modelo_smn(obj_base, alpha_ref, 0.0, 0.0, 0.0, 0.0, cg)
            
            if obj_ref.get('type') == 'scatter':
                fig.add_trace(go.Scatter3d(
                    x=xm, y=ym, z=zm,
                    mode='markers',
                    marker=dict(size=2, color='#64748b', opacity=0.4),
                    name='Modelo STL'
                ))
            else:
                fig.add_trace(go.Mesh3d(
                    x=xm, y=ym, z=zm,
                    i=obj_ref['i'], j=obj_ref['j'], k=obj_ref['k'],
                    color='#3b82f6', opacity=0.25,
                    name=f"Modelo STL (α={alpha_ref:.1f}°)",
                    alphahull=0, showscale=False,
                    lighting=dict(ambient=0.4, diffuse=0.8)
                ))
        
        axis_props = dict(
            showgrid=vis_ejes_4d, zeroline=vis_ejes_4d, showticklabels=vis_ejes_4d,
            showaxeslabels=vis_ejes_4d, showbackground=False
        )
        camera_dict = dict(eye=dict(x=-1.5, y=-1.5, z=1.5))
        
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                camera=camera_dict,
                xaxis=dict(title="X (Estación) [mm]" if vis_ejes_4d else "", autorange="reversed", **axis_props),
                yaxis=dict(title="Y (Envergadura) [mm]" if vis_ejes_4d else "", **axis_props),
                zaxis=dict(title="Z (Altura) [mm]" if vis_ejes_4d else "", **axis_props)
            ),
            height=800,
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)
