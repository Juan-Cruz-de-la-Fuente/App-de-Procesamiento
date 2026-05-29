import streamlit as st
import numpy as np
import pandas as pd
import imageio
import os
import tempfile
import json
import re
from datetime import datetime
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt
import matplotlib.cm as _cm
import matplotlib.colors as _mcolors
from mpl_toolkits.mplot3d import Axes3D as _Axes3D
from scipy.interpolate import griddata as _gd_anim
from scipy.spatial import Delaunay as _Del
from scipy.interpolate import LinearNDInterpolator as _LND

from codigo_fuente.Calculations_Core import rotate_points, calcular_variable_atmosferica
from codigo_fuente import Auth_Manager as auth

# Helper: extraer AOA del nombre
def _aoa_from_name_anim(nombre):
    m = re.search(r'OAO(neg)?(\d+(?:[.,]\d+)?)', str(nombre), re.IGNORECASE)
    if m:
        return (-1 if m.group(1) else 1) * float(str(m.group(2)).replace(',', '.'))
    return None

# Helper: aplicar pose
def _pose_anim(obj_base, alpha_deg, beta_deg, cg):
    x = np.array(obj_base['x'], dtype=float) - cg['x']
    y = np.array(obj_base['y'], dtype=float) - cg['y']
    z = np.array(obj_base['z'], dtype=float) - cg['z']
    x, y, z = rotate_points(x, y, z, 0.0, float(alpha_deg), float(-beta_deg))
    return x + cg['x'], y + cg['y'], z + cg['z']

def show_animacion():
    st.markdown("""
    <div class="header-container">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🎬 ANIMACIÓN 4D
        </h1>
        <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Interpolación de planos de presión y cabeceo del modelo geométrico
        </h2>
    </div>
    """, unsafe_allow_html=True)

    if 'anim4d_grillas' not in st.session_state: st.session_state.anim4d_grillas = None
    if 'anim4d_aoa_range' not in st.session_state: st.session_state.anim4d_aoa_range = None
    if 'anim4d_x_range' not in st.session_state: st.session_state.anim4d_x_range = None
    if 'anim4d_pmin' not in st.session_state: st.session_state.anim4d_pmin = 0.0
    if 'anim4d_pmax' not in st.session_state: st.session_state.anim4d_pmax = 1.0
    if 'anim4d_frames_cache' not in st.session_state: st.session_state.anim4d_frames_cache = []
    if 'anim4d_session_meta' not in st.session_state: st.session_state.anim4d_session_meta = {}

    try:
        mis_superficies_anim = auth.get_user_surfaces_4d(st.session_state.username)
    except AttributeError:
        st.error("Error conectando con base de datos (función get_user_surfaces_4d no encontrada).")
        mis_superficies_anim = []

    if not mis_superficies_anim:
        st.info("⚠️ No hay planos 4D guardados. Ve a **Vis. Estela 4D → Paso 1** para guardar planos primero.")
    else:
        dict_sup_anim = {f"{s[1]} (X={s[2]}mm) [{s[3][:10] if s[3] else ''}]": s for s in mis_superficies_anim}

        c_sel_left, c_sel_right = st.columns([1.2, 2])

        with c_sel_left:
            st.markdown("### 📂 Paso 1: Selección de Planos")

            opciones_var_anim4d = ["Presión Total [Actual]", " ρ_∞", "V_∞", "P_∞"]
            var_anim_sel = st.selectbox("📊 Variable a visualizar:", opciones_var_anim4d, key="var_anim4d_sel")

            modo_fil_anim = st.radio(
                "Filtrar por:",
                ["✅ Individual", " Por Plano (X)", "🎯 Por AOA"],
                key="modo_fil_anim4d",
                horizontal=True
            )

            if modo_fil_anim == "✅ Individual":
                sel_anim_labels = st.multiselect("Seleccionar planos:", list(dict_sup_anim.keys()), key="sel_anim_ind")

            elif modo_fil_anim == " Por Plano (X)":
                x_positions_anim = sorted(set(s[2] for s in mis_superficies_anim))
                x_sel_anim = st.multiselect("Posiciones X [mm]:", x_positions_anim, default=x_positions_anim, key="sel_x_anim")
                sel_anim_labels = [lbl for lbl, s in dict_sup_anim.items() if s[2] in x_sel_anim]
                st.caption(f"📊 {len(sel_anim_labels)} planos en {len(x_sel_anim)} posiciones X")

            else:
                all_aoas_anim = sorted(set(
                    _aoa_from_name_anim(s[1]) for s in mis_superficies_anim
                    if _aoa_from_name_anim(s[1]) is not None
                ))
                if not all_aoas_anim:
                    st.warning("⚠️ No se detectaron AOAs en los nombres (formato: OAO{N} o OAOneg{N})")
                    sel_anim_labels = []
                else:
                    aoas_sel_anim = st.multiselect(
                        "Seleccionar AOAs [°]:", [f"{a}°" for a in all_aoas_anim],
                        default=[f"{a}°" for a in all_aoas_anim], key="sel_aoas_anim4d"
                    )
                    aoas_num_anim = [float(a.replace('°', '')) for a in aoas_sel_anim]
                    sel_anim_labels = [lbl for lbl, s in dict_sup_anim.items()
                                       if _aoa_from_name_anim(s[1]) in aoas_num_anim]
                    st.caption(f"📊 {len(sel_anim_labels)} planos seleccionados")

            pressure_scale_anim = st.slider("Escala de Relieve [presión→X]:", 0.1, 10.0, 1.0, 0.1, key="scale_anim_interp")
            mostrar_modelo_anim = st.checkbox("Mostrar modelo 3D", value=True, key="show_model_anim")

        with c_sel_right:
            if sel_anim_labels:
                st.markdown("### 🔢 Paso 2: Pre-computar Interpolación")
                st.info("Computá la grilla una vez. Luego el slider moverá el gráfico al instante sin recalcular.")

                if st.button("⚡ Pre-computar interpolación", type="primary", use_container_width=True, key="btn_precompute"):
                    items_precomp = []
                    all_y_pc, all_z_pc = [], []
                    all_p_pc = []

                    prog_pc = st.progress(0)
                    status_pc = st.empty()

                    for fi, lbl in enumerate(sel_anim_labels):
                        s_data = list(dict_sup_anim[lbl])
                        try:
                            if not s_data[4]:
                                s_data[4] = auth.get_surface_data_string(s_data[0])
                            df_tmp = pd.DataFrame(json.loads(s_data[4]))
                            df_tmp['Presion'] = calcular_variable_atmosferica(df_tmp, var_anim_sel)
                            df_c = df_tmp.dropna(subset=['Y', 'Z', 'Presion'])
                            aoa_v = _aoa_from_name_anim(s_data[1])
                            items_precomp.append({
                                'aoa': aoa_v if aoa_v is not None else 0.0,
                                'x': float(s_data[2]),
                                'df': df_c,
                                'name': s_data[1]
                            })
                            all_y_pc.extend(df_c['Y'].tolist())
                            all_z_pc.extend(df_c['Z'].tolist())
                            all_p_pc.extend(df_c['Presion'].tolist())
                        except Exception as e:
                            st.warning(f"Error cargando {lbl}: {e}")
                        prog_pc.progress((fi + 1) / len(sel_anim_labels))

                    if items_precomp:
                        items_precomp.sort(key=lambda d: d['aoa'])
                        aoa_arr_pc = np.array([d['aoa'] for d in items_precomp])
                        x_arr_pc   = np.array([d['x']   for d in items_precomp])

                        all_yz_pts = np.column_stack([all_y_pc, all_z_pc])
                        _, uniq_idx = np.unique(np.round(all_yz_pts, 4), axis=0, return_index=True)
                        Y_base = np.array(all_y_pc)[uniq_idx]
                        Z_base = np.array(all_z_pc)[uniq_idx]

                        grillas_pc = []
                        status_pc.text("Interpolando planos a la grilla de puntos reales...")
                        for ii, item in enumerate(items_precomp):
                            df_g = item['df']
                            P_g = _gd_anim(
                                (df_g['Y'].values, df_g['Z'].values), df_g['Presion'].values,
                                (Y_base, Z_base), method='linear'
                            )
                            nan_mask = np.isnan(P_g)
                            if nan_mask.any():
                                P_nn = _gd_anim(
                                    (df_g['Y'].values, df_g['Z'].values), df_g['Presion'].values,
                                    (Y_base[nan_mask], Z_base[nan_mask]), method='nearest'
                                )
                                P_g[nan_mask] = P_nn
                            grillas_pc.append(P_g)
                            prog_pc.progress((ii + 1) / len(items_precomp))

                        st.session_state.anim4d_grillas = {
                            'aoa_arr': aoa_arr_pc,
                            'x_arr': x_arr_pc,
                            'Y': Y_base,
                            'Z': Z_base,
                            'grillas': grillas_pc,
                            'items': items_precomp,
                            'var': var_anim_sel,
                            'modo': 'puntos_reales'
                        }
                        st.session_state.anim4d_pmin = float(np.nanmin(all_p_pc))
                        st.session_state.anim4d_pmax = float(np.nanmax(all_p_pc))
                        st.session_state.anim4d_aoa_range = (float(aoa_arr_pc.min()), float(aoa_arr_pc.max()))
                        status_pc.empty(); prog_pc.empty()
                        n_pts = len(Y_base)
                        st.success(f"✅ Pre-computación completada: {len(items_precomp)} planos | {n_pts:,} puntos reales | AOA {aoa_arr_pc.min():.1f}° → {aoa_arr_pc.max():.1f}°")
                        st.rerun()

        if st.session_state.anim4d_grillas is not None:
            st.markdown("---")
            st.markdown("### 🎛️ Paso 3: Visualización Interactiva")

            g = st.session_state.anim4d_grillas
            aoa_min_v, aoa_max_v = st.session_state.anim4d_aoa_range
            pmin_v = st.session_state.anim4d_pmin
            pmax_v = st.session_state.anim4d_pmax

            c_ctrl_anim, c_plot_anim = st.columns([1, 2.5])

            with c_ctrl_anim:
                if aoa_min_v != aoa_max_v:
                    alpha_slider = st.slider(
                        f"α Alpha [°]  ({aoa_min_v:.1f}° → {aoa_max_v:.1f}°):",
                        min_value=float(aoa_min_v), max_value=float(aoa_max_v),
                        value=float(aoa_min_v), step=0.5, key="alpha_slider_anim4d"
                    )
                else:
                    alpha_slider = float(aoa_min_v)
                    st.info(f"Solo un AOA: {aoa_min_v}°")

                st.markdown("##### 🎨 Visualización")
                vis_modelo_a = st.selectbox("Modelo 3D:", ["Azul Translúcido", "Negro Mate", "Plata Metalizada", "Puntos"], index=0, key="vis_mod_anim")
                c_va1, c_va2 = st.columns(2)
                vis_bg_a = c_va1.selectbox("Fondo:", ["Oscuro (Negro)", "Claro (Blanco)"], index=0, key="vis_bg_anim")
                vis_ejes_a = c_va2.checkbox("Mostrar Ejes 3D", value=True, key="vis_ejes_anim")
                
                st.markdown("---")
                sc_anim = st.slider("Escala Relieve:", 0.1, 10.0, pressure_scale_anim, 0.1, key="sc_anim_live")

                aoa_arr = g['aoa_arr']
                idx_lo = max(0, min(int(np.searchsorted(aoa_arr, alpha_slider)) - 1, len(aoa_arr) - 2))
                idx_hi = idx_lo + 1
                t_v = (alpha_slider - aoa_arr[idx_lo]) / (aoa_arr[idx_hi] - aoa_arr[idx_lo]) if aoa_arr[idx_hi] != aoa_arr[idx_lo] else 0.0
                x_v = (1 - t_v) * g['x_arr'][idx_lo] + t_v * g['x_arr'][idx_hi]

                st.markdown(f"""
                <div style="background:#111; border:1px solid #333; border-radius:8px; padding:10px; margin-top:10px;">
                    <p style="color:#888; font-size:0.75rem; margin:0;">Interpolación activa</p>
                    <p style="color:white; font-size:0.9rem; margin:4px 0;">α = {alpha_slider:.1f}°</p>
                    <p style="color:#aaa; font-size:0.75rem; margin:0;">
                        Entre {g['items'][idx_lo]['name']}<br>
                        y {g['items'][idx_hi]['name']}<br>
                        t = {t_v:.2f} | X ≈ {x_v:.1f} mm
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if st.button("🗑️ Limpiar pre-computación", key="btn_clear_precomp", use_container_width=True):
                    st.session_state.anim4d_grillas = None
                    st.session_state.anim4d_aoa_range = None
                    st.rerun()

            with c_plot_anim:
                P_interp = (1 - t_v) * g['grillas'][idx_lo] + t_v * g['grillas'][idx_hi]
                Y_v = g['Y']
                Z_v = g['Z']
                mask_v = ~np.isnan(P_interp)

                fig_live = go.Figure()

                if mostrar_modelo_anim and 'objeto_referencia_4d' in st.session_state:
                    obj_anim = st.session_state.objeto_referencia_base if 'objeto_referencia_base' in st.session_state else st.session_state.objeto_referencia_4d
                    cg_anim = st.session_state.get('modelo_cg', {'x': 0.0, 'y': 0.0, 'z': 0.0})
                    xm_a, ym_a, zm_a = _pose_anim(obj_anim, alpha_slider, 0.0, cg_anim)
                    obj_ref = st.session_state.objeto_referencia_4d
                    if vis_modelo_a == "Puntos" or obj_ref['type'] == 'scatter':
                        fig_live.add_trace(go.Scatter3d(
                            x=xm_a, y=ym_a, z=zm_a,
                            mode='markers', marker=dict(size=2, color='#888', opacity=0.5),
                            name="Modelo"
                        ))
                    else:
                        if vis_modelo_a == "Negro Mate":
                            color_m, opac = '#222222', 1.0
                            lighting = dict(ambient=0.3, diffuse=0.5, specular=0.1, roughness=0.9)
                        elif vis_modelo_a == "Plata Metalizada":
                            color_m, opac = '#e0e0e0', 1.0
                            lighting = dict(ambient=0.4, diffuse=0.8, specular=1.0, roughness=0.1)
                        else:
                            color_m, opac = '#5588cc', 0.3
                            lighting = dict(ambient=0.4, diffuse=0.8)

                        fig_live.add_trace(go.Mesh3d(
                            x=xm_a, y=ym_a, z=zm_a,
                            i=obj_ref['i'], j=obj_ref['j'], k=obj_ref['k'],
                            color=color_m, opacity=opac, name="Modelo",
                            alphahull=0, showscale=False,
                            lighting=lighting
                        ))

                if mask_v.any():
                    Y_ok = Y_v[mask_v]
                    Z_ok = Z_v[mask_v]
                    P_ok = P_interp[mask_v]
                    P_max_ref = pmax_v
                    X_def_v = x_v - ((P_ok - P_max_ref) * sc_anim)

                    try:
                        pts_yz = np.column_stack([Y_ok, Z_ok])
                        tri_v = _Del(pts_yz)
                        fig_live.add_trace(go.Mesh3d(
                            x=X_def_v, y=Y_ok, z=Z_ok,
                            i=tri_v.simplices[:,0], j=tri_v.simplices[:,1], k=tri_v.simplices[:,2],
                            intensity=P_ok,
                            colorscale='Jet', cmin=pmin_v, cmax=pmax_v,
                            showscale=True, opacity=1.0, flatshading=True,
                            name=f"Presión (α={alpha_slider:.1f}°)",
                            colorbar=dict(title=dict(text=g.get('var','Presión'), side='right'))
                        ))
                    except Exception:
                        fig_live.add_trace(go.Scatter3d(
                            x=X_def_v, y=Y_ok, z=Z_ok,
                            mode='markers',
                            marker=dict(size=3, color=P_ok, colorscale='Jet', showscale=True,
                                        cmin=pmin_v, cmax=pmax_v),
                            name=f"Presión (α={alpha_slider:.1f}°)"
                        ))
                bg_color_a = '#0e1117' if "Oscuro" in vis_bg_a else '#ffffff'
                font_color_a = 'white' if "Oscuro" in vis_bg_a else 'black'
                axis_props_a = dict(showgrid=vis_ejes_a, zeroline=vis_ejes_a, showticklabels=vis_ejes_a, showaxeslabels=vis_ejes_a, showbackground=False)

                fig_live.update_layout(
                    title=f"α = {alpha_slider:.1f}°",
                    scene=dict(
                        aspectmode='data',
                        xaxis=dict(title="X (Estación)" if vis_ejes_a else "", autorange="reversed", **axis_props_a),
                        yaxis=dict(title="Y (Envergadura)" if vis_ejes_a else "", **axis_props_a),
                        zaxis=dict(title="Z (Altura)" if vis_ejes_a else "", **axis_props_a)
                    ),
                    paper_bgcolor=bg_color_a,
                    plot_bgcolor=bg_color_a,
                    font=dict(color=font_color_a),
                    height=620,
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                st.plotly_chart(fig_live, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🎥 Paso 4: Generar Animación GIF")
            st.caption("💡 Matplotlib puro — sin Chrome ni kaleido. Dos modos: 2D suave (contourf) o 4D isométrico con modelo.")

            st.info("💡 **Nota de Rendimiento:** Generar el GIF a alta resolución puede ser muy lento en servidores online (Streamlit Cloud) comparado con una ejecución local, debido a las limitaciones de CPU y al pesado procesamiento gráfico. Si usas la app online, prueba resoluciones bajas y menos frames.")
            c_gif0, c_gif1, c_gif2, c_gif3 = st.columns(4)
            tipo_gif  = c_gif0.radio("Tipo:", ["🗺 2D suave", "🚀 4D"], index=1, key="tipo_gif_sel")
            fps_gif   = c_gif1.slider("FPS:", 1, 60, 18, key="fps_gif_anim")
            n_pas_gif = c_gif2.slider("N° frames:", 5, 120, 60, key="npasos_gif")
            sc_gif    = c_gif3.slider("× relieve (4D):", 0.1, 20.0, 8.1, 0.1, key="sc_gif_anim")
            
            c_opt1, c_opt2 = st.columns(2)
            dpi_gif   = c_opt1.selectbox("Resolución (DPI):", [110, 150, 200, 300, 600], index=4, key="dpi_gif_anim", help="Resolución de imagen. Valores altos mejoran enormemente la calidad, pero aumentan exponencialmente el tiempo de generación.")
            formato_anim = c_opt2.selectbox("Formato de Exportación:", ["GIF (Clásico, max 256 colores)", "MP4 (Alta Calidad True Color)"], index=1, key="formato_anim_sel")

            elev_gif, azim_gif = 25, -135
            if "4D" in tipo_gif:
                st.markdown("##### 📷 Posición de Cámara (Vista 4D)")
                c_cam1, c_cam2, c_cam3 = st.columns(3)
                preset_cam = c_cam1.selectbox("Preajustes:", ["Isométrica", "Opuesta a Isométrica", "Frente (aguas abajo)", "Lateral", "Planta", "Personalizada"], index=5, key="preset_cam_gif")
                if preset_cam == "Isométrica": elev_def, azim_def = 25, -135
                elif preset_cam == "Opuesta a Isométrica": elev_def, azim_def = 25, 45
                elif preset_cam == "Frente (aguas abajo)": elev_def, azim_def = 0, -180
                elif preset_cam == "Lateral": elev_def, azim_def = 0, -90
                elif preset_cam == "Planta": elev_def, azim_def = 90, -90
                else: elev_def, azim_def = 25, 45

                elev_gif = c_cam2.slider("Elevación [°]", -90, 90, elev_def, disabled=(preset_cam != "Personalizada"), key="elev_gif_anim")
                azim_gif = c_cam3.slider("Azimut [°]", -180, 180, azim_def, disabled=(preset_cam != "Personalizada"), key="azim_gif_anim")
                if preset_cam != "Personalizada": elev_gif, azim_gif = elev_def, azim_def

            c_btn1, c_btn2 = st.columns(2)
            btn_preview = c_btn1.button(" Previsualizar Vista (1 frame)", use_container_width=True)
            btn_generar = c_btn2.button("🎥 Generar GIF Completo", type="primary", use_container_width=True)

            if btn_preview or btn_generar:
                alpha_range_gif = np.linspace(aoa_min_v, aoa_max_v, n_pas_gif)
                if btn_preview:
                    alpha_range_gif = [alpha_range_gif[0]]

                status_gif = st.empty()
                prog_gif   = st.progress(0)
                frames_gif = []
                temp_dir_gif = tempfile.mkdtemp()
                norm_gif = _mcolors.Normalize(vmin=pmin_v, vmax=pmax_v)

                N_GRID = 200
                y_lim = (float(g['Y'].min()), float(g['Y'].max()))
                z_lim = (float(g['Z'].min()), float(g['Z'].max()))
                y_reg = np.linspace(y_lim[0], y_lim[1], N_GRID)
                z_reg = np.linspace(z_lim[0], z_lim[1], N_GRID)
                Yr, Zr = np.meshgrid(y_reg, z_reg)

                x_lim_min = float(np.min(g['x_arr'])) - ((pmax_v - pmin_v) * sc_gif)
                x_lim_max = float(np.max(g['x_arr']))
                y_lim_min, y_lim_max = y_lim
                z_lim_min, z_lim_max = z_lim

                if "4D" in tipo_gif and 'objeto_referencia_4d' in st.session_state:
                    obj_lim = st.session_state.get('objeto_referencia_base', st.session_state.objeto_referencia_4d)
                    cg_lim = st.session_state.get('modelo_cg', {'x': 0.0, 'y': 0.0, 'z': 0.0})
                    if 'x' in obj_lim and len(obj_lim['x']) > 0:
                        x_base = np.array(obj_lim['x']) - cg_lim['x']
                        y_base = np.array(obj_lim['y']) - cg_lim['y']
                        z_base = np.array(obj_lim['z']) - cg_lim['z']
                        
                        x_min_mod, x_max_mod = float('inf'), float('-inf')
                        y_min_mod, y_max_mod = float('inf'), float('-inf')
                        z_min_mod, z_max_mod = float('inf'), float('-inf')
                        
                        alphas_para_limites = np.linspace(aoa_min_v, aoa_max_v, n_pas_gif)
                        for a in alphas_para_limites:
                            xr, yr, zr = rotate_points(x_base, y_base, z_base, 0, a, 0)
                            x_min_mod = min(x_min_mod, float(np.min(xr)))
                            x_max_mod = max(x_max_mod, float(np.max(xr)))
                            y_min_mod = min(y_min_mod, float(np.min(yr)))
                            y_max_mod = max(y_max_mod, float(np.max(yr)))
                            z_min_mod = min(z_min_mod, float(np.min(zr)))
                            z_max_mod = max(z_max_mod, float(np.max(zr)))
                            
                        x_lim_min = min(x_lim_min, x_min_mod)
                        x_lim_max = max(x_lim_max, x_max_mod)
                        y_lim_min = min(y_lim_min, y_min_mod)
                        y_lim_max = max(y_lim_max, y_max_mod)
                        z_lim_min = min(z_lim_min, z_min_mod)
                        z_lim_max = max(z_lim_max, z_max_mod)

                try:
                    tri_interp = None
                    last_mask_sum = -1
                    for fi, alpha_i in enumerate(alpha_range_gif):
                        if not btn_preview:
                            status_gif.text(f"Frame {fi+1}/{n_pas_gif}  α={alpha_i:.1f}°")
                        else:
                            status_gif.text(f"Generando previsualización para α={alpha_i:.1f}°...")

                        idx_lo_g = max(0, min(int(np.searchsorted(g['aoa_arr'], alpha_i)) - 1, len(g['aoa_arr']) - 2))
                        idx_hi_g = idx_lo_g + 1
                        denom_t = g['aoa_arr'][idx_hi_g] - g['aoa_arr'][idx_lo_g]
                        t_g = (alpha_i - g['aoa_arr'][idx_lo_g]) / denom_t if denom_t != 0 else 0.0
                        P_g  = (1 - t_g) * g['grillas'][idx_lo_g] + t_g * g['grillas'][idx_hi_g]
                        x_g  = (1 - t_g) * g['x_arr'][idx_lo_g]  + t_g * g['x_arr'][idx_hi_g]
                        mask_g = ~np.isnan(P_g)

                        bg_color_mpl = '#0e1117' if "Oscuro" in vis_bg_a else '#ffffff'
                        text_color_mpl = 'white' if "Oscuro" in vis_bg_a else 'black'

                        if "2D" in tipo_gif:
                            fig_mpl, ax_mpl = _plt.subplots(figsize=(9, 7), facecolor=bg_color_mpl)
                            ax_mpl.set_facecolor(bg_color_mpl)

                            if mask_g.any():
                                Y_ok_g, Z_ok_g, P_ok_g = g['Y'][mask_g], g['Z'][mask_g], P_g[mask_g]
                                current_mask_sum = mask_g.sum()
                                if current_mask_sum != last_mask_sum or tri_interp is None:
                                    tri_interp = _Del(np.column_stack([Y_ok_g, Z_ok_g]))
                                    last_mask_sum = current_mask_sum
                                interp_func = _LND(tri_interp, P_ok_g)
                                Pr = interp_func(Yr, Zr)
                                cf = ax_mpl.contourf(Yr, Zr, Pr, levels=40, cmap='jet', norm=norm_gif)
                                ax_mpl.contour(Yr, Zr, Pr, levels=12, colors='white', linewidths=0.4, alpha=0.4)
                                cb = fig_mpl.colorbar(cf, ax=ax_mpl, label=g.get('var', 'Presión [Pa]'))
                                cb.ax.yaxis.label.set_color(text_color_mpl)
                                cb.ax.tick_params(colors=text_color_mpl)

                            y_mid_g = (y_lim[0] + y_lim[1]) / 2
                            ax_mpl.axvline(y_mid_g, color='cyan', lw=1.2, ls='--', alpha=0.7, label=f'Y_mid={y_mid_g:.0f}')
                            ax_mpl.set_xlim(y_lim); ax_mpl.set_ylim(z_lim)
                            ax_mpl.set_aspect('equal', 'box')
                            
                            ax_mpl.set_title(f"α = {alpha_i:.1f}°  |  Plano YZ — {g.get('var','Presión')}",
                                             color=text_color_mpl, fontsize=13, pad=10)
                                             
                            if not vis_ejes_a:
                                ax_mpl.axis('off')
                            else:
                                ax_mpl.set_xlabel("Y [mm]", color=text_color_mpl)
                                ax_mpl.set_ylabel("Z [mm]", color=text_color_mpl)
                                ax_mpl.tick_params(colors=text_color_mpl)
                                for sp in ax_mpl.spines.values(): sp.set_edgecolor('#444' if "Oscuro" in vis_bg_a else '#ccc')

                            ax_mpl.legend(fontsize=8, facecolor=bg_color_mpl, labelcolor=text_color_mpl,
                                          edgecolor='#444' if "Oscuro" in vis_bg_a else '#ccc', loc='upper right')

                        else:
                            fig_mpl = _plt.figure(figsize=(11, 8), facecolor=bg_color_mpl)
                            ax3 = fig_mpl.add_subplot(111, projection='3d')
                            ax3.set_facecolor(bg_color_mpl)

                            if mask_g.any():
                                Y_ok_g = g['Y'][mask_g]; Z_ok_g = g['Z'][mask_g]; P_ok_g = P_g[mask_g]
                                current_mask_sum = mask_g.sum()
                                if current_mask_sum != last_mask_sum or tri_interp is None:
                                    tri_interp = _Del(np.column_stack([Y_ok_g, Z_ok_g]))
                                    last_mask_sum = current_mask_sum
                                interp_func = _LND(tri_interp, P_ok_g)
                                Pr3 = interp_func(Yr, Zr)
                                P_ref_g = pmax_v
                                Xr3 = x_g - ((Pr3 - P_ref_g) * sc_gif)
                                facecolors_surf = _cm.get_cmap('jet')(norm_gif(Pr3))
                                ax3.plot_surface(Xr3, Yr, Zr, facecolors=facecolors_surf,
                                                 shade=False, alpha=0.9, antialiased=True, 
                                                 rstride=1, cstride=1, linewidth=0)

                            if 'objeto_referencia_4d' in st.session_state:
                                obj_b = st.session_state.get('objeto_referencia_base',
                                        st.session_state.objeto_referencia_4d)
                                cg_g2 = st.session_state.get('modelo_cg', {'x': 0.0, 'y': 0.0, 'z': 0.0})
                                xm_g2, ym_g2, zm_g2 = _pose_anim(obj_b, alpha_i, 0.0, cg_g2)
                                
                                if vis_modelo_a == "Puntos" or obj_b['type'] == 'scatter':
                                    c_mod, op_mod = '#888888', 0.5
                                    ax3.scatter(xm_g2, ym_g2, zm_g2, c=c_mod, s=1, alpha=op_mod, linewidths=0)
                                else:
                                    if vis_modelo_a == "Negro Mate":
                                        c_mod, op_mod = '#222222', 1.0
                                    elif vis_modelo_a == "Plata Metalizada":
                                        c_mod, op_mod = '#aaaaaa', 1.0
                                    else:
                                        c_mod, op_mod = '#5588cc', 0.35
                                    
                                    try:
                                        triangles = np.column_stack([obj_b['i'], obj_b['j'], obj_b['k']])
                                        ax3.plot_trisurf(xm_g2, ym_g2, zm_g2, triangles=triangles,
                                                         color=c_mod, alpha=op_mod, shade=True, linewidth=0, antialiased=True)
                                    except Exception:
                                        ax3.scatter(xm_g2, ym_g2, zm_g2, c=c_mod, s=1, alpha=op_mod, linewidths=0)

                            if not vis_ejes_a:
                                ax3.set_axis_off()
                            else:
                                ax3.set_xlabel("X", color=text_color_mpl, fontsize=9)
                                ax3.set_ylabel("Y", color=text_color_mpl, fontsize=9)
                                ax3.set_zlabel("Z", color=text_color_mpl, fontsize=9)
                                ax3.tick_params(colors=text_color_mpl, labelsize=7)
                                ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
                                edge_c = '#333' if "Oscuro" in vis_bg_a else '#ddd'
                                ax3.xaxis.pane.set_edgecolor(edge_c); ax3.yaxis.pane.set_edgecolor(edge_c); ax3.zaxis.pane.set_edgecolor(edge_c)
                                
                            ax3.set_xlim(x_lim_min, x_lim_max)
                            ax3.set_ylim(y_lim_min, y_lim_max)
                            ax3.set_zlim(z_lim_min, z_lim_max)
                            try:
                                ax3.set_box_aspect((x_lim_max - x_lim_min, y_lim_max - y_lim_min, z_lim_max - z_lim_min))
                            except AttributeError:
                                pass
                                
                            ax3.view_init(elev=elev_gif, azim=azim_gif)
                            title_obj = ax3.set_title(f"α = {alpha_i:.1f}°  |  Vista 4D (Elev: {elev_gif}°, Azim: {azim_gif}°)",
                                          color=text_color_mpl, fontsize=12, pad=12)
                            
                            ax3.invert_xaxis()
                            fig_mpl.tight_layout()

                        fp_gif = os.path.join(temp_dir_gif, f"frame_{fi:03d}.png")
                        fig_mpl.savefig(fp_gif, dpi=dpi_gif, bbox_inches='tight', facecolor=bg_color_mpl)
                        _plt.close(fig_mpl)
                        frames_gif.append(fp_gif)
                        if not btn_preview:
                            prog_gif.progress((fi + 1) / n_pas_gif)

                    if btn_preview:
                        status_gif.empty()
                        st.image(frames_gif[0], caption="Previsualización del Frame 1")
                    else:
                        status_gif.text("Compilando Animación...")
                        ext = ".mp4" if "MP4" in formato_anim else ".gif"
                        gif_path_anim = os.path.join(temp_dir_gif, f"animacion_4d{ext}")
                        images_gif = [imageio.imread(f) for f in frames_gif]
                        
                        if ext == ".mp4":
                            imageio.mimsave(gif_path_anim, images_gif, fps=fps_gif, macro_block_size=2)
                        else:
                            imageio.mimsave(gif_path_anim, images_gif, fps=fps_gif, loop=0)
                        
                        with open(gif_path_anim, "rb") as fg:
                            st.session_state['ultimo_gif_anim4d'] = fg.read()
                        st.session_state['ultimo_gif_nombre'] = f"animacion_2d{ext}" if "2D" in tipo_gif else f"animacion_4d_iso{ext}"
                        st.session_state['ultimo_gif_detalles'] = f"✅ Animación generada: {n_pas_gif} frames · {fps_gif} FPS · {tipo_gif} · {ext.upper()}"
                        st.session_state['ultimo_gif_ext'] = ext
                        frames_bytes_cache = []
                        for fp in frames_gif:
                            with open(fp, 'rb') as _fb: frames_bytes_cache.append(_fb.read())
                        st.session_state['anim4d_frames_cache'] = frames_bytes_cache
                        st.session_state['anim4d_session_meta'] = {
                            'tipo': tipo_gif, 'n_frames': n_pas_gif, 'fps': fps_gif,
                            'dpi': dpi_gif, 'aoa_min': aoa_min_v, 'aoa_max': aoa_max_v,
                            'variable': g.get('var', ''), 'sc_gif': sc_gif,
                            'planos': [d['name'] for d in g.get('items', [])],
                            'fecha': datetime.now().isoformat()
                        }

                except Exception as e_gif:
                    st.error(f"Error generando GIF: {e_gif}")
                    import traceback; st.code(traceback.format_exc())
                finally:
                    status_gif.empty(); prog_gif.empty()

            if 'ultimo_gif_anim4d' in st.session_state:
                st.success(st.session_state['ultimo_gif_detalles'])
                if st.session_state.get('ultimo_gif_ext') == '.mp4':
                    st.video(st.session_state['ultimo_gif_anim4d'], format="video/mp4", autoplay=True, loop=True)
                else:
                    st.image(st.session_state['ultimo_gif_anim4d'])
                st.download_button("📥 Descargar Animación", st.session_state['ultimo_gif_anim4d'],
                                   file_name=st.session_state['ultimo_gif_nombre'],
                                   mime="video/mp4" if st.session_state.get('ultimo_gif_ext') == '.mp4' else "image/gif",
                                   key="dl_gif_anim4d_persistent")

            if st.session_state.get('anim4d_frames_cache'):
                st.markdown("---")
                st.markdown("### 💾 Paso 5: Guardar Interpolación en Drive")
                st.info(f"📦 {len(st.session_state['anim4d_frames_cache'])} frames en caché. "
                        f"Guardálos en Drive para reutilizarlos sin regenerar.")
                c_save1, c_save2 = st.columns([2, 1])
                nombre_sesion_drive = c_save1.text_input(
                    "Nombre de la sesión (carpeta en Drive):",
                    value=f"Interp_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    key="nombre_sesion_anim_drive",
                    help="Será la carpeta: ANIMACION / [nombre] con todos los frames PNG + metadata."
                )
                if c_save2.button("☁️ Guardar en Drive", type="primary", use_container_width=True, key="btn_save_sesion_drive"):
                    if nombre_sesion_drive.strip():
                        with st.spinner(f"Subiendo {len(st.session_state['anim4d_frames_cache'])} frames a Drive..."):
                            try:
                                sid = auth.save_animation_session(
                                    st.session_state.username,
                                    nombre_sesion_drive.strip(),
                                    st.session_state['anim4d_frames_cache'],
                                    st.session_state.get('anim4d_session_meta', {})
                                )
                                if sid:
                                    st.success(f"✅ Sesión **'{nombre_sesion_drive}'** guardada en Drive "
                                               f"({len(st.session_state['anim4d_frames_cache'])} frames).")
                                else:
                                    st.error("Error al crear la carpeta en Drive.")
                            except Exception as _e_save:
                                st.error(f"Error: {_e_save}")
                    else:
                        st.warning("Ingresá un nombre para la sesión.")

            st.markdown("---")
            with st.expander("📂 Cargar Sesión de Frames Guardada (desde Drive)", expanded=False):
                st.caption("Cargá frames previamente guardados y recompilá el video con distintos FPS, formato o estética.")
                try:
                    sesiones_drive = auth.list_animation_sessions(st.session_state.username)
                except Exception:
                    sesiones_drive = []
                if not sesiones_drive:
                    st.info("No hay sesiones guardadas aún. Generá un GIF completo y guardalo en Drive (Paso 5).")
                else:
                    dict_ses = {f"{s[1]} ({s[2][:10] if s[2] else ''})": s for s in sesiones_drive}
                    sel_ses = st.selectbox("Seleccionar sesión:", list(dict_ses.keys()), key="sel_ses_drive")
                    c_load1, c_load2 = st.columns(2)
                    if c_load1.button("⬇️ Cargar frames en memoria", use_container_width=True, key="btn_load_ses"):
                        with st.spinner("Descargando frames desde Drive..."):
                            try:
                                frames_loaded, meta_loaded = auth.load_animation_session(dict_ses[sel_ses][0])
                                if frames_loaded:
                                    st.session_state['anim4d_frames_cache'] = frames_loaded
                                    st.session_state['anim4d_session_meta'] = meta_loaded
                                    st.success(f"✅ {len(frames_loaded)} frames cargados desde '{dict_ses[sel_ses][1]}'.")
                                    if meta_loaded:
                                        st.json(meta_loaded)
                                else:
                                    st.warning("La sesión está vacía o no tiene frames.")
                            except Exception as _e_load:
                                st.error(f"Error cargando sesión: {_e_load}")

                    if st.session_state.get('anim4d_frames_cache') and c_load2.button(
                        "🎬 Recompilar Video", type="primary", use_container_width=True, key="btn_recompile"
                    ):
                        cached = st.session_state['anim4d_frames_cache']
                        meta_c = st.session_state.get('anim4d_session_meta', {})
                        fps_rc   = st.slider("FPS (recompilación):", 1, 60, meta_c.get('fps', 18), key="fps_recomp")
                        fmt_rc   = st.selectbox("Formato:", ["GIF", "MP4"], index=1, key="fmt_recomp")
                        ext_rc   = ".mp4" if fmt_rc == "MP4" else ".gif"
                        with st.spinner("Compilando video desde frames en caché..."):
                            try:
                                import imageio as _iio
                                tmp_rc = tempfile.mkdtemp()
                                paths_rc = []
                                for _idx, _fb in enumerate(cached):
                                    _fp = os.path.join(tmp_rc, f"frame_{_idx:04d}.png")
                                    with open(_fp, 'wb') as _fh: _fh.write(_fb)
                                    paths_rc.append(_fp)
                                imgs_rc = [_iio.imread(p) for p in paths_rc]
                                out_rc  = os.path.join(tmp_rc, f"recompilado{ext_rc}")
                                if ext_rc == ".mp4":
                                    _iio.mimsave(out_rc, imgs_rc, fps=fps_rc, macro_block_size=2)
                                else:
                                    _iio.mimsave(out_rc, imgs_rc, fps=fps_rc, loop=0)
                                with open(out_rc, 'rb') as _fg:
                                    vid_rc = _fg.read()
                                st.session_state['ultimo_gif_anim4d'] = vid_rc
                                st.session_state['ultimo_gif_ext'] = ext_rc
                                n_rc = len(cached)
                                st.session_state['ultimo_gif_nombre'] = f"recompilado{ext_rc}"
                                st.session_state['ultimo_gif_detalles'] = f"✅ Recompilado desde caché: {n_rc} frames · {fps_rc} FPS · {fmt_rc}"
                                st.rerun()
                            except Exception as _e_rc:
                                st.error(f"Error recompilando: {_e_rc}")
