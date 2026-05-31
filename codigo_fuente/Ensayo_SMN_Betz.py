# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import re
from codigo_fuente import Auth_Manager as auth

def _calcular_valores_infinito_smn(txt_bytes, timestamp_str):
    try:
        df_inf = pd.read_csv(io.BytesIO(txt_bytes), sep=';', skip_blank_lines=True)
        df_inf.columns = [str(c).strip() for c in df_inf.columns]
        
        if len(df_inf.columns) > 2:
            first_col = df_inf.columns[0]
            df_inf["ts_clean"] = df_inf[first_col].astype(str).str.split(',').str[0].str.strip()
            df_inf["dt_val"] = pd.to_datetime(df_inf["ts_clean"], format='%d%m%y%H%M%S', errors='coerce')
            
            mask_failed = df_inf["dt_val"].isna()
            if mask_failed.any():
                df_inf.loc[mask_failed, "dt_val"] = pd.to_datetime(
                    df_inf.loc[mask_failed, "ts_clean"], format='%y%m%d%H%M%S', errors='coerce'
                )
            df_inf = df_inf.dropna(subset=["dt_val"])
            
            if df_inf.empty:
                return None
                
            ts_clean = str(timestamp_str).split(',')[0].strip()
            target_dt = pd.to_datetime(ts_clean, format='%d%m%y%H%M%S', errors='coerce')
            if pd.isna(target_dt):
                target_dt = pd.to_datetime(ts_clean, format='%y%m%d%H%M%S', errors='coerce')
            
            if pd.isna(target_dt):
                return None
                
            diffs = (df_inf["dt_val"] - target_dt).abs()
            idx = diffs.idxmin()
            row = df_inf.loc[idx]
            
            T = float(str(row.get("temp_baro", "15")).replace(",", "."))
            P_hpa = float(str(row.get("pres_baro", "1013.25")).replace(",", "."))
            HR = float(str(row.get("hrel", "50")).replace(",", "."))
            
            P_pa = P_hpa * 100.0
            T_kelvin = T + 273.15
            P_v_sat = 6.1078 * (10 ** ((7.5 * T)/(237.3 + T)))
            P_v = HR / 100.0 * P_v_sat
            P_d = P_hpa - P_v
            rho = (P_d * 100) / (287.058 * T_kelvin) + (P_v * 100) / (461.495 * T_kelvin)
            v_inf = float(str(row.get("velocidad", "0.0")).replace(",", "."))
            
            return {
                'rho_inf': float(rho),
                'v_inf': float(v_inf),
                'p_inf': float(P_pa),
                't_inf': float(T)
            }
    except Exception as e:
        st.warning(f"Error al vincular valores en el infinito: {e}")
    return None

def _extraer_aoa_smn(nombre):
    # Regex 1: Formato OAO0, OAOneg4, OAO5.3
    m = re.search(r'OAO(neg)?(\d+(?:[.,]\d+)?)', str(nombre), re.IGNORECASE)
    if m:
        val = float(str(m.group(2)).replace(',', '.'))
        return -val if m.group(1) else val
    
    # Regex 2: Formato 0AOA, neg4AOA, 5.3AOA
    m2 = re.search(r'(neg)?(\d+(?:[.,]\d+)?)\s*aoa', str(nombre), re.IGNORECASE)
    if m2:
        val = float(str(m2.group(2)).replace(',', '.'))
        return -val if m2.group(1) else val
    
    # Regex 3: Cualquier número seguido o precedido por alpha / aoa
    m3 = re.search(r'alpha_?(neg)?(\d+(?:[.,]\d+)?)', str(nombre), re.IGNORECASE)
    if m3:
        val = float(str(m3.group(2)).replace(',', '.'))
        return -val if m3.group(1) else val
        
    return 0.0

def calcular_anchos_integracion(y_vals):
    """
    Calcula los intervalos transversales (dy_i) asociados a cada punto i
    utilizando el método del punto medio.
    """
    N = len(y_vals)
    if N < 2:
        return np.zeros(N)
    
    dy = np.zeros(N)
    # Extremo inicial: primer punto medio a la mitad del intervalo entre y0 e y1
    dy[0] = (y_vals[1] - y_vals[0]) / 2.0
    
    # Puntos internos: distancia entre el punto medio posterior y el anterior
    for i in range(1, N - 1):
        dy[i] = (y_vals[i+1] - y_vals[i-1]) / 2.0
        
    # Extremo final: distancia entre el último punto medio e y_{N-1}
    dy[N-1] = (y_vals[N-1] - y_vals[N-2]) / 2.0
    
    return dy

def show_smn_betz():
    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            📊 MÉTODO DE BETZ - ENSAYO SMN
            </h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Cálculo del Coeficiente de Resistencia (Cd) por Pérdida de Momentum en la Estela
            </h2>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 2px solid #333; margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)

    # --- SESSION STATE INITIALIZATION ---
    if 'smn_archivos_memoria' not in st.session_state:
        st.session_state.smn_archivos_memoria = {}
    if 'betz_matriz_seleccionada' not in st.session_state:
        st.session_state.betz_matriz_seleccionada = pd.DataFrame()
    if 'betz_filename' not in st.session_state:
        st.session_state.betz_filename = ""
    if 'smn_v_inf' not in st.session_state: st.session_state.smn_v_inf = 17.5
    if 'smn_rho_inf' not in st.session_state: st.session_state.smn_rho_inf = 1.2
    if 'smn_p_inf' not in st.session_state: st.session_state.smn_p_inf = -94.0 
    if 'smn_t_inf' not in st.session_state: st.session_state.smn_t_inf = 15.0
    
    # Historial de sesión para las curvas de arrastre Cd vs AOA
    if 'smn_betz_historial' not in st.session_state:
        st.session_state.smn_betz_historial = pd.DataFrame(columns=[
            'Archivo', 'AOA [°]', 'Cd []', 'Arrastre [N/m]', 'V_inf [m/s]', 'rho_inf [kg/m³]'
        ])

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📥 Paso 1: Carga y Sincronización de Datos de Estela")
    st.caption("Cargá un archivo CSV de sonda multiagujero representativo del barrido transversal detrás del cilindro.")
    
    # 1. Cargador de archivos
    c_u1, c_u2 = st.columns(2)
    with c_u1:
        up_smn_betz = st.file_uploader("Subir archivo de estela SMN (.csv)", type=['csv'], key="up_smn_betz")
    with c_u2:
        up_infinito_betz = st.file_uploader("Subir archivo de condiciones atmosféricas (.txt)", type=['txt'], key="up_infinito_betz")

    timestamp_detectado = None
    aoa_autodetectado = 0.0
    
    if up_smn_betz:
        st.session_state.betz_filename = up_smn_betz.name
        # Detección del timestamp
        ts_m = re.search(r'(\d{10,14})', up_smn_betz.name)
        if ts_m:
            timestamp_detectado = ts_m.group(1)
            st.success(f"📅 Timestamp detectado en el nombre del archivo: `{timestamp_detectado}`")
        elif up_infinito_betz:
            timestamp_detectado = st.text_input("Ingresar Timestamp manualmente (DDMMYYHHMMSS):", key="ts_manual_betz")

        # Detección automática del AOA
        aoa_autodetectado = _extraer_aoa_smn(up_smn_betz.name)
        st.info(f"📐 Ángulo de Ataque (AOA) detectado del nombre del archivo: **{aoa_autodetectado:.1f}°**")

    # Vincular valores del infinito
    if up_infinito_betz and timestamp_detectado:
        inf_vals = _calcular_valores_infinito_smn(up_infinito_betz.read(), timestamp_detectado)
        if inf_vals:
            st.session_state.smn_v_inf = inf_vals['v_inf']
            st.session_state.smn_rho_inf = inf_vals['rho_inf']
            st.session_state.smn_p_inf = inf_vals['p_inf']
            st.session_state.smn_t_inf = inf_vals['t_inf']
            
            # Sincronizar con inputs de la UI
            st.session_state.smn_betz_v_inf_input = inf_vals['v_inf']
            st.session_state.smn_betz_rho_inf_input = inf_vals['rho_inf']
            st.session_state.smn_betz_p_inf_input = inf_vals['p_inf']
            st.session_state.smn_betz_t_inf_input = inf_vals['t_inf']
            
            st.success(f"✅ Valores atmosféricos en el infinito vinculados: V_∞={inf_vals['v_inf']} m/s, ρ_∞={inf_vals['rho_inf']:.4f} kg/m³")
            st.rerun()

    # Procesar archivo CSV
    if up_smn_betz:
        try:
            up_smn_betz.seek(0)
            try:
                df_raw = pd.read_csv(up_smn_betz, sep=';', decimal=',', encoding='utf-8')
            except UnicodeDecodeError:
                up_smn_betz.seek(0)
                df_raw = pd.read_csv(up_smn_betz, sep=';', decimal=',', encoding='latin-1')
            except Exception:
                up_smn_betz.seek(0)
                df_raw = pd.read_csv(up_smn_betz, sep=',', decimal='.', encoding='utf-8')
                if 'Posicion Sonda X[mm]' not in df_raw.columns:
                    up_smn_betz.seek(0)
                    df_raw = pd.read_csv(up_smn_betz, sep=',', decimal='.', encoding='latin-1')
            
            required = ['Posicion Sonda X[mm]', 'Posicion Sonda Y[mm]']
            if not all(col in df_raw.columns for col in required):
                st.error("❌ El archivo CSV no contiene columnas válidas de posición ('Posicion Sonda X[mm]' y 'Posicion Sonda Y[mm]').")
            else:
                st.success(f"✅ Archivo leído correctamente: {len(df_raw)} puntos de control cargados en memoria.")
                
                df_proc = pd.DataFrame()
                df_proc['Y'] = df_raw['Posicion Sonda X[mm]'].astype(float)
                df_proc['Z'] = df_raw['Posicion Sonda Y[mm]'].astype(float)
                
                var_mappings = {
                    'Presion_Est': 'Presion estatica [Pa]',
                    'Presion_Tot': 'Presion total [Pa]',
                    'Vel_Tot': 'Velocidad [m/seg]',
                    'Vx': 'Velocidad X [m/seg]',
                    'Vy': 'Velocidad Y [m/seg]',
                    'Vz': 'Velocidad Z [m/seg]'
                }
                
                for k, col in var_mappings.items():
                    found_col = next((c for c in df_raw.columns if c.replace(' ', '').lower() == col.replace(' ', '').lower() or k.lower() in c.lower()), None)
                    if found_col is not None:
                        df_proc[k] = df_raw[found_col].astype(float)
                    else:
                        df_proc[k] = 0.0
                
                name_mem = up_smn_betz.name.replace('.csv', '')
                st.session_state.smn_archivos_memoria[name_mem] = df_proc
                st.session_state.betz_matriz_seleccionada = df_proc
        except Exception as e:
            st.error(f"Error procesando CSV: {e}")

    # Configuración de condiciones de referencia (manual fallback)
    st.markdown("---")
    st.markdown("##### 🌐 Condiciones Atmosféricas de Referencia")
    c_inf1, c_inf2, c_inf3, c_inf4 = st.columns(4)
    st.session_state.smn_v_inf = c_inf1.number_input("Velocidad V_∞ [m/s]:", value=st.session_state.smn_v_inf, format="%.2f", key="smn_betz_v_inf_input")
    st.session_state.smn_rho_inf = c_inf2.number_input("Densidad ρ_∞ [kg/m³]:", value=st.session_state.smn_rho_inf, format="%.4f", key="smn_betz_rho_inf_input")
    st.session_state.smn_p_inf = c_inf3.number_input("Presión P_∞ [Pa]:", value=st.session_state.smn_p_inf, format="%.1f", key="smn_betz_p_inf_input")
    st.session_state.smn_t_inf = c_inf4.number_input("Temperatura T_∞ [°C]:", value=st.session_state.smn_t_inf, format="%.1f", key="smn_betz_t_inf_input")

    # Selección de datos desde la Base de Datos o la Memoria
    st.markdown("---")
    st.markdown("##### 📥 Seleccionar Plano de Estela para Análisis")
    modo_carga = st.radio("Origen de Datos:", ["🗄️ Base de Datos (Drive)", "🧠 Memoria de Sesión"], horizontal=True, key="betz_modo_carga")
    
    if modo_carga == "🗄️ Base de Datos (Drive)":
        try:
            drv_files = auth.get_smn_files_2d(st.session_state.username)
        except Exception as e:
            st.error(f"⚠️ Error cargando archivos desde Drive: {e}")
            drv_files = []
            
        if not drv_files:
            st.info("No hay archivos guardados en tu Google Drive.")
        else:
            dict_drv = {f[1]: f for f in drv_files}
            sel_drv = st.selectbox("Seleccionar Archivo en Drive:", ["-- Seleccionar --"] + list(dict_drv.keys()), key="sel_drv_betz")
            if sel_drv != "-- Seleccionar --":
                if 'last_drv_betz' not in st.session_state or st.session_state.last_drv_betz != sel_drv:
                    with st.spinner("Descargando plano de estela..."):
                        raw = auth.download_file_2d(dict_drv[sel_drv][0])
                        if raw:
                            df_active = pd.read_csv(io.BytesIO(raw), sep=';', decimal=',')
                            if 'Y' not in df_active.columns:
                                df_active = pd.read_csv(io.BytesIO(raw), sep=',', decimal='.')
                            
                            # Restaurar valores del infinito
                            if 'V_inf' in df_active.columns:
                                val_v = float(df_active['V_inf'].iloc[0])
                                val_rho = float(df_active['rho_inf'].iloc[0])
                                val_p = float(df_active['P_inf'].iloc[0])
                                val_t = float(df_active['T_inf'].iloc[0]) if 'T_inf' in df_active.columns else 15.0
                                
                                st.session_state.smn_v_inf = val_v
                                st.session_state.smn_rho_inf = val_rho
                                st.session_state.smn_p_inf = val_p
                                st.session_state.smn_t_inf = val_t
                                
                                st.session_state.smn_betz_v_inf_input = val_v
                                st.session_state.smn_betz_rho_inf_input = val_rho
                                st.session_state.smn_betz_p_inf_input = val_p
                                st.session_state.smn_betz_t_inf_input = val_t
                            
                            st.session_state.betz_matriz_seleccionada = df_active
                            st.session_state.betz_filename = sel_drv
                            st.session_state.last_drv_betz = sel_drv
                            st.rerun()
    else:
        if not st.session_state.smn_archivos_memoria:
            st.warning("⚠️ No hay planos cargados en la memoria de la sesión.")
        else:
            sel_mem = st.selectbox("Seleccionar Plano en Memoria:", list(st.session_state.smn_archivos_memoria.keys()), key="sel_mem_betz")
            if st.button("📥 Cargar Plano de Memoria", use_container_width=True):
                st.session_state.betz_matriz_seleccionada = st.session_state.smn_archivos_memoria[sel_mem]
                st.session_state.betz_filename = sel_mem
                df_active = st.session_state.smn_archivos_memoria[sel_mem]
                if 'V_inf' in df_active.columns:
                    val_v = float(df_active['V_inf'].iloc[0])
                    val_rho = float(df_active['rho_inf'].iloc[0])
                    val_p = float(df_active['P_inf'].iloc[0])
                    val_t = float(df_active['T_inf'].iloc[0]) if 'T_inf' in df_active.columns else 15.0
                    
                    st.session_state.smn_v_inf = val_v
                    st.session_state.smn_rho_inf = val_rho
                    st.session_state.smn_p_inf = val_p
                    st.session_state.smn_t_inf = val_t
                    
                    st.session_state.smn_betz_v_inf_input = val_v
                    st.session_state.smn_betz_rho_inf_input = val_rho
                    st.session_state.smn_betz_p_inf_input = val_p
                    st.session_state.smn_betz_t_inf_input = val_t
                st.success("✅ Plano de memoria cargado.")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # --- PROCESS AND INTEGRATE ---
    if st.session_state.betz_matriz_seleccionada is not None and not st.session_state.betz_matriz_seleccionada.empty:
        df_full = st.session_state.betz_matriz_seleccionada.copy()
        
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("⚙️ Paso 2: Configuración del Algoritmo e Integración Numérica")
        
        c_param1, c_param2 = st.columns(2)
        
        # Extracción inicial del AOA para mostrar en input
        default_aoa = _extraer_aoa_smn(st.session_state.betz_filename) if st.session_state.betz_filename else 0.0
        
        with c_param1:
            st.markdown("##### 📏 Parámetros del Cilindro")
            d_cyl = st.number_input("Diámetro del Cilindro (d) [mm]:", value=50.0, step=1.0, format="%.1f")
            l_cyl = st.number_input("Longitud del Cilindro (L) [mm]:", value=300.0, step=10.0, format="%.1f")
            aoa_val = st.number_input("Ángulo de Ataque (α) [°]:", value=default_aoa, step=0.5, format="%.1f")
            
        with c_param2:
            st.markdown("##### 🧬 Configuración del Método Físico")
            # Selección de eje transversal de barrido
            y_range = df_full['Y'].max() - df_full['Y'].min()
            z_range = df_full['Z'].max() - df_full['Z'].min()
            default_axis = 'Y' if y_range >= z_range else 'Z'
            
            sweep_axis = st.selectbox(
                "Eje transversal de barrido (Estela):", 
                ['Y', 'Z'], 
                index=0 if default_axis == 'Y' else 1,
                help="El eje físico transversal a la estela por donde se integra el déficit. Por defecto se elige el que tiene mayor rango."
            )
            
            # Opción de presión estática uniforme
            use_const_ps = st.checkbox(
                "Asumir presión estática uniforme en la estela (P_s = P_∞)",
                value=True,
                help="Simplificación clásica del método de Betz donde se considera que la presión estática de la estela ya se igualó a la del flujo libre."
            )
            
            # Opción para calibración de velocidad de corriente libre
            vinf_mode = st.radio(
                "Velocidad de Corriente Libre (U_∞):",
                ["Manual / Atmosférica de Referencia", "Calcular desde los bordes del barrido"],
                index=0,
                help="Permite tomar la velocidad de referencia de los sensores del infinito o calcularla promediando los extremos del barrido transversal."
            )
            
        # --- FILTRADO DE REBANADA / SLICING PARA PLANOS 2D ---
        other_axis = 'Z' if sweep_axis == 'Y' else 'Y'
        other_vals = df_full[other_axis].unique()
        
        # Si el plano contiene múltiples perfiles de barrido
        if len(other_vals) > 1:
            st.markdown("---")
            st.markdown("##### 🔪 Detección de Plano 2D: Selección de Rebanada")
            st.warning(f"Se detectaron múltiples posiciones de '{other_axis}' en el archivo. Seleccioná una coordenada específica para el barrido 1D de Betz.")
            
            sel_slice = st.select_slider(
                f"Seleccionar Rebanada en {other_axis} [mm]:",
                options=sorted(other_vals),
                value=sorted(other_vals)[len(other_vals)//2]
            )
            
            # Tolerancia para filtrar decimales
            df_slice = df_full[np.abs(df_full[other_axis] - sel_slice) < 0.1].copy()
            st.info(f"Puntos en la rebanada seleccionada: **{len(df_slice)}**")
        else:
            df_slice = df_full.copy()
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # --- CÁLCULO NUMÉRICO DE BETZ ---
        if len(df_slice) >= 3:
            # Limpiar duplicados y ordenar por el eje de barrido transversal
            df_clean = df_slice.dropna(subset=[sweep_axis, 'Presion_Tot', 'Presion_Est']).copy()
            df_clean = df_clean.drop_duplicates(subset=[sweep_axis]).sort_values(sweep_axis)
            
            y_pts_mm = df_clean[sweep_axis].values
            y_pts_m = y_pts_mm / 1000.0  # Pasar a metros para integración en SI
            
            pt_vals = df_clean['Presion_Tot'].values
            ps_vals = df_clean['Presion_Est'].values
            
            rho_inf = st.session_state.smn_rho_inf
            p_inf = st.session_state.smn_p_inf
            
            # Velocidad de corriente libre
            if vinf_mode == "Calcular desde los bordes del barrido":
                # Promediar los extremos (primeros 2 y últimos 2 puntos)
                edges_pt = np.concatenate([pt_vals[:2], pt_vals[-2:]])
                edges_ps = np.concatenate([ps_vals[:2], ps_vals[-2:]])
                edges_q = np.maximum(0.0, edges_pt - edges_ps)
                edges_u = np.sqrt(2.0 * edges_q / rho_inf)
                u_inf = float(np.mean(edges_u))
                st.success(f"Velocidad de corriente libre calculada de los bordes: **U_∞ = {u_inf:.3f} m/s**")
            else:
                u_inf = float(st.session_state.smn_v_inf)
            
            # Aplicar Bernoulli para velocidad local u(y)
            if use_const_ps:
                # Usar presión estática uniforme P_s = P_∞
                # st.session_state.smn_p_inf es la presión en el infinito, pero típicamente los transductores de estela miden presión diferencial o relativa.
                # Si st.session_state.smn_p_inf es absoluta y Pt es relativa, tenemos cuidado.
                # Por Bernoulli local: q_local = Pt(y) - Ps_elegida
                # Si asumimos estática uniforme, q_local = Pt(y) - Ps_borde (o Pt - p_inf si p_inf es estática)
                # Una forma muy robusta es estimar la presión estática libre como el promedio en los extremos de ps_vals
                ps_ref = np.mean(np.concatenate([ps_vals[:2], ps_vals[-2:]])) if len(ps_vals) >= 4 else p_inf
                q_local = np.maximum(0.0, pt_vals - ps_ref)
            else:
                # Usar la estática local real en la estela
                q_local = np.maximum(0.0, pt_vals - ps_vals)
                
            u_local = np.sqrt(2.0 * q_local / rho_inf)
            
            # Déficit de Momentum: rho * u * (U_∞ - u)
            momentum_deficit = rho_inf * u_local * (u_inf - u_local)
            
            # Calcular intervalos de integración transversales dy_i usando el método de puntos medios
            dy_m = calcular_anchos_integracion(y_pts_m)
            
            # Integrar resistencia por unidad de longitud (Drag per unit span)
            # D' = sum (deficit_i * dy_i)
            drag_per_span_elements = momentum_deficit * dy_m
            drag_per_span = np.sum(drag_per_span_elements)
            
            # Fuerza de arrastre total
            drag_total = drag_per_span * (l_cyl / 1000.0)
            
            # Coeficiente de Resistencia Cd
            q_inf = 0.5 * rho_inf * (u_inf ** 2)
            d_meters = d_cyl / 1000.0
            l_meters = l_cyl / 1000.0
            
            cd_val = 0.0
            if q_inf > 0 and d_meters > 0:
                cd_val = drag_per_span / (q_inf * d_meters)
                
            # --- CARD DE MÉTRICAS PRINCIPALES ---
            st.markdown("### 📈 Coeficiente de Resistencia de Betz")
            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
            
            with c_m1:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 1.2rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.6); text-transform: uppercase;">Coeficiente Cd</div>
                    <div style="font-size: 3rem; font-weight: 900; color: #10b981; margin: 0.5rem 0;">{cd_val:.4f}</div>
                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.4);">Adimensional</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c_m2:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 1.2rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.6); text-transform: uppercase;">Arrastre Unitario (D')</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #3b82f6; margin: 0.8rem 0;">{drag_per_span:.4f} <span style="font-size: 1.2rem;">N/m</span></div>
                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.4);">Por metro de envergadura</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c_m3:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 1.2rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.6); text-transform: uppercase;">Fuerza Total (D)</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #f59e0b; margin: 0.8rem 0;">{drag_total:.4f} <span style="font-size: 1.2rem;">N</span></div>
                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.4);">Cilindro de L={l_cyl} mm</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c_m4:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 1.2rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.6); text-transform: uppercase;">Ángulo de Ataque (α)</div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: #a855f7; margin: 0.6rem 0;">{aoa_val:.1f}°</div>
                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.4);">De: {st.session_state.betz_filename[:15]}...</div>
                </div>
                """, unsafe_allow_html=True)

            # Botón de guardado en el historial
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Guardar Punto de Arrastre en Historial de Sesión", use_container_width=True, type="primary"):
                # Evitar duplicados del mismo archivo y AOA en el historial
                already_exists = not st.session_state.smn_betz_historial[
                    (st.session_state.smn_betz_historial['Archivo'] == st.session_state.betz_filename) & 
                    (st.session_state.smn_betz_historial['AOA [°]'] == aoa_val)
                ].empty
                
                if already_exists:
                    st.warning("⚠️ Este punto (Archivo y AOA) ya ha sido registrado en el historial de esta sesión.")
                else:
                    nuevo_registro = pd.DataFrame([{
                        'Archivo': st.session_state.betz_filename,
                        'AOA [°]': aoa_val,
                        'Cd []': cd_val,
                        'Arrastre [N/m]': drag_per_span,
                        'V_inf [m/s]': u_inf,
                        'rho_inf [kg/m³]': rho_inf
                    }])
                    st.session_state.smn_betz_historial = pd.concat(
                        [st.session_state.smn_betz_historial, nuevo_registro], ignore_index=True
                    )
                    st.success(f"✅ Punto guardado correctamente en el historial: α={aoa_val}°, Cd={cd_val:.4f}")
                    st.rerun()

            # --- GRÁFICOS INTERACTIVOS EN PESTAÑAS ---
            st.markdown("<br>", unsafe_allow_html=True)
            t_plot1, t_plot2, t_plot3 = st.tabs([
                "🌫️ Pérdida de Momentum (Integral de Resistencia)", 
                "⚡ Perfil de Velocidades (u vs U_∞)", 
                "🎈 Distribución de Presiones (Pt & Ps)"
            ])
            
            # Pestaña 1: Integral de Pérdida de Momentum
            with t_plot1:
                fig1 = go.Figure()
                # Sombreado bajo la curva del déficit de momentum
                fig1.add_trace(go.Scatter(
                    x=y_pts_mm, y=momentum_deficit,
                    mode='lines+markers',
                    name='Déficit de Momentum local',
                    line=dict(color='#ef4444', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(239, 68, 68, 0.15)',
                    hovertemplate='Y: %{x:.2f} mm<br>Déficit: %{y:.3f} N/m³<extra></extra>'
                ))
                
                fig1.update_layout(
                    title="Distribución del Déficit de Cantidad de Movimiento (Estela)",
                    xaxis_title=f"{sweep_axis} (Transversal) [mm]",
                    yaxis_title="Déficit de Momentum [N/m³]",
                    height=500,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    hovermode='x'
                )
                st.plotly_chart(fig1, use_container_width=True)
                st.caption("💡 El sombreado rojo representa el área física integrada. A mayor déficit de momentum (área sombreada), mayor es la resistencia total del cilindro.")

            # Pestaña 2: Perfil de Velocidades
            with t_plot2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=y_pts_mm, y=u_local,
                    mode='lines+markers',
                    name='Velocidad local en la estela u(y)',
                    line=dict(color='#3b82f6', width=3),
                    hovertemplate='Y: %{x:.2f} mm<br>Velocidad: %{y:.2f} m/s<extra></extra>'
                ))
                fig2.add_trace(go.Scatter(
                    x=y_pts_mm, y=np.full_like(y_pts_mm, u_inf),
                    mode='lines',
                    name='Corriente Libre U_∞',
                    line=dict(color='#10b981', dash='dash', width=2),
                    hovertemplate='Corriente Libre: %{y:.2f} m/s<extra></extra>'
                ))
                
                fig2.update_layout(
                    title="Perfil del Déficit de Velocidad en la Estela",
                    xaxis_title=f"{sweep_axis} (Transversal) [mm]",
                    yaxis_title="Velocidad [m/s]",
                    height=500,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    hovermode='x'
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Pestaña 3: Distribución de Presiones
            with t_plot3:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=y_pts_mm, y=pt_vals,
                    mode='lines+markers',
                    name='Presión Total (Pt)',
                    line=dict(color='#c084fc', width=2),
                    hovertemplate='Y: %{x:.2f} mm<br>Pt: %{y:.1f} Pa<extra></extra>'
                ))
                fig3.add_trace(go.Scatter(
                    x=y_pts_mm, y=ps_vals,
                    mode='lines+markers',
                    name='Presión Estática (Ps)',
                    line=dict(color='#60a5fa', width=2),
                    hovertemplate='Y: %{x:.2f} mm<br>Ps: %{y:.1f} Pa<extra></extra>'
                ))
                
                fig3.update_layout(
                    title="Distribución de Presiones en la Estela",
                    xaxis_title=f"{sweep_axis} (Transversal) [mm]",
                    yaxis_title="Presión [Pa]",
                    height=500,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    hovermode='x'
                )
                st.plotly_chart(fig3, use_container_width=True)

            # --- CONSIDERACIONES CRÍTICAS DE LABORATORIO ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("⚠️ Consideraciones Críticas y Validación de Ensayos")
            
            col_cr1, col_cr2 = st.columns(2)
            
            with col_cr1:
                st.markdown("##### 📏 Límites e Integración en Extremos")
                u_esq_izq = u_local[0]
                u_esq_der = u_local[-1]
                rec_izq = (u_esq_izq / u_inf) * 100
                rec_der = (u_esq_der / u_inf) * 100
                
                limite_sano = rec_izq >= 95.0 and rec_der >= 95.0
                
                if limite_sano:
                    st.success(f"✅ **Límites de Estela Óptimos**:\n- Extremo Izquierdo: {rec_izq:.1f}% de U_∞\n- Extremo Derecho: {rec_der:.1f}% de U_∞\n\nEl barrido cubre completamente la zona de perturbación.")
                else:
                    st.warning(f"⚠️ **Recuperación incompleta en los extremos del barrido**:\n- Izquierdo: {rec_izq:.1f}% de U_∞\n- Derecho: {rec_der:.1f}% de U_∞\n\n**Recomendación**: La velocidad no regresó completamente a la corriente libre en los extremos. La resistencia integrada (Cd) podría estar ligeramente subestimada. Es aconsejable ampliar el rango transversal del barrido en el túnel de viento.")
            
            with col_cr2:
                st.markdown("##### ⚖️ Simetría del Perfil de la Estela")
                # Centroide de momentum
                num_cent = np.sum(y_pts_mm * drag_per_span_elements)
                den_cent = np.sum(drag_per_span_elements)
                y_cent = num_cent / den_cent if den_cent > 0 else 0.0
                
                # Centro geométrico del barrido
                y_geom = (y_pts_mm[0] + y_pts_mm[-1]) / 2.0
                desfase = np.abs(y_cent - y_geom)
                
                # Índice de Simetría dividiendo en lados izquierdo y derecho del centroide
                izq_mask = y_pts_mm < y_cent
                der_mask = y_pts_mm >= y_cent
                drag_izq = np.sum(drag_per_span_elements[izq_mask])
                drag_der = np.sum(drag_per_span_elements[der_mask])
                
                if drag_izq + drag_der > 0:
                    ind_sim = (1.0 - (np.abs(drag_izq - drag_der) / (drag_izq + drag_der))) * 100
                else:
                    ind_sim = 100.0
                
                st.markdown(f"""
                - **Centroide de la estela (Déficit)**: `{y_cent:.2f} mm`
                - **Centro Geométrico del Barrido**: `{y_geom:.2f} mm`
                - **Desfase del centroide**: `{desfase:.2f} mm`
                """)
                
                if ind_sim >= 85.0:
                    st.success(f"⚖️ **Alta simetría del perfil**: `{ind_sim:.1f}%` de coincidencia entre ambas mitades del barrido.")
                else:
                    st.warning(f"⚠️ **Perfil asimétrico**: `{ind_sim:.1f}%` de coincidencia entre ambas mitades.\n\nRevisar posible desalineación física de la sonda con el flujo o inestabilidad severa del túnel.")
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            st.warning("⚠️ Se necesitan al menos 3 puntos de medición válidos en el barrido transversal para realizar los cálculos del método de Betz.")

    # --- PESTAÑA HISTORIAL Y POLAR DE ARRASTRE Cd vs AOA ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📊 Historial de Puntos y Curva Polar de Arrastre (Cd vs α)")
    
    if st.session_state.smn_betz_historial.empty:
        st.info("Aún no has guardado puntos en el historial de esta sesión. Realizá cálculos de Betz y guardalos para graficar la polar de arrastre del cilindro.")
    else:
        c_h1, c_h2 = st.columns([1.2, 1.8])
        
        with c_h1:
            st.markdown("##### Puntos Registrados")
            st.dataframe(st.session_state.smn_betz_historial[[
                'AOA [°]', 'Cd []', 'Arrastre [N/m]'
            ]], use_container_width=True, hide_index=True)
            
            if st.button("🗑️ Limpiar Historial de Puntos", use_container_width=True):
                st.session_state.smn_betz_historial = pd.DataFrame(columns=[
                    'Archivo', 'AOA [°]', 'Cd []', 'Arrastre [N/m]', 'V_inf [m/s]', 'rho_inf [kg/m³]'
                ])
                st.success("Historial de la sesión borrado.")
                st.rerun()
                
        with c_h2:
            st.markdown("##### Gráfico Polar: Cd vs AOA (α)")
            hist_df = st.session_state.smn_betz_historial.copy().sort_values('AOA [°]')
            
            fig_polar = go.Figure()
            fig_polar.add_trace(go.Scatter(
                x=hist_df['AOA [°]'], y=hist_df['Cd []'],
                mode='lines+markers',
                marker=dict(size=8, color='#a855f7'),
                line=dict(color='#a855f7', width=3),
                name='Arrastre del Cilindro',
                hovertemplate='AOA: %{x:.1f}°<br>Cd: %{y:.4f}<extra></extra>'
            ))
            
            fig_polar.update_layout(
                title="Curva del Coeficiente de Resistencia vs Ángulo de Ataque (Cd vs α)",
                xaxis_title="Ángulo de Ataque (α) [°]",
                yaxis_title="Coeficiente de Resistencia Cd []",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )
            st.plotly_chart(fig_polar, use_container_width=True)
            
    st.markdown("</div>", unsafe_allow_html=True)
