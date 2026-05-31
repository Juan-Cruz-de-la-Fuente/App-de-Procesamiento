# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import re
from scipy.interpolate import griddata
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

def calcular_anchos_integracion(coords_1d):
    """
    Calcula los intervalos de integracion (d_i) para un array de coordenadas 1D ordenado
    utilizando el metodo del punto medio.
    """
    N = len(coords_1d)
    if N == 0:
        return np.array([])
    if N == 1:
        return np.array([1.0])  # Ancho unitario de fallback si hay un solo punto
    
    d = np.zeros(N)
    d[0] = (coords_1d[1] - coords_1d[0]) / 2.0
    for i in range(1, N - 1):
        d[i] = (coords_1d[i+1] - coords_1d[i-1]) / 2.0
    d[N-1] = (coords_1d[N-1] - coords_1d[N-2]) / 2.0
    return d

def show_smn_betz():
    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            📊 MÉTODO DE BETZ - EN PLANO Y-Z
            </h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Integración de Resistencia de Estela Bidimensional 2D
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
            'Archivo', 'AOA [°]', 'Cd []', 'Arrastre [N]', 'V_inf [m/s]', 'rho_inf [kg/m³]'
        ])

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📥 Paso 1: Carga y Sincronización de Datos de Estela")
    st.caption("Cargá un archivo CSV de sonda multiagujero que contenga el mapeo bidimensional en el plano transversal Y-Z.")
    
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
                st.error("❌ El archivo CSV no contiene columnas válidas de posición.")
            else:
                st.success(f"✅ Archivo leído correctamente: {len(df_raw)} puntos de control en el plano.")
                
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

    # --- PROCESS AND INTEGRATE IN Y-Z PLANE ---
    if st.session_state.betz_matriz_seleccionada is not None and not st.session_state.betz_matriz_seleccionada.empty:
        df_full = st.session_state.betz_matriz_seleccionada.copy()
        
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("⚙️ Paso 2: Configuración del Algoritmo e Integración Numérica Y-Z")
        
        c_param1, c_param2 = st.columns(2)
        
        # Extracción inicial del AOA para mostrar en input
        default_aoa = _extraer_aoa_smn(st.session_state.betz_filename) if st.session_state.betz_filename else 0.0
        
        with c_param1:
            st.markdown("##### 📏 Geometría del Modelo Ensayo")
            s_ref = st.number_input(
                "Área de Referencia (S_ref) [cm²]:", 
                value=150.0, 
                step=10.0, 
                format="%.1f",
                help="El área de referencia del modelo ensayado utilizada para adimensionalizar y obtener el coeficiente de resistencia Cd."
            )
            aoa_val = st.number_input("Ángulo de Ataque (α) [°]:", value=default_aoa, step=0.5, format="%.1f")
            
        with c_param2:
            st.markdown("##### 🧬 Configuración de Presiones y Velocidad")
            
            # Opción de presión estática uniforme
            use_const_ps = st.checkbox(
                "Asumir presión estática uniforme en la estela (P_s = P_∞)",
                value=True,
                help="Simplificación clásica del método de Betz donde se considera que la presión estática de la estela ya se igualó a la del flujo libre."
            )
            
            # Opción para calibración de velocidad de corriente libre
            vinf_mode = st.radio(
                "Velocidad de Corriente Libre (U_∞):",
                ["Manual / Atmosférica de Referencia", "Calcular desde los bordes del barrido Y-Z"],
                index=0,
                help="Permite tomar la velocidad de referencia de los sensores del infinito o calcularla promediando los extremos del barrido transversal."
            )
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # --- CÁLCULO NUMÉRICO DE BETZ EN 2D ---
        df_clean = df_full.dropna(subset=['Y', 'Z', 'Presion_Tot', 'Presion_Est']).copy()
        
        if len(df_clean) >= 4:
            # 1. Obtener coordenadas únicas y ordenadas en Y y Z
            y_uniq = np.sort(df_clean['Y'].unique())
            z_uniq = np.sort(df_clean['Z'].unique())
            
            # 2. Calcular los anchos de punto medio para cada eje único
            dy_uniq = calcular_anchos_integracion(y_uniq)
            dz_uniq = calcular_anchos_integracion(z_uniq)
            
            # 3. Crear diccionarios para mapeo rápido
            y_to_dy = dict(zip(y_uniq, dy_uniq))
            z_to_dz = dict(zip(z_uniq, dz_uniq))
            
            # 4. Calcular el área elemental dA_i para cada punto individual en metros cuadrados
            df_clean['dy_val'] = df_clean['Y'].map(y_to_dy)
            df_clean['dz_val'] = df_clean['Z'].map(z_to_dz)
            df_clean['dA_m2'] = (df_clean['dy_val'] / 1000.0) * (df_clean['dz_val'] / 1000.0)
            
            # Puntos y valores físicos
            rho_inf = st.session_state.smn_rho_inf
            p_inf = st.session_state.smn_p_inf
            
            pt_vals = df_clean['Presion_Tot'].values
            ps_vals = df_clean['Presion_Est'].values
            
            # Velocidad de corriente libre
            if vinf_mode == "Calcular desde los bordes del barrido Y-Z":
                # Definir puntos en el borde exterior del plano
                boundary_mask = (df_clean['Y'] == y_uniq[0]) | (df_clean['Y'] == y_uniq[-1]) | \
                                (df_clean['Z'] == z_uniq[0]) | (df_clean['Z'] == z_uniq[-1])
                edges_pt = df_clean.loc[boundary_mask, 'Presion_Tot'].values
                edges_ps = df_clean.loc[boundary_mask, 'Presion_Est'].values
                edges_q = np.maximum(0.0, edges_pt - edges_ps)
                edges_u = np.sqrt(2.0 * edges_q / rho_inf)
                u_inf = float(np.mean(edges_u)) if len(edges_u) > 0 else float(st.session_state.smn_v_inf)
                st.success(f"Velocidad de corriente libre calculada de los bordes del plano: **U_∞ = {u_inf:.3f} m/s**")
            else:
                u_inf = float(st.session_state.smn_v_inf)
            
            # Bernoulli local para u(y, z)
            if use_const_ps:
                # Estimar presión estática libre como el promedio en los bordes
                boundary_mask = (df_clean['Y'] == y_uniq[0]) | (df_clean['Y'] == y_uniq[-1]) | \
                                (df_clean['Z'] == z_uniq[0]) | (df_clean['Z'] == z_uniq[-1])
                ps_ref = np.mean(df_clean.loc[boundary_mask, 'Presion_Est'].values) if boundary_mask.any() else p_inf
                q_local = np.maximum(0.0, pt_vals - ps_ref)
            else:
                q_local = np.maximum(0.0, pt_vals - ps_vals)
                
            u_local = np.sqrt(2.0 * q_local / rho_inf)
            
            # Déficit local de momentum
            momentum_deficit = rho_inf * u_local * (u_inf - u_local)
            df_clean['u_local'] = u_local
            df_clean['deficit'] = momentum_deficit
            
            # Integrar resistencia total: Drag = sum (deficit_i * dA_i)
            df_clean['drag_elemental'] = momentum_deficit * df_clean['dA_m2']
            drag_total = float(df_clean['drag_elemental'].sum())
            
            # Coeficiente de Resistencia Cd
            q_inf = 0.5 * rho_inf * (u_inf ** 2)
            s_ref_m2 = (s_ref / 10000.0)  # Convertir cm² a m²
            
            cd_val = 0.0
            if q_inf > 0 and s_ref_m2 > 0:
                cd_val = drag_total / (q_inf * s_ref_m2)
                
            # --- CARD DE MÉTRICAS PRINCIPALES ---
            st.markdown("### 📈 Coeficiente de Resistencia 2D (Betz)")
            c_m1, c_m2, c_m3 = st.columns(3)
            
            with c_m1:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 0.95rem; color: rgba(255,255,255,0.6); text-transform: uppercase;">Coeficiente de Resistencia Cd</div>
                    <div style="font-size: 3.5rem; font-weight: 900; color: #10b981; margin: 0.5rem 0;">{cd_val:.4f}</div>
                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.4);">Base S_ref = {s_ref} cm²</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c_m2:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 0.95rem; color: rgba(255,255,255,0.6); text-transform: uppercase;">Fuerza de Arrastre Total (D)</div>
                    <div style="font-size: 3rem; font-weight: 800; color: #3b82f6; margin: 0.7rem 0;">{drag_total:.4f} <span style="font-size: 1.5rem;">N</span></div>
                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.4);">Integración en plano Y-Z</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c_m3:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 0.95rem; color: rgba(255,255,255,0.6); text-transform: uppercase;">Ángulo de Ataque (α)</div>
                    <div style="font-size: 3rem; font-weight: 800; color: #a855f7; margin: 0.7rem 0;">{aoa_val:.1f}°</div>
                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.4);">Extraído: {st.session_state.betz_filename[:15]}...</div>
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
                        'Arrastre [N]': drag_total,
                        'V_inf [m/s]': u_inf,
                        'rho_inf [kg/m³]': rho_inf
                    }])
                    st.session_state.smn_betz_historial = pd.concat(
                        [st.session_state.smn_betz_historial, nuevo_registro], ignore_index=True
                    )
                    st.success(f"✅ Punto guardado correctamente en el historial: α={aoa_val}°, Cd={cd_val:.4f}")
                    st.rerun()

            # --- MAPAS DE CONTORNO EN PESTAÑAS 2D ---
            st.markdown("<br>", unsafe_allow_html=True)
            t_plot1, t_plot2, t_plot3 = st.tabs([
                "🌫️ Mapa 2D de Pérdida de Momentum", 
                "⚡ Perfil de Velocidades de la Estela u(y,z)", 
                "🎈 Distribución de Presión Total Pt(y,z)"
            ])
            
            # Preparar interpolación 2D para graficación suave
            y_coords = df_clean['Y'].values
            z_coords = df_clean['Z'].values
            
            grid_y = np.linspace(y_coords.min(), y_coords.max(), 100)
            grid_z = np.linspace(z_coords.min(), z_coords.max(), 100)
            Gy, Gz = np.meshgrid(grid_y, grid_z)
            
            # Pestaña 1: Déficit de momentum en 2D
            with t_plot1:
                G_def = griddata((y_coords, z_coords), momentum_deficit, (Gy, Gz), method='cubic')
                fig1 = go.Figure()
                fig1.add_trace(go.Contour(
                    x=grid_y, y=grid_z, z=G_def,
                    colorscale='Turbo',
                    colorbar=dict(title="Déficit [N/m³]"),
                    hovertemplate='Y: %{x:.1f} mm<br>Z: %{y:.1f} mm<br>Déficit: %{z:.2f} N/m³<extra></extra>'
                ))
                fig1.add_trace(go.Scatter(
                    x=y_coords, y=z_coords,
                    mode='markers',
                    marker=dict(size=3, color='white', opacity=0.4),
                    name='Puntos medidos'
                ))
                fig1.update_layout(
                    title="Mapeo 2D del Déficit de Cantidad de Movimiento (Estela)",
                    xaxis_title="Y (Envergadura) [mm]",
                    yaxis_title="Z (Altura) [mm]",
                    height=600,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white")
                )
                fig1.update_xaxes(scaleanchor="y", scaleratio=1)
                st.plotly_chart(fig1, use_container_width=True)
                st.caption("💡 La zona con mayor intensidad de color representa el núcleo de la estela perturbada donde la resistencia se concentra.")

            # Pestaña 2: Campo de velocidades en 2D
            with t_plot2:
                G_u = griddata((y_coords, z_coords), u_local, (Gy, Gz), method='cubic')
                fig2 = go.Figure()
                fig2.add_trace(go.Contour(
                    x=grid_y, y=grid_z, z=G_u,
                    colorscale='Turbo',
                    colorbar=dict(title="Velocidad [m/s]"),
                    hovertemplate='Y: %{x:.1f} mm<br>Z: %{y:.1f} mm<br>u: %{z:.2f} m/s<extra></extra>'
                ))
                fig2.add_trace(go.Scatter(
                    x=y_coords, y=z_coords,
                    mode='markers',
                    marker=dict(size=3, color='white', opacity=0.4),
                    name='Puntos medidos'
                ))
                fig2.update_layout(
                    title="Mapeo 2D del Campo de Velocidades en la Estela",
                    xaxis_title="Y (Envergadura) [mm]",
                    yaxis_title="Z (Altura) [mm]",
                    height=600,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white")
                )
                fig2.update_xaxes(scaleanchor="y", scaleratio=1)
                st.plotly_chart(fig2, use_container_width=True)

            # Pestaña 3: Presión Total en 2D
            with t_plot3:
                G_pt = griddata((y_coords, z_coords), pt_vals, (Gy, Gz), method='cubic')
                fig3 = go.Figure()
                fig3.add_trace(go.Contour(
                    x=grid_y, y=grid_z, z=G_pt,
                    colorscale='Turbo',
                    colorbar=dict(title="Presión [Pa]"),
                    hovertemplate='Y: %{x:.1f} mm<br>Z: %{y:.1f} mm<br>Pt: %{z:.1f} Pa<extra></extra>'
                ))
                fig3.add_trace(go.Scatter(
                    x=y_coords, y=z_coords,
                    mode='markers',
                    marker=dict(size=3, color='white', opacity=0.4),
                    name='Puntos medidos'
                ))
                fig3.update_layout(
                    title="Distribución de Presión Total Pt en el Plano Y-Z",
                    xaxis_title="Y (Envergadura) [mm]",
                    yaxis_title="Z (Altura) [mm]",
                    height=600,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white")
                )
                fig3.update_xaxes(scaleanchor="y", scaleratio=1)
                st.plotly_chart(fig3, use_container_width=True)

            # --- CONSIDERACIONES CRÍTICAS DE LABORATORIO 2D ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("⚠️ Diagnóstico de Estela y Validaciones en el Plano 2D")
            
            col_cr1, col_cr2 = st.columns(2)
            
            with col_cr1:
                st.markdown("##### 📐 Límites del Plano y Recuperación de Flujo")
                # Calcular velocidad promedio de los puntos que forman los bordes del plano medido
                boundary_mask = (df_clean['Y'] == y_uniq[0]) | (df_clean['Y'] == y_uniq[-1]) | \
                                (df_clean['Z'] == z_uniq[0]) | (df_clean['Z'] == z_uniq[-1])
                boundary_u = u_local[boundary_mask]
                avg_boundary_u = np.mean(boundary_u) if len(boundary_u) > 0 else 0.0
                rec_percent = (avg_boundary_u / u_inf) * 100
                
                limite_sano = rec_percent >= 95.0
                
                if limite_sano:
                    st.success(f"✅ **Límites de Estela Óptimos**:\n- Promedio en Bordes: {rec_percent:.1f}% de U_∞\n\nEl plano de medición contiene perfectamente la estela sin fugas significativas de momentum.")
                else:
                    st.warning(f"⚠️ **Recuperación incompleta en los bordes del plano**:\n- Promedio en Bordes: {rec_percent:.1f}% de U_∞\n\n**Recomendación**: La velocidad en el perímetro del plano no regresó por completo a la corriente libre. La resistencia total ($C_d$) podría estar subestimada. Aconsejamos ensanchar el barrido en Y y Z para capturar todo el déficit.")
            
            with col_cr2:
                st.markdown("##### ⚖️ Centrado y Simetría Bidimensional de la Estela")
                
                # Centroide de momentum en Y y Z (Centro del déficit)
                drag_elem = df_clean['drag_elemental'].values
                sum_drag = drag_total
                
                if sum_drag > 0:
                    y_cent = float(np.sum(y_coords * drag_elem) / sum_drag)
                    z_cent = float(np.sum(z_coords * drag_elem) / sum_drag)
                else:
                    y_cent, z_cent = 0.0, 0.0
                    
                # Centro geométrico del plano de medición
                y_geom = (y_uniq[0] + y_uniq[-1]) / 2.0
                z_geom = (z_uniq[0] + z_uniq[-1]) / 2.0
                
                desfase_y = np.abs(y_cent - y_geom)
                desfase_z = np.abs(z_cent - z_geom)
                
                st.markdown(f"""
                - **Centroide de Estela (Def.)**: `Y = {y_cent:.1f} mm, Z = {z_cent:.1f} mm`
                - **Centro Geométrico del Mapeo**: `Y = {y_geom:.1f} mm, Z = {z_geom:.1f} mm`
                - **Desalineación Espacial**: `ΔY = {desfase_y:.1f} mm, ΔZ = {desfase_z:.1f} mm`
                """)
                
                if desfase_y < (y_uniq[-1] - y_uniq[0]) * 0.15 and desfase_z < (z_uniq[-1] - z_uniq[0]) * 0.15:
                    st.success("⚖️ **Estela Alineada**: El núcleo de la estela se encuentra centrado dentro de la malla de medición.")
                else:
                    st.warning("⚠️ **Estela Desalineada / Asimétrica**: El centro de la estela está significativamente desplazado hacia un lateral del plano de medición. Revisar alineación del modelo o centrar la grilla de la sonda.")
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            st.warning("⚠️ Se necesitan al menos 4 puntos de medición válidos en el plano para realizar la triangulación e interpolación Y-Z.")

    # --- PESTAÑA HISTORIAL Y POLAR DE ARRASTRE Cd vs AOA ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📊 Historial de Puntos y Curva Polar de Arrastre (Cd vs α)")
    
    if st.session_state.smn_betz_historial.empty:
        st.info("Aún no has guardado puntos en el historial de esta sesión. Guardá cálculos para graficar la polar de arrastre del modelo.")
    else:
        c_h1, c_h2 = st.columns([1.2, 1.8])
        
        with c_h1:
            st.markdown("##### Puntos Registrados")
            st.dataframe(st.session_state.smn_betz_historial[[
                'AOA [°]', 'Cd []', 'Arrastre [N]'
            ]], use_container_width=True, hide_index=True)
            
            if st.button("🗑️ Limpiar Historial de Puntos", use_container_width=True):
                st.session_state.smn_betz_historial = pd.DataFrame(columns=[
                    'Archivo', 'AOA [°]', 'Cd []', 'Arrastre [N]', 'V_inf [m/s]', 'rho_inf [kg/m³]'
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
                name='Arrastre del Modelo',
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
