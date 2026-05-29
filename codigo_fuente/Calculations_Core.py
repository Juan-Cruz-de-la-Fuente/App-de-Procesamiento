import pandas as pd
import numpy as np
import re
import os
import io
import json
import tempfile
import itertools
import random
from datetime import datetime
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

def calcular_variable_atmosferica(df, variable):
    res = df.get('Presion', pd.Series([0]*len(df)))
    if variable == 'Presión Total [Actual]':
        return res
    elif variable == 'rho_inf':
        return df.get('rho_inf', 1.225).fillna(1.225)
    elif variable == 'V_inf':
        return df.get('V_inf', 0.0).fillna(0.0)
    elif variable == 'P_inf':
        return df.get('P_inf', 101325.0).fillna(101325.0)
    elif variable == 'T_inf':
        return df.get('T_inf', 15.0).fillna(15.0)
    return res

def rotate_points(x, y, z, angle_x, angle_y, angle_z):
    """Rotates 3D points around X, Y, Z axes (angles in degrees)."""
    rad_x = np.radians(angle_x)
    rad_y = np.radians(angle_y)
    rad_z = np.radians(angle_z)
    
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(rad_x), -np.sin(rad_x)],
        [0, np.sin(rad_x), np.cos(rad_x)]
    ])
    
    ry = np.array([
        [np.cos(rad_y), 0, np.sin(rad_y)],
        [0, 1, 0],
        [-np.sin(rad_y), 0, np.cos(rad_y)]
    ])
    
    rz = np.array([
        [np.cos(rad_z), -np.sin(rad_z), 0],
        [np.sin(rad_z), np.cos(rad_z), 0],
        [0, 0, 1]
    ])
    
    R = rz @ (ry @ rx)
    points = np.vstack([x, y, z])
    rotated_points = R @ points
    return rotated_points[0,:], rotated_points[1,:], rotated_points[2,:]

def extraer_tiempo_y_coordenadas_YZ(nombre_archivo):
    tiempo = None
    y_traverser = None 
    z_base = None

    nombre = os.path.basename(str(nombre_archivo))
    nombre_sin_ext = re.sub(r'\.\w+$', '', nombre)

    partes = nombre_sin_ext.split('_')
    if partes and partes[-1].isdigit():
        try:
            tiempo = int(partes[-1])
        except:
            tiempo = None

    if tiempo is None:
        tiempo_match = re.search(r"[Tt](\d+)\s*[sS]?$", nombre_sin_ext)
        if tiempo_match:
            tiempo = int(tiempo_match.group(1))

    x_match = re.search(r"[Xx][_\-=]?(-?\d+)", nombre_sin_ext)
    if x_match:
        try:
            y_traverser = int(x_match.group(1))
        except:
            pass
    
    if y_traverser is None:
        m = re.search(r"[Xx]\s*(\d+)", nombre_sin_ext)
        if m:
            y_traverser = int(m.group(1))

    y_match = re.search(r"[Yy][_\-=]?(-?\d+)", nombre_sin_ext)
    if y_match:
        try:
            z_base = int(y_match.group(1))
        except:
            pass
            
    if z_base is None:
        m = re.search(r"[Yy]\s*(\d+)", nombre_sin_ext)
        if m:
            z_base = int(m.group(1))

    return tiempo, y_traverser, z_base

def normalizar_nombre_sensor(sensor_text):
    if pd.isna(sensor_text):
        return None
    s = str(sensor_text).strip()
    if not s:
        return None

    m = re.search(r'(?i)presion[-_ ]*sensor[_\-]?(\d+)[_\-](\d+)', s)
    if m:
        offset = int(m.group(1))
        idx = int(m.group(2))
        sensor_global = offset * 12 + idx
        return f"Presion-Sensor {sensor_global}"

    m2 = re.search(r'(?i)presion[-_ ]*sensor[_\-\s]*(\d+)', s)
    if m2:
        sensor_global = int(m2.group(1))
        return f"Presion-Sensor {sensor_global}"

    nums = re.findall(r'(\d+)', s)
    if nums:
        if len(nums) >= 2:
            offset = int(nums[-2])
            idx = int(nums[-1])
            if 0 <= offset <= 9 and 1 <= idx <= 12:
                sensor_global = offset * 12 + idx
                return f"Presion-Sensor {sensor_global}"
        sensor_global = int(nums[-1])
        return f"Presion-Sensor {sensor_global}"

    return s

def obtener_numero_sensor_desde_columna(col_name):
    if pd.isna(col_name):
        return None
    s = str(col_name)
    m = re.search(r'(?i)presion[-_ ]*sensor[_\-\s]*(\d+)', s)
    if m:
        return int(m.group(1))
    nums = re.findall(r'(\d+)', s)
    if nums:
        return int(nums[-1])
    return None

def calcular_altura_absoluta_z(sensor_num, z_base_ref, posicion_inicial, distancia_entre_tomas, n_sensores, orden="asc"):
    if sensor_num is None:
        return None
    toma_index = int(sensor_num)
    if orden == "asc":
        z_total = z_base_ref + (toma_index - 1) * distancia_entre_tomas
    else:
        z_total = z_base_ref + (n_sensores - toma_index) * distancia_entre_tomas
    return z_total

def extraer_nombre_base_archivo(nombre_archivo):
    """Extraer nombre base del archivo (sin extensión y sin 'incertidumbre_')"""
    nombre_base = os.path.basename(str(nombre_archivo)).replace('.csv', '').replace('incertidumbre_', '').replace('_', ' ')
    return ' '.join(word.capitalize() for word in nombre_base.split())

def procesar_promedios(archivo_csv, orden="asc", archivo_infinito=None):
    """Procesar archivo de incertidumbre y detectar automáticamente la cantidad de sensores."""
    try:
        df_raw = pd.read_csv(archivo_csv, sep=";", header=None, dtype=str)

        index_final = df_raw[df_raw.apply(lambda row: row.astype(str).str.contains("importante", case=False).any(), axis=1)].index
        if not index_final.empty:
            df_raw = df_raw.iloc[:index_final[0]]

        resultados = []
        for i in range(0, df_raw.shape[0], 10):
            bloque = df_raw.iloc[i:i+10]
            if bloque.empty or len(bloque) < 3:
                continue

            archivo = bloque.iloc[0, 0]
            raw_sensores = bloque.iloc[0, 1:].tolist()
            muestras = bloque.iloc[1, 1] if 1 < bloque.shape[1] else None

            sensores_lista = []
            for entry in raw_sensores:
                if pd.isna(entry): continue
                s = str(entry).strip()
                if ';' in s:
                    sensores_lista.extend([p.strip() for p in s.split(';') if p.strip()])
                else:
                    sensores_lista.append(s)

            valores_lista = []
            for entry in bloque.iloc[2, 1:].tolist():
                if pd.isna(entry): continue
                s = str(entry).strip()
                if ';' in s:
                    valores_lista.extend([p.strip() for p in s.split(';')])
                else:
                    valores_lista.append(s)

            n = max(len(sensores_lista), len(valores_lista))
            sensores_lista = (sensores_lista + [None] * n)[:n]
            valores_lista = (valores_lista + [None] * n)[:n]

            fila = {"Archivo": archivo, "Numero de muestras": muestras}
            for sensor_raw, valor_raw in zip(sensores_lista, valores_lista):
                nombre_sensor_norm = normalizar_nombre_sensor(sensor_raw)
                if nombre_sensor_norm is None: continue
                valor = valor_raw
                if isinstance(valor, str):
                    valor = valor.replace(',', '.').strip()
                try:
                    valor_num = float(valor) if (valor is not None and str(valor) != '') else np.nan
                except:
                    valor_num = np.nan
                fila[nombre_sensor_norm] = valor_num
            resultados.append(fila)

        df_resultado = pd.DataFrame(resultados)
        if "Archivo" in df_resultado.columns:
            coordenadas_tiempo = df_resultado["Archivo"].apply(extraer_tiempo_y_coordenadas_YZ)
            df_resultado["Tiempo_s"] = [coord[0] for coord in coordenadas_tiempo]
            df_resultado["Pos_Y_Traverser"] = [coord[1] for coord in coordenadas_tiempo]
            df_resultado["Pos_Z_Base"] = [coord[2] for coord in coordenadas_tiempo]

            def _extract_ts(n):
                m = re.search(r'(\d{10,14})', str(n))
                return m.group(1) if m else None
            df_resultado["Timestamp"] = df_resultado["Archivo"].apply(_extract_ts)

            inf_file = archivo_infinito if archivo_infinito else "Valores en el infinito.txt"
            df_resultado["rho_inf"] = 1.225
            df_resultado["V_inf"] = 0.0
            df_resultado["P_inf"] = 101325.0

            if os.path.exists(inf_file) or not isinstance(inf_file, str):
                try:
                    df_inf = pd.read_csv(inf_file, sep=";", engine="python", skip_blank_lines=True)
                    df_inf.columns = [str(c).strip() for c in df_inf.columns]
                    if len(df_inf.columns) > 2:
                        first_col = df_inf.columns[0]
                        df_inf["ts_clean"] = df_inf[first_col].astype(str).str.split(',').str[0].str.strip()
                        df_inf["dt_val"] = pd.to_datetime(df_inf["ts_clean"], format='%d%m%y%H%M%S', errors='coerce')
                        mask_failed = df_inf["dt_val"].isna()
                        if mask_failed.any():
                            df_inf.loc[mask_failed, "dt_val"] = pd.to_datetime(df_inf.loc[mask_failed, "ts_clean"], format='%y%m%d%H%M%S', errors='coerce')
                        df_inf = df_inf.dropna(subset=["dt_val"])

                        def get_inf_values(ts_str):
                            try:
                                if ts_str is None or str(ts_str) == 'None': return 1.225, 0.0, 101325.0, 15.0
                                ts_clean = str(ts_str).split(',')[0].strip()
                                target_dt = pd.to_datetime(ts_clean, format='%d%m%y%H%M%S', errors='coerce')
                                if pd.isna(target_dt): target_dt = pd.to_datetime(ts_clean, format='%y%m%d%H%M%S', errors='coerce')
                                if pd.isna(target_dt): return 1.225, 0.0, 101325.0, 15.0
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
                                return rho, v_inf, P_pa, T
                            except:
                                return 1.225, 0.0, 101325.0, 15.0

                        recs = df_resultado["Timestamp"].apply(get_inf_values)
                        df_resultado["rho_inf"] = [r[0] for r in recs]
                        df_resultado["V_inf"] = [r[1] for r in recs]
                        df_resultado["P_inf"] = [r[2] for r in recs]
                        df_resultado["T_inf"] = [r[3] for r in recs]
                except: pass
        
        sensores_cols = [c for c in df_resultado.columns if re.search(r'Presion[-_ ]*Sensor', str(c), re.IGNORECASE)]
        df_resultado.attrs["n_sensores"] = max([obtener_numero_sensor_desde_columna(c) for c in sensores_cols if obtener_numero_sensor_desde_columna(c) is not None], default=0)
        return df_resultado
    except Exception as e:
        return None

def crear_archivos_individuales_por_tiempo_y_posicion(df_resultado, nombre_archivo_fuente):
    sub_archivos = {}
    nombre_base = extraer_nombre_base_archivo(nombre_archivo_fuente)
    nombre_original = os.path.splitext(os.path.basename(nombre_archivo_fuente))[0]
    y_vals = df_resultado["Pos_Y_Traverser"].dropna().unique()
    for y_valor in sorted(y_vals):
        df_y = df_resultado[df_resultado["Pos_Y_Traverser"] == y_valor]
        t_vals = df_y["Tiempo_s"].dropna().unique()
        for tiempo in sorted(t_vals):
            df_yt = df_y[df_y["Tiempo_s"] == tiempo]
            clave_sub_archivo = f"{nombre_original}_X{int(y_valor) if pd.notna(y_valor) else 0}_T{tiempo}s"
            num_z = len(df_yt['Pos_Z_Base'].unique()) if 'Pos_Z_Base' in df_yt.columns else 1
            sub_archivos[clave_sub_archivo] = {
                'archivo_fuente': nombre_base,
                'archivo_origen': nombre_original,
                'tiempo': tiempo,
                'pos_y_traverser': y_valor,
                'datos': df_yt,
                'nombre_archivo': f"{clave_sub_archivo}.csv",
                'num_posiciones_z': num_z
            }
    return sub_archivos

def crear_sub_archivos_3d_por_tiempo_y_posicion(df_datos, nombre_archivo):
    sub_archivos = {}
    tiempos_unicos = df_datos["Tiempo_s"].dropna().unique()
    for tiempo in tiempos_unicos:
        df_tiempo = df_datos[df_datos["Tiempo_s"] == tiempo].copy()
        clave_sub_archivo = f"{nombre_archivo}_T{tiempo}s"
        sub_archivos[clave_sub_archivo] = {
            'archivo_fuente': nombre_archivo,
            'tiempo': tiempo,
            'datos': df_tiempo,
            'nombre_archivo': f"{nombre_archivo}_T{tiempo}s.csv"
        }
    return sub_archivos

def calcular_area_bajo_curva(z_datos, presion_datos):
    if len(z_datos) < 2 or len(presion_datos) < 2: return 0
    area = 0
    for i in range(len(z_datos) - 1):
        h = z_datos[i + 1] - z_datos[i]
        area += h * (presion_datos[i] + presion_datos[i + 1]) / 2
    return abs(area)

def calcular_posiciones_sensores(distancia_toma_12, distancia_entre_tomas, n_sensores, orden="asc"):
    posiciones = {}
    for sensor_num in range(1, n_sensores + 1):
        y_position = (sensor_num - 1) * distancia_entre_tomas if orden == "asc" else (n_sensores - sensor_num) * distancia_entre_tomas
        posiciones[f"Presion-Sensor {sensor_num}"] = {'x': 0, 'y': y_position, 'sensor_fisico': sensor_num}
    return posiciones

def extraer_datos_para_grafico(sub_archivo, configuracion, variable='Presion Total'):
    datos_tiempo = sub_archivo['datos']
    distancia_entre_tomas = configuracion.get('distancia_entre_tomas', 10.0)
    posicion_inicial = configuracion.get('distancia_toma_12', 0)
    orden = configuracion.get('orden', 'asc')
    z_datos, presion_datos = [], []
    sensor_cols = [c for c in datos_tiempo.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]
    n_sensores = max([obtener_numero_sensor_desde_columna(c) for c in sensor_cols], default=0)
    for _, fila in datos_tiempo.iterrows():
        z_base_ref = fila.get('Pos_Z_Base', 0)
        for col in sensor_cols:
            sensor_num = obtener_numero_sensor_desde_columna(col)
            if sensor_num is None: continue
            z_total = calcular_altura_absoluta_z(sensor_num, z_base_ref, posicion_inicial, distancia_entre_tomas, n_sensores, orden)
            presion = fila.get(col, None)
            if pd.isna(presion): continue
            try:
                presion_val = float(str(presion).replace(',', '.'))
                valor_final = presion_val
                if variable == 'P_t / Rho_inf':
                    rho = float(fila.get('rho_inf', 1.225))
                    valor_final = presion_val / rho if rho != 0 else 0
                elif variable == 'Velocidad Infinito':
                    valor_final = float(fila.get('V_inf', 0.0))
                elif variable == 'Presion Infinito':
                    valor_final = float(fila.get('P_inf', 101325.0))
                z_datos.append(z_total)
                presion_datos.append(valor_final)
            except: continue
    if z_datos and presion_datos:
        datos_ordenados = sorted(zip(z_datos, presion_datos))
        z_ordenado, presion_ordenada = zip(*datos_ordenados)
        return list(z_ordenado), list(presion_ordenada)
    return [], []

def unir_archivos_incertidumbre(archivos_lista, nombre_salida):
    try:
        contenido_unido = []
        puntos_sobrepuestos = []
        coordenadas_vistas = set()
        for archivo in archivos_lista:
            df_raw = pd.read_csv(archivo, sep=";", header=None, dtype=str)
            index_final = df_raw[df_raw.apply(lambda row: row.astype(str).str.contains("importante", case=False).any(), axis=1)].index
            if not index_final.empty: df_raw = df_raw.iloc[:index_final[0]]
            for i in range(0, df_raw.shape[0], 10):
                bloque = df_raw.iloc[i:i+10]
                if bloque.empty or len(bloque) < 3: continue
                nombre_archivo_bloque = bloque.iloc[0, 0]
                tiempo, y_trav, z_base = extraer_tiempo_y_coordenadas_YZ(nombre_archivo_bloque)
                coordenada_key = (y_trav, z_base, tiempo)
                if coordenada_key in coordenadas_vistas: puntos_sobrepuestos.append(coordenada_key)
                else: coordenadas_vistas.add(coordenada_key)
                contenido_unido.extend(bloque.values.tolist())
        df_unido = pd.DataFrame(contenido_unido)
        return df_unido.to_csv(sep=';', index=False, header=False), puntos_sobrepuestos
    except: return None, []

def extraer_matriz_presiones_completa(archivo_incertidumbre, configuracion, archivo_infinito=None):
    try:
        datos = procesar_promedios(archivo_incertidumbre, configuracion.get("orden", "asc"), archivo_infinito)
        if datos is None: return pd.DataFrame(columns=["Y", "Z", "Presion"])
        registros = []
        sensor_cols = [c for c in datos.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]
        for _, fila in datos.iterrows():
            y_trav, z_base = fila.get("Pos_Y_Traverser"), fila.get("Pos_Z_Base")
            if pd.isna(y_trav) or pd.isna(z_base): continue
            for col in sensor_cols:
                num = obtener_numero_sensor_desde_columna(col)
                if num is None: continue
                z_real = calcular_altura_absoluta_z(num, z_base, configuracion.get("distancia_toma_12", -120), configuracion.get("distancia_entre_tomas", 10.0), len(sensor_cols), configuracion.get("orden", "asc"))
                presion = fila.get(col, None)
                if presion is None or pd.isna(presion): continue
                try: registros.append((float(y_trav), float(z_real), float(str(presion).replace(",", "."))))
                except: continue
        df = pd.DataFrame(registros, columns=["Y", "Z", "Presion"]).sort_values(by=["Y", "Z"]).reset_index(drop=True)
        return df
    except: return pd.DataFrame(columns=["Y", "Z", "Presion"])

def crear_vtk_superficie_3d_delaunay(df_matriz, nombre_base, posicion_x=0.0):
    try:
        puntos_y, puntos_z, presiones = df_matriz["Y"].values, df_matriz["Z"].values, df_matriz["Presion"].values
        mask = ~np.isnan(puntos_y) & ~np.isnan(puntos_z) & ~np.isnan(presiones)
        puntos_y, puntos_z, presiones = puntos_y[mask], puntos_z[mask], presiones[mask]
        if len(puntos_y) < 3: return None
        tri = Delaunay(np.column_stack([puntos_y, puntos_z]))
        n_points, n_triangles = len(puntos_y), len(tri.simplices)
        nombre_archivo_vtk = f"{nombre_base}_superficie_3D.vtk"
        vtk_content = f"# vtk DataFile Version 3.0\nSuperficie 3D\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS {n_points} float\n"
        for i in range(n_points): vtk_content += f"{posicion_x + presiones[i]:.6f} {puntos_y[i]:.6f} {puntos_z[i]:.6f}\n"
        vtk_content += f"\nCELLS {n_triangles} {4 * n_triangles}\n"
        for simplex in tri.simplices: vtk_content += f"3 {simplex[0]} {simplex[1]} {simplex[2]}\n"
        vtk_content += f"\nCELL_TYPES {n_triangles}\n" + "5\n" * n_triangles
        vtk_content += f"\nPOINT_DATA {n_points}\nSCALARS Presion float 1\nLOOKUP_TABLE default\n"
        for p in presiones: vtk_content += f"{p:.6f}\n"
        with open(nombre_archivo_vtk, "w") as f: f.write(vtk_content)
        return nombre_archivo_vtk
    except: return None

def crear_vtk_plano_presion_2d(df_matriz, nombre_base, posicion_x=0.0):
    try:
        y_vals, z_vals = sorted(df_matriz['Y'].unique()), sorted(df_matriz['Z'].unique())
        ny, nz = len(y_vals), len(z_vals)
        presion_map = {(float(row['Y']), float(row['Z'])): float(row['Presion']) for _, row in df_matriz.iterrows()}
        lines = ["# vtk DataFile Version 3.0", "Plano 2D", "ASCII", "DATASET STRUCTURED_GRID", f"DIMENSIONS 1 {ny} {nz}", f"POINTS {ny * nz} float"]
        presiones_ordenadas = []
        for z in z_vals:
            for y in y_vals:
                lines.append(f"{posicion_x:.6f} {y:.6f} {z:.6f}")
                presiones_ordenadas.append(presion_map.get((float(y), float(z)), 0.0))
        lines.append(f"\nPOINT_DATA {ny * nz}\nSCALARS Presion float 1\nLOOKUP_TABLE default")
        for p in presiones_ordenadas: lines.append(f"{p:.6f}")
        vtk_str = "\n".join(lines)
        nombre_archivo = f"{nombre_base}_plano2D.vtk"
        with open(nombre_archivo, "w") as f: f.write(vtk_str)
        return nombre_archivo, vtk_str.encode('ascii')
    except: return None
