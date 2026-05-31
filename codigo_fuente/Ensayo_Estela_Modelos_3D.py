import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import tempfile
import plotly.graph_objects as go
from codigo_fuente import Auth_Manager as auth

# Intentar importar pyvista, puede que no esté en requirements (agregar después si falla)
try:
    import pyvista as pv
except ImportError:
    pass

def show_modelos():
    st.markdown("""
    <div class="header-container">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            📦 GESTOR DE MODELOS 3D
        </h1>
        <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Importa o carga modelos de referencia para visualización 4D
        </h2>
    </div>
    """, unsafe_allow_html=True)

    if 'modelo_modo' not in st.session_state:
        st.session_state.modelo_modo = 'bd'
    if 'modelo_cg' not in st.session_state:
        st.session_state.modelo_cg = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    if 'modelo_nombre_bd' not in st.session_state:
        st.session_state.modelo_nombre_bd = None

    c_conf, c_preview = st.columns([1.2, 2])

    with c_conf:
        st.markdown("### 🔽 Fuente del Modelo")
        modo_opts = {"🗄 Cargar de Base de Datos": "bd", "📂 Importar Archivo (STL / CSV)": "importar"}
        modo_sel_label = st.radio("Seleccionar fuente:", list(modo_opts.keys()), index=0 if st.session_state.modelo_modo == 'bd' else 1, horizontal=True, key="modo_modelo_radio")
        st.session_state.modelo_modo = modo_opts[modo_sel_label]

        st.markdown("---")

        if st.session_state.modelo_modo == 'bd':
            try:
                saved_objs = auth.get_user_objects(st.session_state.username)
            except AttributeError:
                st.error("Error: Función get_user_objects no encontrada.")
                saved_objs = []

            if not saved_objs:
                st.info("No hay modelos guardados en la base de datos. Importa uno nuevo.")
            else:
                obj_labels = {f"📦 {name} ({o_type}) — {f_date}": (obj_id, name, o_type, d_json, f_date) for obj_id, name, o_type, d_json, f_date in saved_objs}
                sel_label = st.selectbox("Seleccionar modelo guardado:", list(obj_labels.keys()), key="sel_modelo_bd")
                
                if sel_label:
                    obj_id, name, o_type, d_json, f_date = obj_labels[sel_label]
                    
                    st.markdown(f"""
                    <div style="background:#111; border:1px solid #333; border-radius:8px; padding:12px; margin-bottom:12px;">
                        <p style="color:#888; margin:0; font-size:0.8rem;">Nombre</p>
                        <p style="color:white; font-weight:bold; margin:0 0 8px 0;">{name}</p>
                        <p style="color:#888; margin:0; font-size:0.8rem;">Tipo / Guardado</p>
                        <p style="color:#aaa; margin:0;">{o_type} | {f_date}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    try:
                        _data_preview = json.loads(d_json)
                        if 'cg' in _data_preview:
                            st.session_state.modelo_cg = _data_preview['cg']
                    except: pass

                    col_load, col_del = st.columns(2)
                    with col_load:
                        if st.button("📥 Seleccionar este modelo", use_container_width=True, type="primary", key="btn_select_bd"):
                            try:
                                data_loaded = json.loads(d_json)
                                data_loaded['x'] = np.array(data_loaded['x'])
                                data_loaded['y'] = np.array(data_loaded['y'])
                                data_loaded['z'] = np.array(data_loaded['z'])
                                if 'i' in data_loaded:
                                    data_loaded['i'] = np.array(data_loaded['i'])
                                    data_loaded['j'] = np.array(data_loaded['j'])
                                    data_loaded['k'] = np.array(data_loaded['k'])
                                if 'cg' in data_loaded:
                                    st.session_state.modelo_cg = data_loaded['cg']

                                st.session_state.objeto_referencia_4d = data_loaded
                                st.session_state.objeto_referencia_base = data_loaded.copy()
                                st.session_state.modelo_nombre_bd = name
                                st.success(f"✅ '{name}' cargado.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error al cargar: {e}")
                    with col_del:
                        if st.button("🗑️ Eliminar", use_container_width=True, key="btn_del_bd"):
                            try:
                                auth.delete_user_object(obj_id)
                                st.session_state.modelo_nombre_bd = None
                                st.rerun()
                            except:
                                st.error("Error al eliminar")

        else:
            st.markdown("##### 📂 Cargar Archivo STL, STEP o CSV")
            st.caption("El modelo se importa con los ejes del archivo. Los desplazamientos se configuran en 4D.")
            use_auto_center_imp = st.checkbox(" Auto-centrar objeto", value=True, key="auto_center_imp")
            file_obj = st.file_uploader("Cargar archivo (STL, STEP, STP o CSV):", type=['csv', 'stl', 'step', 'stp'], key="uploader_modelo_imp")

            if file_obj and st.button("📥 Procesar e importar", type="primary", use_container_width=True, key="btn_importar_modelo"):
                file_ext = file_obj.name.split('.')[-1].lower()
                x_points = y_points = z_points = faces_i = faces_j = faces_k = None
                obj_type = None

                try:
                    if file_ext == 'csv':
                        df_obj = pd.read_csv(file_obj, sep=None, engine='python')
                        cols_map = {c.lower(): c for c in df_obj.columns}
                        if 'x' in cols_map and 'y' in cols_map and 'z' in cols_map:
                            x_points = df_obj[cols_map['x']].values
                            y_points = df_obj[cols_map['y']].values
                            z_points = df_obj[cols_map['z']].values
                            obj_type = 'scatter'
                        else:
                            st.error("El CSV requiere columnas X, Y, Z")

                    elif file_ext == 'stl':
                        import struct
                        # Parsea STL usando un parser nativo rápido sin depender de pyvista (evitando dependencias de C++ VTK)
                        file_bytes = file_obj.read()
                        
                        try:
                            if len(file_bytes) < 84:
                                raise ValueError("El archivo es demasiado pequeño para ser un STL válido.")
                            
                            num_tri = struct.unpack('<I', file_bytes[80:84])[0]
                            expected = 84 + num_tri * 50
                            
                            if len(file_bytes) == expected:
                                # STL Binario
                                buffer = file_bytes[84:]
                                dtype = np.dtype([
                                    ('normal', '<f4', (3,)),
                                    ('v0', '<f4', (3,)),
                                    ('v1', '<f4', (3,)),
                                    ('v2', '<f4', (3,)),
                                    ('attr', '<u2')
                                ])
                                mesh_d = np.frombuffer(buffer, dtype=dtype, count=num_tri)
                                all_v = np.stack([mesh_d['v0'], mesh_d['v1'], mesh_d['v2']], axis=1)
                                flat_v = all_v.reshape(-1, 3)
                                vertices_u, inv = np.unique(flat_v, axis=0, return_inverse=True)
                                faces_u = inv.reshape(-1, 3)
                                v_arr, f_arr = vertices_u.astype(np.float64), faces_u.astype(np.int32)
                            else:
                                # STL ASCII
                                try:
                                    text = file_bytes.decode('utf-8', errors='ignore')
                                except Exception:
                                    raise ValueError("No se pudo decodificar el archivo STL como ASCII.")
                                v_list = []
                                lines_txt = text.split('\n')
                                for l in lines_txt:
                                    p_line = l.strip().split()
                                    if len(p_line) >= 4 and p_line[0].lower() == 'vertex':
                                        try:
                                            v_list.append([float(p_line[1]), float(p_line[2]), float(p_line[3])])
                                        except ValueError:
                                            pass
                                num_t = len(v_list) // 3
                                if num_t == 0:
                                    raise ValueError("No se encontraron vértices válidos en el archivo ASCII STL.")
                                triangles_arr = np.array(v_list[:num_t * 3], dtype=np.float64).reshape(-1, 3, 3)
                                flat_v = triangles_arr.reshape(-1, 3)
                                vertices_u, inv = np.unique(flat_v, axis=0, return_inverse=True)
                                faces_u = inv.reshape(-1, 3)
                                v_arr, f_arr = vertices_u, faces_u
                                
                            x_points = v_arr[:, 0]
                            y_points = v_arr[:, 1]
                            z_points = v_arr[:, 2]
                            faces_i = f_arr[:, 0]
                            faces_j = f_arr[:, 1]
                            faces_k = f_arr[:, 2]
                            obj_type = 'mesh'
                        except Exception as parse_e:
                            st.error(f"Error procesando el archivo STL: {parse_e}")
                            st.stop()

                    elif file_ext in ['step', 'stp']:
                        import trimesh
                        import cascadio
                        import io
                        
                        file_bytes = file_obj.read()
                        try:
                            # Guardar temporalmente los bytes del archivo step para cascadio
                            with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp:
                                tmp.write(file_bytes)
                                tmp_path = tmp.name
                            
                            # Crear una ruta temporal para el archivo GLB de salida
                            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp_glb:
                                tmp_glb_path = tmp_glb.name
                            
                            try:
                                # Convertir STEP a GLB escribiendo al archivo temporal
                                cascadio.step_to_glb(tmp_path, tmp_glb_path)
                                
                                # Cargar con trimesh desde el archivo
                                mesh = trimesh.load(tmp_glb_path, file_type="glb")
                                
                                # Si es un Scene (grupo de mallas), fusionamos
                                if isinstance(mesh, trimesh.Scene):
                                    if len(mesh.geometry) == 0:
                                        raise ValueError("El archivo STEP no contiene geometrías válidas.")
                                    mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
                                    
                                v_arr = np.array(mesh.vertices, dtype=np.float64)
                                f_arr = np.array(mesh.faces, dtype=np.int32)
                                
                                x_points = v_arr[:, 0]
                                y_points = v_arr[:, 1]
                                z_points = v_arr[:, 2]
                                faces_i = f_arr[:, 0]
                                faces_j = f_arr[:, 1]
                                faces_k = f_arr[:, 2]
                                obj_type = 'mesh'
                            finally:
                                # Limpiar archivos temporales
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
                                try:
                                    os.unlink(tmp_glb_path)
                                except:
                                    pass
                        except Exception as parse_e:
                            st.error(f"Error procesando el archivo STEP/STP: {parse_e}")
                            st.stop()

                    if x_points is not None:
                        if use_auto_center_imp:
                            cx = (np.min(x_points) + np.max(x_points)) / 2
                            cy = (np.min(y_points) + np.max(y_points)) / 2
                            cz = (np.min(z_points) + np.max(z_points)) / 2
                            x_points -= cx; y_points -= cy; z_points -= cz

                        obj_data = {'type': obj_type, 'x': x_points, 'y': y_points, 'z': z_points, 'name': file_obj.name}
                        if obj_type == 'mesh':
                            obj_data.update({'i': faces_i, 'j': faces_j, 'k': faces_k})

                        st.session_state.objeto_referencia_4d = obj_data
                        st.session_state.objeto_referencia_base = obj_data.copy()
                        st.session_state.modelo_nombre_bd = None
                        st.success(f"✅ Importado: {file_obj.name}")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error procesando archivo: {e}")

        if 'objeto_referencia_4d' in st.session_state:
            st.markdown("---")
            st.markdown("#### 🎯 Centro de Gravedad (CG) — Punto de Rotación")
            cg_cols = st.columns(3)
            st.session_state.modelo_cg['x'] = cg_cols[0].number_input("CG — X", value=float(st.session_state.modelo_cg.get('x', 0.0)), step=5.0, format="%.1f", key="cg_x")
            st.session_state.modelo_cg['y'] = cg_cols[1].number_input("CG — Y", value=float(st.session_state.modelo_cg.get('y', 0.0)), step=5.0, format="%.1f", key="cg_y")
            st.session_state.modelo_cg['z'] = cg_cols[2].number_input("CG — Z", value=float(st.session_state.modelo_cg.get('z', 0.0)), step=5.0, format="%.1f", key="cg_z")

            if st.button(" Auto-centrar CG", use_container_width=True, key="btn_autocg"):
                obj_actual = st.session_state.objeto_referencia_4d
                st.session_state.modelo_cg['x'] = 0.0
                st.session_state.modelo_cg['y'] = float((np.min(obj_actual['y']) + np.max(obj_actual['y'])) / 2)
                st.session_state.modelo_cg['z'] = float((np.min(obj_actual['z']) + np.max(obj_actual['z'])) / 2)
                st.rerun()

            st.markdown("---")
            nombre_actual = st.session_state.objeto_referencia_4d.get('name', 'MiModelo')
            nombre_modelo = st.text_input(" Nombre del modelo:", value=nombre_actual, key="nombre_modelo_final")

            c_btn1, c_btn2 = st.columns(2)
            lbl_save = f"🔄 ACTUALIZAR '{st.session_state.modelo_nombre_bd}'" if st.session_state.modelo_nombre_bd else "💾 GUARDAR modelo"

            with c_btn1:
                if st.button(lbl_save, use_container_width=True, type="primary", key="btn_guardar_modelo"):
                    obj_to_save = st.session_state.objeto_referencia_4d.copy()
                    obj_to_save['name'] = nombre_modelo
                    obj_to_save['cg'] = st.session_state.modelo_cg.copy()

                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.ndarray): return obj.tolist()
                            return json.JSONEncoder.default(self, obj)

                    try:
                        json_str = json.dumps(obj_to_save, cls=NumpyEncoder)
                        if auth.save_user_object(st.session_state.username, nombre_modelo, obj_to_save['type'], json_str):
                            st.success(f"✅ Guardado.")
                            st.session_state.modelo_nombre_bd = nombre_modelo
                    except Exception as e:
                        st.error(f"Error serializando: {e}")

            with c_btn2:
                if st.button("✅ USAR MODELO EN 4D", use_container_width=True, key="btn_usar_modelo"):
                    st.session_state.objeto_referencia_4d['name'] = nombre_modelo
                    st.session_state.objeto_referencia_4d['cg'] = st.session_state.modelo_cg.copy()
                    st.success(f"✅ Modelo listo para 4D.")

    with c_preview:
        st.markdown("### ⚙️ Vista Previa 3D")
        if 'objeto_referencia_4d' in st.session_state:
            obj = st.session_state.objeto_referencia_4d
            cg = st.session_state.modelo_cg
            fig_prev = go.Figure()

            if obj['type'] == 'mesh':
                fig_prev.add_trace(go.Mesh3d(x=obj['x'], y=obj['y'], z=obj['z'], i=obj['i'], j=obj['j'], k=obj['k'], color='#4a90d9', opacity=0.75, name=obj['name'], alphahull=0))
            else:
                fig_prev.add_trace(go.Scatter3d(x=obj['x'], y=obj['y'], z=obj['z'], mode='markers', marker=dict(size=2, color='#4a90d9'), name=obj['name']))

            fig_prev.add_trace(go.Scatter3d(x=[cg['x']], y=[cg['y']], z=[cg['z']], mode='markers+text', marker=dict(size=10, color='#ff4444', symbol='cross'), text=["CG"], textposition="top center"))

            fig_prev.update_layout(scene=dict(aspectmode='data'), height=550, margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_prev, use_container_width=True)
        else:
            st.info("Ningún modelo cargado.")
