# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageOps
import plotly.graph_objects as go
import io
import struct
import json
from codigo_fuente import Auth_Manager as auth
from codigo_fuente import Drive_Connection as drive_api

# ═══════════════════════════════════════════════════════════════
#  FUNCIONES DE APOYO Y MATEMÁTICAS (DE UTILS.PY Y APP-FINAL.PY)
# ═══════════════════════════════════════════════════════════════

def compute_face_normals(vertices, faces):
    """Calcula los vectores normales unitarios de cada cara del modelo."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    face_normals = normals / norms
    return face_normals

def estimate_camera_matrix(width, height, focal_factor=1.2):
    """
    Estima la matriz intrínseca K de la cámara asumiendo un lente estándar
    y que el centro óptico se encuentra en el centro de la imagen.
    """
    f = focal_factor * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    return K

def calibrate_camera(object_points, image_points, K, dist_coeffs=None, use_ransac=False):
    """
    Ejecuta SolvePnP o SolvePnPRansac de manera extremadamente robusta utilizando una cascada
    de flags (SQPNP -> IPPE -> ITERATIVE -> EPNP) para evitar errores de DLT con pocos puntos.
    Retorna (rvec, tvec, C, pitch, roll, yaw, success)
    """
    if len(object_points) < 4:
        return None, None, None, 0, 0, 0, False
        
    if dist_coeffs is None:
        dist_coeffs = np.zeros(4, dtype=np.float64)
        
    # Cascada de algoritmos para máxima robustez
    flags_to_try = [
        cv2.SOLVEPNP_SQPNP,
        cv2.SOLVEPNP_IPPE,
        cv2.SOLVEPNP_ITERATIVE,
        cv2.SOLVEPNP_EPNP
    ]
    
    success = False
    rvec, tvec = None, None
    
    obj_pts_f64 = object_points.astype(np.float64)
    img_pts_f64 = image_points.astype(np.float64)
    
    for flag in flags_to_try:
        try:
            if use_ransac:
                ret, r, t, inliers = cv2.solvePnPRansac(
                    obj_pts_f64,
                    img_pts_f64,
                    K,
                    dist_coeffs,
                    flags=flag,
                    reprojectionError=8.0,
                    iterationsCount=150
                )
                if ret:
                    success = True
                    rvec = r
                    tvec = t
                    break
            else:
                ret, r, t = cv2.solvePnP(
                    obj_pts_f64,
                    img_pts_f64,
                    K,
                    dist_coeffs,
                    flags=flag
                )
                if ret:
                    success = True
                    rvec = r
                    tvec = t
                    break
        except cv2.error:
            continue
            
    if not success or rvec is None or tvec is None:
        return None, None, None, 0, 0, 0, False
        
    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ tvec
    C = C.flatten()
    
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
        roll = np.arctan2(R[2,1], R[2,2])
    else:
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(-R[1,2], R[1,1])
        roll = 0.0
        
    return rvec, tvec, C, np.degrees(pitch), np.degrees(roll), np.degrees(yaw), True

def snap_to_closest_vertex(P, vertices):
    """
    Encuentra el vértice más cercano de la malla STL al punto P (X, Y, Z).
    Retorna el punto (X_snap, Y_snap, Z_snap) del vértice.
    """
    if vertices is None or len(vertices) == 0:
        return P
    P = np.array(P, dtype=np.float64)
    dists = np.linalg.norm(vertices - P, axis=1)
    closest_idx = np.argmin(dists)
    return vertices[closest_idx]

def get_camera_wireframe(C, rvec, tvec, scale=1.0):
    R, _ = cv2.Rodrigues(rvec)
    aspect = 4.0 / 3.0
    w = scale * 0.4 * aspect
    h = scale * 0.4
    d = scale * 0.8
    
    corners_cam = np.array([
        [0, 0, 0],
        [-w, -h, d],
        [w, -h, d],
        [w, h, d],
        [-w, h, d]
    ])
    
    corners_world = (R.T @ corners_cam.T).T + C.reshape(1, 3)
    
    path = [
        corners_world[0], corners_world[1], corners_world[2], corners_world[0],
        corners_world[3], corners_world[4], corners_world[0],
        [None, None, None],
        corners_world[1], corners_world[2], corners_world[3], corners_world[4], corners_world[1]
    ]
    
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    zs = [p[2] for p in path]
    
    return xs, ys, zs

def backward_projection(vertices, faces, images, calibrations):
    """Mapeo de texturas inverso (Backward Projection) optimizado y vectorizado."""
    num_faces = len(faces)
    face_colors = np.full((num_faces, 3), 220, dtype=np.uint8)
    
    if not calibrations or not images:
        return [f"rgb({c[0]},{c[1]},{c[2]})" for c in face_colors]
        
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    centroids = (v0 + v1 + v2) / 3.0
    face_normals = compute_face_normals(vertices, faces)
    
    camera_names = list(calibrations.keys())
    num_faces = len(faces)
    
    best_dot = np.full(num_faces, -np.inf)
    best_cam_idx = np.full(num_faces, -1, dtype=np.int32)
    
    for idx, cam_name in enumerate(camera_names):
        cal = calibrations[cam_name]
        rvec = cal['rvec']
        tvec = cal['tvec']
        R, _ = cv2.Rodrigues(rvec)
        C = -R.T @ tvec
        
        vec_to_cam = C.flatten() - centroids
        vec_norms = np.linalg.norm(vec_to_cam, axis=1, keepdims=True)
        vec_norms = np.where(vec_norms == 0, 1.0, vec_norms)
        dir_to_cam = vec_to_cam / vec_norms
        
        dots = np.sum(face_normals * dir_to_cam, axis=1)
        
        is_better = dots > best_dot
        best_dot = np.where(is_better, dots, best_dot)
        best_cam_idx = np.where(is_better, idx, best_cam_idx)
        
    for idx, cam_name in enumerate(camera_names):
        faces_mask = (best_cam_idx == idx) & (best_dot > 0.05)
        if not np.any(faces_mask):
            continue
            
        cal = calibrations[cam_name]
        rvec = cal['rvec']
        tvec = cal['tvec']
        K = cal['K']
        img = images[cam_name]
        H, W, _ = img.shape
        
        R, _ = cv2.Rodrigues(rvec)
        active_centroids = centroids[faces_mask]
        
        pts_cam = R @ active_centroids.T + tvec.reshape(3, 1)
        xc = pts_cam[0, :]
        yc = pts_cam[1, :]
        zc = pts_cam[2, :]
        
        valid_z = zc > 0.001
        if not np.any(valid_z):
            continue
            
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        u = fx * (xc / zc) + cx
        v = fy * (yc / zc) + cy
        
        valid_uv = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        valid_mask = valid_z & valid_uv
        
        if not np.any(valid_mask):
            continue
            
        cols = u[valid_mask].astype(np.int32)
        rows = v[valid_mask].astype(np.int32)
        sampled_colors = img[rows, cols]
        
        active_indices = np.where(faces_mask)[0]
        valid_face_indices = active_indices[valid_mask]
        face_colors[valid_face_indices] = sampled_colors
        
    return [f"rgb({c[0]},{c[1]},{c[2]})" for c in face_colors]

def load_uploaded_image(file_bytes):
    pil_img = Image.open(io.BytesIO(file_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    
    np_arr = np.array(pil_img)
    if len(np_arr.shape) == 2:
        np_arr = cv2.cvtColor(np_arr, cv2.COLOR_GRAY2RGB)
    elif np_arr.shape[2] == 4:
        np_arr = cv2.cvtColor(np_arr, cv2.COLOR_RGBA2RGB)
        
    return pil_img, np_arr

def get_scaled_K(K, target_shape, source_shape):
    if K is None:
        return None
    if source_shape is None or tuple(target_shape) == tuple(source_shape):
        return K.copy()
    
    w_scale = target_shape[0] / source_shape[0]
    h_scale = target_shape[1] / source_shape[1]
    
    K_scaled = K.copy()
    K_scaled[0, 0] *= w_scale
    K_scaled[1, 1] *= h_scale
    K_scaled[0, 2] *= w_scale
    K_scaled[1, 2] *= h_scale
    return K_scaled

def draw_points_on_image(pil_img, points, scale_factor=1.0):
    draw_img = pil_img.copy()
    draw = ImageDraw.Draw(draw_img)
    w, h = draw_img.size
    r = max(int(max(w, h) * 0.015), 6)
    for idx, pt in enumerate(points):
        u = int(pt["u"] * scale_factor)
        v = int(pt["v"] * scale_factor)
        draw.ellipse([u - r, v - r, u + r, v + r], outline="#ffffff", width=3)
        draw.ellipse([u - r + 2, v - r + 2, u + r - 2, v + r - 2], outline="#ef4444", width=2)
        draw.ellipse([u - r//2, v - r//2, u + r//2, v + r//2], fill="#f59e0b")
        label = str(idx + 1)
        tx, ty = u + r + 5, v - r - 5
        tw = len(label) * 11 + 6
        draw.rectangle([tx - 2, ty - 2, tx + tw, ty + 16], fill="#0f172a", outline="#ffffff", width=1)
        draw.text((tx + 2, ty), label, fill="#f8fafc")
    return draw_img

def K_matrix_display(K):
    if K is None:
        return ""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return f"$$K = \\begin{{bmatrix}} {fx:.1f} & 0 & {cx:.1f} \\\\ 0 & {fy:.1f} & {cy:.1f} \\\\ 0 & 0 & 1 \\end{{bmatrix}}$$"

# ═══════════════════════════════════════════════════════════════
#  GÓOGLE DRIVE PROJECT MANAGEMENT HELPERS
# ═══════════════════════════════════════════════════════════════

def get_datafusion_projects(username):
    datafusion_id = auth.get_folder_datafusion(username)
    if not datafusion_id:
        return []
    service = drive_api.get_service()
    if not service:
        return []
    query = f"'{datafusion_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name, createdTime)').execute()
    return results.get('files', [])


# ═══════════════════════════════════════════════════════════════
#  COMPONENTE PRINCIPAL: SHOW_DATAFUSION
# ═══════════════════════════════════════════════════════════════

def show_data_fusion():
    st.markdown("""
        <div class="header-container">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🎨 PLATAFORMA DE DATA FUSION
            </h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Mapeo Interactivo de Texturas 2D sobre Modelos 3D
            </h2>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 2px solid #333; margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)

    # --- SESSION STATE INITIALIZATION ---
    if "df_images" not in st.session_state: st.session_state.df_images = []
    if "df_current_image_idx" not in st.session_state: st.session_state.df_current_image_idx = 0
    if "df_points_data" not in st.session_state: st.session_state.df_points_data = {}
    if "df_last_clicks" not in st.session_state: st.session_state.df_last_clicks = {}
    if "df_calibrations" not in st.session_state: st.session_state.df_calibrations = {}
    if "df_camera_profiles" not in st.session_state: st.session_state.df_camera_profiles = {"Cámara A (Principal)": 1.0, "Cámara B (Secundaria)": 1.0}
    if "df_image_camera_assignments" not in st.session_state: st.session_state.df_image_camera_assignments = {}
    if "df_object_rotation_angles" not in st.session_state: st.session_state.df_object_rotation_angles = {}
    if "df_K_matrix" not in st.session_state: st.session_state.df_K_matrix = None
    if "df_dist_coeffs" not in st.session_state: st.session_state.df_dist_coeffs = None
    if "df_calibration_rms" not in st.session_state: st.session_state.df_calibration_rms = None
    if "df_K_source" not in st.session_state: st.session_state.df_K_source = "estimada"
    if "df_calibration_shape" not in st.session_state: st.session_state.df_calibration_shape = None
    if "df_offset_x" not in st.session_state: st.session_state.df_offset_x = 0.0
    if "df_offset_y" not in st.session_state: st.session_state.df_offset_y = 0.0
    if "df_offset_z" not in st.session_state: st.session_state.df_offset_z = 0.0
    if "df_use_ransac" not in st.session_state: st.session_state.df_use_ransac = True
    if "df_face_colors" not in st.session_state: st.session_state.df_face_colors = None
    if "df_project_name" not in st.session_state: st.session_state.df_project_name = ""
    if "df_project_id" not in st.session_state: st.session_state.df_project_id = None
    if "df_stl_units" not in st.session_state: st.session_state.df_stl_units = "mm"
    if "df_pro_mode" not in st.session_state:
        st.session_state.df_pro_mode = {
            "iso": 400,
            "shutter": "1/30",
            "focus": "Manual — Fixed",
            "wb": "Fluorescente neutro (4000K)",
            "locked": False,
        }

    username = st.session_state.username

    # ═══════════════════════════════════════════════════════════════
    #  PANEL DE PROYECTOS (EXPANDER SUPERIOR)
    # ═══════════════════════════════════════════════════════════════
    with st.expander("📁 GESTIÓN DE PROYECTOS (Google Drive Sync)", expanded=True):
        col_list, col_create = st.columns(2)
        
        with col_list:
            st.markdown("#### 📂 Cargar Proyecto")
            projects = []
            try:
                projects = get_datafusion_projects(username)
            except Exception as e:
                st.error(f"Error listando proyectos: {e}")
                
            if not projects:
                st.caption("No se encontraron proyectos guardados en Drive.")
            else:
                proj_dict = {p['name']: p for p in projects}
                sel_proj = st.selectbox("Seleccionar Proyecto:", ["-- Seleccionar Proyecto --"] + list(proj_dict.keys()), key="df_load_project_ui")
                if sel_proj != "-- Seleccionar --" and sel_proj != "-- Seleccionar Proyecto --":
                    if st.session_state.df_project_name != sel_proj:
                        proj_info = proj_dict[sel_proj]
                        with st.spinner("Descargando proyecto completo..."):
                            # Descargar archivos
                            files = drive_api.list_files(proj_info['id'])
                            config_file = next((f for f in files if f['name'] == 'project_config.json'), None)
                            
                            # Cargar configuraciones
                            if config_file:
                                cfg_bytes = drive_api.download_file(config_file['id'])
                                if cfg_bytes:
                                    try:
                                        cfg = json.loads(cfg_bytes.decode('utf-8'))
                                        st.session_state.df_points_data = cfg.get("points_data", {})
                                        st.session_state.df_camera_profiles = cfg.get("camera_profiles", {})
                                        st.session_state.df_image_camera_assignments = cfg.get("image_camera_assignments", {})
                                        st.session_state.df_object_rotation_angles = cfg.get("object_rotation_angles", {})
                                        st.session_state.df_offset_x = cfg.get("offset_x", 0.0)
                                        st.session_state.df_offset_y = cfg.get("offset_y", 0.0)
                                        st.session_state.df_offset_z = cfg.get("offset_z", 0.0)
                                        
                                        K_list = cfg.get("K_matrix")
                                        st.session_state.df_K_matrix = np.array(K_list) if K_list is not None else None
                                        
                                        dist_list = cfg.get("dist_coeffs")
                                        st.session_state.df_dist_coeffs = np.array(dist_list) if dist_list is not None else None
                                        
                                        st.session_state.df_calibration_rms = cfg.get("calibration_rms")
                                        st.session_state.df_K_source = cfg.get("K_source", "estimada")
                                        
                                        shape_list = cfg.get("calibration_shape")
                                        st.session_state.df_calibration_shape = tuple(shape_list) if shape_list is not None else None
                                        st.session_state.df_use_ransac = cfg.get("use_ransac", True)
                                        st.session_state.df_stl_units = cfg.get("stl_units", "mm")
                                        
                                        cal_data = cfg.get("calibrations", {})
                                        st.session_state.df_calibrations = {}
                                        for cname, cal in cal_data.items():
                                            st.session_state.df_calibrations[cname] = {
                                                "rvec": np.array(cal["rvec"]),
                                                "tvec": np.array(cal["tvec"]),
                                                "K": np.array(cal["K"]),
                                                "C": np.array(cal["C"]),
                                                "pitch": cal["pitch"],
                                                "roll": cal["roll"],
                                                "yaw": cal["yaw"]
                                            }
                                    except Exception as json_err:
                                        st.error(f"Error parseando project_config.json: {json_err}")
                                        
                            # Cargar fotos
                            new_images = []
                            for f in files:
                                if f['name'] != 'project_config.json' and f['name'].lower().endswith(('.png', '.jpg', '.jpeg')):
                                    raw_img = drive_api.download_file(f['id'])
                                    if raw_img:
                                        try:
                                            pil_img, np_arr = load_uploaded_image(raw_img)
                                            new_images.append({"name": f['name'], "pil": pil_img, "np": np_arr, "raw": raw_img})
                                        except Exception as img_err:
                                            st.error(f"Error cargando foto {f['name']}: {img_err}")
                                            
                            st.session_state.df_images = new_images
                            st.session_state.df_project_id = proj_info['id']
                            st.session_state.df_project_name = sel_proj
                            st.session_state.df_current_image_idx = 0
                            st.session_state.df_last_clicks = {}
                            st.session_state.df_face_colors = None
                            
                        st.success(f"✅ Proyecto **{sel_proj}** cargado con éxito ({len(new_images)} fotos descargadas).")
                        st.rerun()

        with col_create:
            st.markdown("#### 🆕 Crear Proyecto Nuevo")
            nuevo_nombre = st.text_input("Nombre del Proyecto:", placeholder="Ej: Ensayo_Perfil_Betz")
            if st.button("🚀 CREAR PROYECTO", use_container_width=True, type="primary"):
                if nuevo_nombre.strip():
                    with st.spinner("Creando carpeta en Drive..."):
                        df_root_id = auth.get_folder_datafusion(username)
                        new_folder_id = drive_api.get_or_create_folder(nuevo_nombre.strip(), df_root_id)
                        
                        st.session_state.df_project_id = new_folder_id
                        st.session_state.df_project_name = nuevo_nombre.strip()
                        # Reset local variables
                        st.session_state.df_images = []
                        st.session_state.df_current_image_idx = 0
                        st.session_state.df_points_data = {}
                        st.session_state.df_last_clicks = {}
                        st.session_state.df_calibrations = {}
                        st.session_state.df_offset_x = 0.0
                        st.session_state.df_offset_y = 0.0
                        st.session_state.df_offset_z = 0.0
                        st.session_state.df_face_colors = None
                        
                    st.success(f"✅ Proyecto **{nuevo_nombre.strip()}** creado y configurado como activo.")
                    st.rerun()
                else:
                    st.warning("Ingrese un nombre de proyecto válido.")
                    
        # Acciones de guardado rápido si hay un proyecto activo
        if st.session_state.df_project_id:
            st.markdown("---")
            c_info, c_save = st.columns([2, 1])
            c_info.markdown(f"📂 Proyecto Activo: **{st.session_state.df_project_name}** | Fotos en memoria: **{len(st.session_state.df_images)}**")
            
            if c_save.button("💾 GUARDAR CAMBIOS EN GOOGLE DRIVE", use_container_width=True, type="primary"):
                with st.spinner("Guardando archivos y configuraciones en tu Drive..."):
                    # Serializar configuraciones
                    cfg_data = {
                        "points_data": st.session_state.df_points_data,
                        "camera_profiles": st.session_state.df_camera_profiles,
                        "image_camera_assignments": st.session_state.df_image_camera_assignments,
                        "object_rotation_angles": st.session_state.df_object_rotation_angles,
                        "offset_x": st.session_state.df_offset_x,
                        "offset_y": st.session_state.df_offset_y,
                        "offset_z": st.session_state.df_offset_z,
                        "K_matrix": st.session_state.df_K_matrix.tolist() if st.session_state.df_K_matrix is not None else None,
                        "dist_coeffs": st.session_state.df_dist_coeffs.tolist() if st.session_state.df_dist_coeffs is not None else None,
                        "calibration_rms": st.session_state.df_calibration_rms,
                        "K_source": st.session_state.df_K_source,
                        "calibration_shape": list(st.session_state.df_calibration_shape) if st.session_state.df_calibration_shape is not None else None,
                        "use_ransac": st.session_state.df_use_ransac,
                        "stl_units": st.session_state.df_stl_units,
                        "calibrations": {}
                    }
                    
                    # Convert calibrations
                    for cname, cal in st.session_state.df_calibrations.items():
                        cfg_data["calibrations"][cname] = {
                            "rvec": cal["rvec"].tolist(),
                            "tvec": cal["tvec"].tolist(),
                            "K": cal["K"].tolist(),
                            "C": cal["C"].tolist(),
                            "pitch": cal["pitch"],
                            "roll": cal["roll"],
                            "yaw": cal["yaw"]
                        }
                        
                    json_str = json.dumps(cfg_data, indent=2)
                    # Subir config
                    drive_api.upload_file(json_str, 'project_config.json', st.session_state.df_project_id, mimetype='application/json')
                    
                    # Subir imágenes
                    for img in st.session_state.df_images:
                        drive_api.upload_file(img['raw'], img['name'], st.session_state.df_project_id, mimetype='image/png')
                        
                st.success("✅ Todo guardado correctamente en tu Google Drive!")

    # ═══════════════════════════════════════════════════════════════
    #  CHEQUEO DE PROYECTO ACTIVO E INTEGRACIÓN DE MODELO
    # ═══════════════════════════════════════════════════════════════
    if not st.session_state.df_project_id:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.warning("⚠️ **Sin Proyecto Activo:** Por favor cree o seleccione un proyecto en el panel superior antes de continuar con la calibración y el mapeo.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Detección del modelo 3D activo
    global_model_detected = False
    v, f_arr = None, None
    model_name = ""
    
    if 'objeto_referencia_4d' in st.session_state and st.session_state.objeto_referencia_4d is not None:
        obj = st.session_state.objeto_referencia_4d
        if obj.get('type') == 'mesh':
            global_model_detected = True
            v = np.stack([obj['x'], obj['y'], obj['z']], axis=1)
            f_arr = np.stack([obj['i'], obj['j'], obj['k']], axis=1).astype(np.int32)
            model_name = obj.get('name', 'Modelo Global')
            
    # Sidebar local para el módulo
    st.sidebar.markdown("### 🎨 Configuración Data Fusion")
    st.session_state.df_stl_units = st.sidebar.selectbox(
        "Unidad del Modelo STL:",
        ["mm", "cm", "m"],
        index=["mm", "cm", "m"].index(st.session_state.df_stl_units),
        key="df_stl_units_sidebar"
    )
    
    if global_model_detected:
        st.sidebar.success(f"✅ Malla STL Activa: `{model_name}`\nCaras: {len(f_arr)} | Vértices: {len(v)}")
    else:
        st.sidebar.warning("⚠️ Sin malla STL activa en la sección **MODELOS**.")

    # ═══════════════════════════════════════════════════════════════
    #  TABS DE SECCIÓN ACTIVAS
    # ═══════════════════════════════════════════════════════════════
    sub_page = st.radio(
        "Seleccionar fase de trabajo:",
        [
            "📷 0. Parámetros de Cámara",
            "📂 1. Carga de Fotos",
            "🎯 2. Marcador 2D-3D",
            "🔭 3. Calibración de Poses (SolvePnP)",
            "🎨 4. Proyección Texturas",
        ],
        horizontal=True,
        label_visibility="collapsed"
    )
    st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    #  TABS 0: PARÁMETROS DE CÁMARA
    # ═══════════════════════════════════════════════════════════════
    if sub_page == "📷 0. Parámetros de Cámara":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("📷 Calibración de Lente y Matriz K")
        st.caption("Especifique los parámetros geométricos del lente para eliminar la distorsión del SolvePnP.")
        
        pm = st.session_state.df_pro_mode
        disabled = pm["locked"]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            iso_opts = [50, 100, 200, 400, 800, 1600, 3200]
            pm["iso"] = st.selectbox("ISO", iso_opts, index=iso_opts.index(pm["iso"]) if pm["iso"] in iso_opts else 1, disabled=disabled, key="df_iso")
        with col2:
            shut_opts = ["1/1000", "1/500", "1/250", "1/125", "1/60", "1/30"]
            pm["shutter"] = st.selectbox("Obturador", shut_opts, index=shut_opts.index(pm["shutter"]) if pm["shutter"] in shut_opts else 2, disabled=disabled, key="df_shutter")
        with col3:
            focus_opts = ["Manual — Infinity", "Manual — Fixed", "AF-S"]
            pm["focus"] = st.selectbox("Enfoque", focus_opts, index=focus_opts.index(pm["focus"]) if pm["focus"] in focus_opts else 0, disabled=disabled, key="df_focus")
        with col4:
            wb_opts = ["Tungsten (2800K)", "Fluorescente neutro (4000K)", "Sol directo (5500K)", "Nublado (6000K)", "Sombra (7500K)"]
            pm["wb"] = st.selectbox("Balance de Blancos", wb_opts, index=wb_opts.index(pm["wb"]) if pm["wb"] in wb_opts else 1, disabled=disabled, key="df_wb")
            
        c_l, c_u = st.columns(2)
        if not pm["locked"]:
            if c_l.button("🔒 Bloquear Parámetros (Consistencia)", use_container_width=True):
                st.session_state.df_pro_mode["locked"] = True
                st.success("✅ Configuración bloqueada. Mantenga estos valores idénticos en su cámara física.")
                st.rerun()
        else:
            st.success("🔒 Parámetros de Cámara Bloqueados.")
            if c_u.button("🔓 Desbloquear", use_container_width=True):
                st.session_state.df_pro_mode["locked"] = False
                st.rerun()
                
        st.markdown("---")
        
        # Calibración Chessboard
        st.markdown("#### ♟️ Calibración con Tablero Ajedrez Chessboard")
        c_cols, c_rows, c_size = st.columns(3)
        cb_cols = c_cols.number_input("Esquinas Internas Horiz. (Columnas):", min_value=3, max_value=20, value=8)
        cb_rows = c_rows.number_input("Esquinas Internas Vert. (Filas):", min_value=3, max_value=20, value=5)
        square_size = c_size.number_input("Medida de Cuadrado [mm]:", min_value=1.0, max_value=100.0, value=30.0)
        
        cb_files = st.file_uploader("Arrastre fotos del tablero Chessboard aquí (15-20 recomendadas):", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if cb_files:
            if st.button("🔬 EJECUTAR CALIBRACIÓN DE LENTE", use_container_width=True, type="primary"):
                with st.spinner("Detectando subpíxeles y calibrando..."):
                    objp = np.zeros((cb_rows * cb_cols, 3), np.float32)
                    objp[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2) * square_size
                    
                    obj_points = []
                    img_points = []
                    img_shape = None
                    found = 0
                    
                    for f in cb_files:
                        raw = f.read()
                        _, np_cb = load_uploaded_image(raw)
                        gray = cv2.cvtColor(np_cb, cv2.COLOR_RGB2GRAY)
                        img_shape = gray.shape[::-1]
                        
                        ret, corners = cv2.findChessboardCorners(gray, (cb_cols, cb_rows), None)
                        if ret:
                            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                            obj_points.append(objp)
                            img_points.append(corners2)
                            found += 1
                            
                    if found < 4:
                        st.error(f"Fallo al calibrar: solo {found} fotos válidas detectadas. Se requieren al menos 4.")
                    else:
                        rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)
                        st.session_state.df_K_matrix = K
                        st.session_state.df_dist_coeffs = dist
                        st.session_state.df_calibration_rms = rms
                        st.session_state.df_K_source = "chessboard"
                        st.session_state.df_calibration_shape = img_shape
                        st.success(f"✅ Calibración finalizada con {found} fotos. RMS: {rms:.4f} px")
                        st.rerun()

        # Ingreso manual de K
        st.markdown("---")
        st.markdown("#### ✏️ Ingreso Manual de Parámetros Intrínsecos")
        mx_col1, mx_col2, mx_col3, mx_col4 = st.columns(4)
        m_fx = mx_col1.number_input("fx (focal px horizontal)", value=float(st.session_state.df_K_matrix[0,0]) if st.session_state.df_K_matrix is not None else 3000.0)
        m_fy = mx_col2.number_input("fy (focal px vertical)", value=float(st.session_state.df_K_matrix[1,1]) if st.session_state.df_K_matrix is not None else 3000.0)
        m_cx = mx_col3.number_input("cx (centro óptico X)", value=float(st.session_state.df_K_matrix[0,2]) if st.session_state.df_K_matrix is not None else 2016.0)
        m_cy = mx_col4.number_input("cy (centro óptico Y)", value=float(st.session_state.df_K_matrix[1,2]) if st.session_state.df_K_matrix is not None else 1512.0)
        
        if st.button("💾 GUARDAR K MANUALMENTE", use_container_width=True):
            st.session_state.df_K_matrix = np.array([[m_fx, 0, m_cx], [0, m_fy, m_cy], [0, 0, 1]], dtype=np.float64)
            st.session_state.df_dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            st.session_state.df_K_source = "manual"
            st.success("✅ Matriz K guardada manualmente.")
            st.rerun()

        if st.session_state.df_K_matrix is not None:
            st.markdown("##### 📊 Matriz K Actual:")
            st.markdown(K_matrix_display(st.session_state.df_K_matrix))
            
        st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    #  TABS 1: CARGA DE FOTOS
    # ═══════════════════════════════════════════════════════════════
    elif sub_page == "📂 1. Carga de Fotos":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("📸 Imágenes del Ensayo (Oil Flow / Visualizaciones)")
        
        uploaded = st.file_uploader("Arrastre sus fotos de ensayo aquí (.png, .jpg, .jpeg):", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded:
            new_imgs = list(st.session_state.df_images)
            for uf in uploaded:
                exists = next((img for img in new_imgs if img["name"] == uf.name), None)
                if not exists:
                    try:
                        raw = uf.read()
                        pil_img, np_arr = load_uploaded_image(raw)
                        new_imgs.append({"name": uf.name, "pil": pil_img, "np": np_arr, "raw": raw})
                        if uf.name not in st.session_state.df_points_data:
                            st.session_state.df_points_data[uf.name] = []
                    except Exception as e:
                        st.error(f"Error al analizar '{uf.name}': {e}")
            st.session_state.df_images = new_imgs
            st.success(f"✅ {len(st.session_state.df_images)} imágenes cargadas y listas en el proyecto.")
            
        if st.session_state.df_images:
            st.markdown("---")
            st.markdown("##### Fotos actualmente en el proyecto:")
            for idx, img in enumerate(st.session_state.df_images):
                col_n, col_btn = st.columns([4, 1])
                col_n.write(f"🖼️ `{img['name']}` ({img['pil'].size[0]}x{img['pil'].size[1]} px)")
                if col_btn.button("🗑️ Quitar", key=f"del_img_{idx}"):
                    st.session_state.df_images.pop(idx)
                    st.session_state.df_points_data.pop(img['name'], None)
                    st.session_state.df_last_clicks.pop(img['name'], None)
                    st.session_state.df_calibrations.pop(img['name'], None)
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    #  TABS 2: MARCADOR 2D-3D
    # ═══════════════════════════════════════════════════════════════
    elif sub_page == "🎯 2. Marcador 2D-3D":
        from streamlit_image_coordinates import streamlit_image_coordinates
        
        if not st.session_state.df_images:
            st.warning("⚠️ Cargue al menos una foto en la pestaña anterior para habilitar el marcador.")
            return
            
        if not global_model_detected:
            st.warning("⚠️ Debe cargar una malla STL en la pestaña global **MODELOS** para poder marcar sus coordenadas 3D y realizar snapping.")
            return

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("🎯 Marcado Interactivo y Vinculación 3D")
        st.markdown("<p style='color:#bbb;'>Haga clic en la imagen para marcar los puntos de control fiduciales físicos. Ingrese sus coordenadas 3D o permita que se ajusten con snapping automático.</p>", unsafe_allow_html=True)

        imgs = st.session_state.df_images
        n_imgs = len(imgs)
        
        col_prev, col_title, col_next = st.columns([1, 4, 1])
        with col_prev:
            if st.button("◀ Anterior", use_container_width=True) and n_imgs > 1:
                st.session_state.df_current_image_idx = (st.session_state.df_current_image_idx - 1) % n_imgs
                st.rerun()
        with col_title:
            st.markdown(f"<h4 style='text-align:center;color:#a78bfa;'>Imagen {st.session_state.df_current_image_idx + 1} de {n_imgs}</h4>", unsafe_allow_html=True)
        with col_next:
            if st.button("Siguiente ▶", use_container_width=True) and n_imgs > 1:
                st.session_state.df_current_image_idx = (st.session_state.df_current_image_idx + 1) % n_imgs
                st.rerun()

        active_img = imgs[st.session_state.df_current_image_idx]
        image_name = active_img["name"]
        pil_img = active_img["pil"]
        orig_w, orig_h = pil_img.size

        # Aplicar undistort si está calibrado
        is_undistorted = False
        if st.session_state.df_K_matrix is not None and st.session_state.df_dist_coeffs is not None and st.session_state.df_K_source == "chessboard":
            scaled_K = get_scaled_K(st.session_state.df_K_matrix, target_shape=(orig_w, orig_h), source_shape=st.session_state.df_calibration_shape)
            np_orig = np.array(pil_img)
            np_disp = cv2.undistort(np_orig, scaled_K, st.session_state.df_dist_coeffs)
            pil_img_for_display = Image.fromarray(np_disp)
            is_undistorted = True
        else:
            pil_img_for_display = pil_img

        # Resize for display
        max_width = 900
        if orig_w > max_width:
            scale_factor = max_width / orig_w
            pil_display = pil_img_for_display.resize((max_width, int(orig_h * scale_factor)), Image.Resampling.LANCZOS)
        else:
            scale_factor = 1.0
            pil_display = pil_img_for_display

        st.markdown(f"**Archivo Activo:** `{image_name}` ({orig_w}x{orig_h} px)")
        
        points = st.session_state.df_points_data.get(image_name, [])
        marked_pil = draw_points_on_image(pil_display, points, scale_factor)

        # Coordinate selection widget with stable key to prevent iframe unmounting/freezing
        value = streamlit_image_coordinates(
            marked_pil, 
            key=f"coords_df_clicker_{image_name}"
        )
        if value is not None:
            click_xy = (value["x"], value["y"])
            last_xy = st.session_state.df_last_clicks.get(image_name)
            if click_xy != last_xy:
                st.session_state.df_last_clicks[image_name] = click_xy
                orig_u = float(click_xy[0] / scale_factor)
                orig_v = float(click_xy[1] / scale_factor)
                st.session_state.df_points_data[image_name].append(
                    {"u": orig_u, "v": orig_v, "X": 0.0, "Y": 0.0, "Z": 0.0}
                )
                st.rerun()

        col_del, col_clear = st.columns(2)
        if col_del.button("🗑️ Eliminar último punto", use_container_width=True) and points:
            st.session_state.df_points_data[image_name].pop()
            st.session_state.df_last_clicks.pop(image_name, None)
            st.rerun()
        if col_clear.button("🧹 Limpiar todos los puntos", use_container_width=True) and points:
            st.session_state.df_points_data[image_name] = []
            st.session_state.df_last_clicks.pop(image_name, None)
            st.rerun()

        st.markdown("---")
        st.markdown("#### 📍 Coordenadas de los Puntos Fiduciales")
        points = st.session_state.df_points_data.get(image_name, [])
        if not points:
            st.info("Haga clic sobre la foto para agregar marcas.")
        else:
            units = st.session_state.df_stl_units
            snap_active = st.checkbox("Ajustar al vértice STL más cercano (Snapping)", value=False)
            
            # Rotación de objeto física si aplica
            rot_key = f"rot_{image_name}"
            if rot_key not in st.session_state.df_object_rotation_angles:
                st.session_state.df_object_rotation_angles[rot_key] = 0.0
            st.session_state.df_object_rotation_angles[rot_key] = st.number_input("Ángulo de rotación física del modelo alrededor de Z [grados]:", value=st.session_state.df_object_rotation_angles[rot_key], step=5.0)
            
            coord_mode = st.radio("Sistema de coordenadas de entrada:", ["Cartesiano (X, Y, Z)", "Cilíndrico (Z, R, φ)"], horizontal=True)
            
            with st.form(key=f"df_points_form_{image_name}"):
                st.markdown("Ingrese los valores físicos de coordenadas:")
                new_coords = []
                for i, pt in enumerate(points):
                    cols = st.columns([1, 1, 1, 2, 2, 2])
                    cols[0].write(f"P{i+1}")
                    cols[1].write(f"u: {int(pt['u'])}")
                    cols[2].write(f"v: {int(pt['v'])}")
                    
                    if coord_mode == "Cartesiano (X, Y, Z)":
                        rx = cols[3].number_input("X", value=float(pt["X"]), step=1.0, key=f"df_x_{image_name}_{i}", label_visibility="collapsed")
                        ry = cols[4].number_input("Y", value=float(pt["Y"]), step=1.0, key=f"df_y_{image_name}_{i}", label_visibility="collapsed")
                        rz = cols[5].number_input("Z", value=float(pt["Z"]), step=1.0, key=f"df_z_{image_name}_{i}", label_visibility="collapsed")
                        new_coords.append((rx, ry, rz))
                    else:
                        rz = cols[3].number_input("Z (Alt)", value=float(pt["Z"]), step=1.0, key=f"df_cz_{image_name}_{i}", label_visibility="collapsed")
                        curr_r = float(np.sqrt(pt["X"]**2 + pt["Y"]**2))
                        rr = cols[4].number_input("R (Rad)", value=curr_r, step=1.0, min_value=0.0, key=f"df_cr_{image_name}_{i}", label_visibility="collapsed")
                        curr_phi = float(np.degrees(np.arctan2(pt["Y"], pt["X"])))
                        rphi = cols[5].number_input("φ (Grd)", value=curr_phi, step=5.0, key=f"df_cphi_{image_name}_{i}", label_visibility="collapsed")
                        
                        # Convertir a cartesiano
                        phi_rad = np.radians(rphi)
                        rx = rr * np.cos(phi_rad)
                        ry = rr * np.sin(phi_rad)
                        new_coords.append((rx, ry, rz))
                        
                submitted = st.form_submit_button("💾 APLICAR Y AJUSTAR COORDENADAS", use_container_width=True)
                if submitted:
                    for idx in range(len(points)):
                        rx, ry, rz = new_coords[idx]
                        if snap_active and v is not None:
                            P_snap = snap_to_closest_vertex(np.array([rx, ry, rz]), v)
                            points[idx]["X"] = float(P_snap[0])
                            points[idx]["Y"] = float(P_snap[1])
                            points[idx]["Z"] = float(P_snap[2])
                        else:
                            points[idx]["X"] = rx
                            points[idx]["Y"] = ry
                            points[idx]["Z"] = rz
                    st.success("Coordenadas actualizadas.")
                    st.rerun()
                    
            if len(points) >= 4:
                st.success(f"✅ {len(points)} puntos listos para calibración SolvePnP.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    #  TABS 3: CALIBRACIÓN DE POSES (SOLVEPNP)
    # ═══════════════════════════════════════════════════════════════
    elif sub_page == "🔭 3. Calibración de Poses (SolvePnP)":
        if not global_model_detected:
            st.warning("⚠️ Suba un modelo STL en la sección **MODELOS**.")
            return
            
        ready_imgs = [img["name"] for img in st.session_state.df_images if len(st.session_state.df_points_data.get(img["name"], [])) >= 4]
        if not ready_imgs:
            st.warning("💡 Necesita al menos **4 puntos fiduciales** marcados en cada foto para calibrar.")
            return

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("🔭 Resección de Pose con SolvePnP")
        st.caption("Resuelve la posición 3D (X, Y, Z) de la cámara física y su orientación de ángulos de Euler.")

        # Robustez y offset
        st.session_state.df_use_ransac = st.checkbox("Usar RANSAC para ignorar puntos outliers", value=st.session_state.df_use_ransac)
        
        st.markdown("##### 📏 Corrección de Offset del origen STL:")
        off_cols = st.columns(3)
        st.session_state.df_offset_x = off_cols[0].number_input("Offset X:", value=st.session_state.df_offset_x, step=5.0)
        st.session_state.df_offset_y = off_cols[1].number_input("Offset Y:", value=st.session_state.df_offset_y, step=5.0)
        st.session_state.df_offset_z = off_cols[2].number_input("Offset Z:", value=st.session_state.df_offset_z, step=5.0)

        # Cámara profiles
        st.markdown("##### Perfiles de factor de longitud focal estimado:")
        updated = {}
        for cam_name, current in list(st.session_state.df_camera_profiles.items()):
            val = st.slider(f"Longitud Focal (Zoom) — {cam_name}", 0.5, 4.0, float(current), 0.05, key=f"df_slider_{cam_name}")
            updated[cam_name] = val
        st.session_state.df_camera_profiles = updated
        
        # Asignar perfiles
        st.markdown("##### Asignación de cámara a fotos:")
        cams = list(st.session_state.df_camera_profiles.keys())
        for img_name in ready_imgs:
            curr = st.session_state.df_image_camera_assignments.get(img_name, cams[0])
            if curr not in cams: curr = cams[0]
            st.session_state.df_image_camera_assignments[img_name] = st.selectbox(f"`{img_name}`:", cams, index=cams.index(curr), key=f"df_assign_{img_name}")

        if st.button("🚀 INICIAR CALIBRACIÓN DE CÁMARAS", use_container_width=True, type="primary"):
            with st.spinner("Ejecutando algoritmos SolvePnP robustos..."):
                cal_count = 0
                for img_name in ready_imgs:
                    pts = st.session_state.df_points_data[img_name]
                    img_dict = next(i for i in st.session_state.df_images if i["name"] == img_name)
                    H, W, _ = img_dict["np"].shape
                    
                    obj_pts = np.array([[p["X"], p["Y"], p["Z"]] for p in pts], dtype=np.float64)
                    obj_pts[:, 0] += st.session_state.df_offset_x
                    obj_pts[:, 1] += st.session_state.df_offset_y
                    obj_pts[:, 2] += st.session_state.df_offset_z
                    
                    img_pts = np.array([[p["u"], p["v"]] for p in pts], dtype=np.float64)
                    
                    if st.session_state.df_K_matrix is not None:
                        K = get_scaled_K(st.session_state.df_K_matrix, target_shape=(W, H), source_shape=st.session_state.df_calibration_shape)
                    else:
                        assigned = st.session_state.df_image_camera_assignments.get(img_name, cams[0])
                        zoom = st.session_state.df_camera_profiles.get(assigned, 1.2)
                        K = estimate_camera_matrix(W, H, zoom)
                        
                    dist = st.session_state.df_dist_coeffs if st.session_state.df_K_source == "chessboard" else None
                    rvec, tvec, C, pitch, roll, yaw, success = calibrate_camera(obj_pts, img_pts, K, dist, st.session_state.df_use_ransac)
                    
                    if success:
                        st.session_state.df_calibrations[img_name] = {
                            "rvec": rvec, "tvec": tvec, "K": K, "C": C, "pitch": pitch, "roll": roll, "yaw": yaw
                        }
                        cal_count += 1
                if cal_count > 0:
                    st.success(f"🎉 Calibración de poses resuelta para {cal_count} foto(s)!")
                    st.session_state.df_face_colors = None
                    st.rerun()

        # Mostrar tabla de resultados y visualización 3D
        if st.session_state.df_calibrations:
            st.markdown("---")
            st.markdown("#### 🔭 Posición y Ángulos Calculados:")
            rows = []
            units = st.session_state.df_stl_units
            for name, cal in st.session_state.df_calibrations.items():
                C = cal["C"]
                rows.append({
                    "Foto": name,
                    f"X [{units}]": round(C[0], 2),
                    f"Y [{units}]": round(C[1], 2),
                    f"Z [{units}]": round(C[2], 2),
                    "Pitch": f"{cal['pitch']:.1f}°",
                    "Roll": f"{cal['roll']:.1f}°",
                    "Yaw": f"{cal['yaw']:.1f}°"
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            
            st.markdown("#### 🌐 Entorno 3D Reconstruido:")
            span = np.linalg.norm(v.max(axis=0) - v.min(axis=0))
            scale = span * 0.12
            
            traces = [go.Mesh3d(
                x=v[:,0] + st.session_state.df_offset_x,
                y=v[:,1] + st.session_state.df_offset_y,
                z=v[:,2] + st.session_state.df_offset_z,
                i=f_arr[:,0], j=f_arr[:,1], k=f_arr[:,2],
                color='#475569', opacity=0.5, flatshading=True
            )]
            
            for name, cal in st.session_state.df_calibrations.items():
                xs, ys, zs = get_camera_wireframe(cal["C"], cal["rvec"], cal["tvec"], scale)
                traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color='#a78bfa', width=3), name=f"Cam: {name}"))
                traces.append(go.Scatter3d(x=[cal["C"][0]], y=[cal["C"][1]], z=[cal["C"][2]], mode='markers+text', text=[name], textposition="top center", marker=dict(color='#ef4444', size=7), name=f"Centro: {name}"))
                
            fig = go.Figure(data=traces)
            fig.update_layout(
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", bgcolor="#090d16", aspectmode="data"),
                paper_bgcolor="rgba(0,0,0,0)", height=600, margin=dict(l=0,r=0,b=0,t=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    #  TABS 4: PROYECCIÓN TEXTURAS
    # ═══════════════════════════════════════════════════════════════
    elif sub_page == "🎨 4. Proyección Texturas":
        if not global_model_detected:
            st.warning("⚠️ Cargue un modelo STL.")
            return
            
        if not st.session_state.df_calibrations:
            st.warning("⚠️ Debe calibrar al menos una cámara en la pestaña anterior.")
            return

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("🎨 Proyección Inversa de Textura (Backward Projection)")
        st.caption("Pinta los colores reales de las fotografías sobre las caras visibles de la malla 3D.")

        if st.button("🎨 INICIAR TEXTURIZADO DEL MODELO", use_container_width=True, type="primary"):
            with st.spinner("Ejecutando mapeo vectorizado..."):
                try:
                    images_dict = {}
                    for img in st.session_state.df_images:
                        arr = img["np"].copy()
                        if st.session_state.df_K_matrix is not None and st.session_state.df_dist_coeffs is not None and st.session_state.df_K_source == "chessboard":
                            H, W, _ = arr.shape
                            scaled_K = get_scaled_K(st.session_state.df_K_matrix, target_shape=(W, H), source_shape=st.session_state.df_calibration_shape)
                            arr = cv2.undistort(arr, scaled_K, st.session_state.df_dist_coeffs)
                        images_dict[img["name"]] = arr
                        
                    v_offsetted = v.copy()
                    v_offsetted[:, 0] += st.session_state.df_offset_x
                    v_offsetted[:, 1] += st.session_state.df_offset_y
                    v_offsetted[:, 2] += st.session_state.df_offset_z
                    
                    colors = backward_projection(v_offsetted, f_arr, images_dict, st.session_state.df_calibrations)
                    st.session_state.df_face_colors = colors
                    st.success("✅ Texturizado completado con éxito!")
                except Exception as e:
                    st.error(f"Fallo en Backward Projection: {e}")

        if st.session_state.df_face_colors is not None:
            st.markdown("#### 🌐 Visualización del Modelo Texturizado:")
            units = st.session_state.df_stl_units
            
            fig = go.Figure(data=[go.Mesh3d(
                x=v[:,0] + st.session_state.df_offset_x,
                y=v[:,1] + st.session_state.df_offset_y,
                z=v[:,2] + st.session_state.df_offset_z,
                i=f_arr[:,0], j=f_arr[:,1], k=f_arr[:,2],
                facecolor=st.session_state.df_face_colors,
                opacity=1.0, flatshading=True,
                lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1, roughness=0.8)
            )])
            
            fig.update_layout(
                scene=dict(xaxis_title=f"X [{units}]", yaxis_title=f"Y [{units}]", zaxis_title=f"Z [{units}]", bgcolor="#090d16", aspectmode="data"),
                paper_bgcolor="rgba(0,0,0,0)", height=750, margin=dict(l=0,r=0,b=0,t=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.balloons()
            
        st.markdown("</div>", unsafe_allow_html=True)
