import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
import plotly.graph_objects as go
import io

# Funciones de utils.py integradas más abajo


# ═══════════════════════════════════════════════════════════════
#  FUNCIONES DE UTILS.PY (INTEGRADAS)
# ═══════════════════════════════════════════════════════════════

import struct
import numpy as np
import cv2

def parse_stl(file_bytes):
    """
    Parsea archivos STL (tanto binarios como ASCII) a partir de un buffer de bytes.
    Retorna (vertices, faces) como numpy arrays.
    """
    if len(file_bytes) < 84:
        raise ValueError("El archivo es demasiado pequeño para ser un STL válido.")
    
    # Leer el número de triángulos para verificar si coincide con el tamaño del archivo binario
    num_triangles = struct.unpack('<I', file_bytes[80:84])[0]
    expected_size = 84 + num_triangles * 50
    
    if len(file_bytes) == expected_size:
        return parse_binary_stl(file_bytes, num_triangles)
    else:
        return parse_ascii_stl(file_bytes)

def parse_binary_stl(file_bytes, num_triangles):
    """Parsea STL en formato binario de forma super veloz utilizando NumPy."""
    buffer = file_bytes[84:]
    # Cada triángulo contiene: normal(3f4), v0(3f4), v1(3f4), v2(3f4), atributo(1u2) = 50 bytes
    dtype = np.dtype([
        ('normal', '<f4', (3,)),
        ('v0', '<f4', (3,)),
        ('v1', '<f4', (3,)),
        ('v2', '<f4', (3,)),
        ('attr', '<u2')
    ])
    mesh_data = np.frombuffer(buffer, dtype=dtype, count=num_triangles)
    
    # Apilar vértices a lo largo del eje 1 para conservar la contigüidad
    all_vertices = np.stack([mesh_data['v0'], mesh_data['v1'], mesh_data['v2']], axis=1) # shape (N, 3, 3)
    flat_vertices = all_vertices.reshape(-1, 3)
    
    # Encontrar vértices únicos para indexación compartida (necesario para Plotly Mesh3d)
    vertices, inverse = np.unique(flat_vertices, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    
    return vertices.astype(np.float64), faces.astype(np.int32)

def parse_ascii_stl(file_bytes):
    """Parsea un archivo STL en formato ASCII."""
    try:
        text = file_bytes.decode('utf-8', errors='ignore')
    except Exception:
        raise ValueError("No se pudo decodificar el archivo STL como ASCII.")
    
    vertices_list = []
    lines = text.split('\n')
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0].lower() == 'vertex':
            try:
                vertices_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                pass
                
    num_triangles = len(vertices_list) // 3
    if num_triangles == 0:
        raise ValueError("No se encontraron vértices válidos en el archivo ASCII STL.")
        
    triangles = np.array(vertices_list[:num_triangles * 3], dtype=np.float64).reshape(-1, 3, 3)
    flat_vertices = triangles.reshape(-1, 3)
    
    vertices, inverse = np.unique(flat_vertices, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    
    return vertices, faces

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

def calibrate_camera(object_points, image_points, K):
    """
    Ejecuta SolvePnP de manera extremadamente robusta utilizando una cascada
    de flags (SQPNP -> IPPE -> ITERATIVE -> EPNP) para evitar errores de DLT con pocos puntos.
    Retorna (rvec, tvec, C, pitch, roll, yaw, success)
    """
    if len(object_points) < 4:
        return None, None, None, 0, 0, 0, False
        
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
            # Intentar resolver usando el método actual
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
            # Capturar y continuar al siguiente si hay un fallo interno o falta de puntos en DLT
            continue
            
    if not success or rvec is None or tvec is None:
        return None, None, None, 0, 0, 0, False
        
    # Calcular matriz de rotación R
    R, _ = cv2.Rodrigues(rvec)
    
    # Centro de la cámara en coordenadas del mundo: C = -R^T * tvec
    C = -R.T @ tvec
    C = C.flatten()
    
    # Extracción de ángulos Euler (Pitch, Roll, Yaw) utilizando convención estándar
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
    """
    Calcula los puntos X, Y, Z del alambre de una pirámide en coordenadas del mundo,
    para graficar el frustum de la cámara en Plotly.
    """
    R, _ = cv2.Rodrigues(rvec)
    aspect = 4.0 / 3.0
    w = scale * 0.4 * aspect
    h = scale * 0.4
    d = scale * 0.8
    
    # Esquinas en el espacio de la cámara
    corners_cam = np.array([
        [0, 0, 0],       # Centro (Ápice de la pirámide)
        [-w, -h, d],     # Esquina superior izquierda
        [w, -h, d],      # Esquina superior derecha
        [w, h, d],       # Esquina inferior derecha
        [-w, h, d]       # Esquina inferior izquierda
    ])
    
    # Transformar a coordenadas del mundo: P_world = R^T * P_cam + C
    corners_world = (R.T @ corners_cam.T).T + C.reshape(1, 3)
    
    # Trazar el camino de la pirámide 3D
    path = [
        corners_world[0], corners_world[1], corners_world[2], corners_world[0],
        corners_world[3], corners_world[4], corners_world[0],
        [None, None, None], # Salto
        corners_world[1], corners_world[2], corners_world[3], corners_world[4], corners_world[1]
    ]
    
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    zs = [p[2] for p in path]
    
    return xs, ys, zs

def backward_projection(vertices, faces, images, calibrations):
    """
    Mapeo de texturas inverso (Backward Projection) optimizado y vectorizado con NumPy.
    
    vertices: Array de vértices (V, 3)
    faces: Array de caras (F, 3)
    images: Diccionario de imágenes numpy (H, W, 3)
    calibrations: Diccionario de calibraciones resueltas {nombre_img: {rvec, tvec, K}}
    
    Retorna:
    face_colors: Lista de colores en formato 'rgb(r,g,b)' para cada una de las caras.
    """
    num_faces = len(faces)
    # Color por defecto: gris claro elegante
    face_colors = np.full((num_faces, 3), 220, dtype=np.uint8)
    
    if not calibrations or not images:
        return [f"rgb({c[0]},{c[1]},{c[2]})" for c in face_colors]
        
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Calcular centroides y normales de cada cara
    centroids = (v0 + v1 + v2) / 3.0
    face_normals = compute_face_normals(vertices, faces)
    
    camera_names = list(calibrations.keys())
    num_cameras = len(camera_names)
    
    if num_cameras == 0:
        return [f"rgb({c[0]},{c[1]},{c[2]})" for c in face_colors]
        
    # Inicializar vectores para seleccionar la mejor cámara por cara
    best_dot = np.full(num_faces, -np.inf)
    best_cam_idx = np.full(num_faces, -1, dtype=np.int32)
    
    for idx, cam_name in enumerate(camera_names):
        cal = calibrations[cam_name]
        rvec = cal['rvec']
        tvec = cal['tvec']
        R, _ = cv2.Rodrigues(rvec)
        
        # Centro de cámara
        C = -R.T @ tvec
        
        # Vector desde el centroide de la cara hacia la cámara
        vec_to_cam = C.flatten() - centroids
        vec_norms = np.linalg.norm(vec_to_cam, axis=1, keepdims=True)
        vec_norms = np.where(vec_norms == 0, 1.0, vec_norms)
        dir_to_cam = vec_to_cam / vec_norms
        
        # Producto escalar: si es positivo, la cara apunta en dirección a la cámara
        dots = np.sum(face_normals * dir_to_cam, axis=1)
        
        # Encontrar el valor máximo de visibilidad
        is_better = dots > best_dot
        best_dot = np.where(is_better, dots, best_dot)
        best_cam_idx = np.where(is_better, idx, best_cam_idx)
        
    # Proyectar y muestrear los colores de cada cara usando su mejor cámara
    for idx, cam_name in enumerate(camera_names):
        # Filtrar caras asignadas a esta cámara con ángulo de visibilidad positivo
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
        
        # Proyectar centroides: P_cam = R * P_world + tvec
        pts_cam = R @ active_centroids.T + tvec.reshape(3, 1)
        xc = pts_cam[0, :]
        yc = pts_cam[1, :]
        zc = pts_cam[2, :]
        
        # Filtrar puntos detrás de la lente
        valid_z = zc > 0.001
        if not np.any(valid_z):
            continue
            
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Coordenadas de píxel proyectadas
        u = fx * (xc / zc) + cx
        v = fy * (yc / zc) + cy
        
        # Comprobar límites de imagen
        valid_uv = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        valid_mask = valid_z & valid_uv
        
        if not np.any(valid_mask):
            continue
            
        # Muestrear el píxel más cercano (Nearest Neighbor)
        cols = u[valid_mask].astype(np.int32)
        rows = v[valid_mask].astype(np.int32)
        
        sampled_colors = img[rows, cols]
        
        # Guardar colores
        active_indices = np.where(faces_mask)[0]
        valid_face_indices = active_indices[valid_mask]
        face_colors[valid_face_indices] = sampled_colors

    return [f"rgb({c[0]},{c[1]},{c[2]})" for c in face_colors]


# ═══════════════════════════════════════════════════════════════
#  FIN DE UTILS.PY
# ═══════════════════════════════════════════════════════════════

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Data Fusion - STL Texture Mapper",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS Premium (Glassmorphism & Sleek Dark Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(17, 24, 39, 0.7);
        padding: 8px 12px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        color: #9ca3af;
        border-radius: 8px;
        padding: 10px 16px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        color: #7c3aed !important;
        background-color: rgba(124, 58, 237, 0.15) !important;
        font-weight: 600;
    }
    .glass-card {
        background: rgba(17, 24, 39, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.3);
    }
    [data-testid="stSidebar"] {
        background-color: #0d1222;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stButton>button {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45);
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
    }
    .rainbow-bar {
        height: 4px;
        background: linear-gradient(90deg, #7c3aed, #06b6d4, #10b981);
        border-radius: 2px;
        margin-bottom: 30px;
    }
    .info-box {
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.92em;
        color: #c4b5fd;
    }
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.92em;
        color: #fcd34d;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SESSION STATE — inicialización única
# ─────────────────────────────────────────
defaults = {
    "images": [],
    "current_image_idx": 0,
    "points_data": {},
    # FIX: guardamos el último click POR IMAGEN con su timestamp para no agregar el mismo punto dos veces
    "last_clicks": {},
    "calibrations": {},
    "vertices": None,
    "faces": None,
    "stl_units": "mm",
    "stl_file_name": "",
    "camera_profiles": {"Cámara A (Principal)": 1.2, "Cámara B (Secundaria)": 1.8},
    "image_camera_assignments": {},
    # Ángulo de rotación del objeto alrededor de Z por imagen (en grados)
    "object_rotation_angles": {},
    # Matriz K calibrada con tablero
    "K_matrix": None,
    "dist_coeffs": None,
    "calibration_rms": None,
    "K_source": "estimada",   # "estimada" | "chessboard" | "manual"
    # Modo Pro — bloqueo de parámetros de cámara
    "pro_mode": {
        "iso": 100,
        "shutter": "1/250",
        "focus": "Manual — Infinity",
        "wb": "Daylight / Sol directo (5500K)",
        "locked": False,
    },
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
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
    """Devuelve una representación markdown con LaTeX de la matriz K."""
    if K is None:
        return ""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return (
        f"$$K = \\begin{{bmatrix}} {fx:.1f} & 0 & {cx:.1f} \\\\ "
        f"0 & {fy:.1f} & {cy:.1f} \\\\ 0 & 0 & 1 \\end{{bmatrix}}$$"
    )

# ─────────────────────────────────────────
#  CABECERA
# ─────────────────────────────────────────
st.markdown("<h1>🎨 Plataforma Avanzada de Data Fusion</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #9ca3af; font-size: 1.15em;'>Mapeo interactivo de texturas 2D sobre modelos aerodinámicos 3D mediante Resección Fotogramétrica.</p>", unsafe_allow_html=True)
st.markdown("<div class='rainbow-bar'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛠️ Configuración Global")
    st.session_state.stl_units = st.selectbox(
        "Sistema de Unidades (3D)",
        ["mm", "cm", "m"],
        index=["mm", "cm", "m"].index(st.session_state.stl_units),
        help="Define la unidad física en la que está construido tu modelo STL.",
    )
    st.markdown("---")
    st.markdown("### 📋 Flujo de Trabajo")
    st.markdown("""
    0. **📷 Parámetros de Cámara**: Calibra la matriz K con tablero.
    1. **📂 Cargar Datos**: Sube tu `.stl` y las fotos.
    2. **🎯 Marcar Puntos**: Marca fiduciales y coordenadas 3D.
    3. **🔭 Calibrar Pose**: Resección con SolvePnP.
    4. **🎨 Proyectar**: Modelo texturizado final.
    """)
    st.markdown("---")
    # Estado de la K
    k_status = st.session_state.K_source
    if k_status == "chessboard":
        rms = st.session_state.calibration_rms
        st.success(f"✅ K calibrada (tablero) — RMS: {rms:.4f} px")
    elif k_status == "manual":
        st.info("ℹ️ K ingresada manualmente")
    else:
        st.warning("⚠️ K estimada (sin calibración)")
    st.caption("Laboratorio de Aerodinámica y Fluidos")

# ─────────────────────────────────────────
#  NAVEGACIÓN
# ─────────────────────────────────────────
page = st.radio(
    "Sección activa",
    [
        "📷 0. Parámetros de Cámara",
        "📂 1. Carga de Modelo y Fotos",
        "🎯 2. Marcador 2D-3D",
        "🔭 3. Calibración de Cámaras",
        "🎨 4. Texturizado Final",
    ],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  SECCIÓN 0 — PARÁMETROS DE CÁMARA (NUEVO)
# ══════════════════════════════════════════
if page == "📷 0. Parámetros de Cámara":

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📷 Configuración de Cámara y Cálculo de Matriz Intrínseca K")
    st.markdown("""
    <div class='info-box'>
    La matriz intrínseca <b>K</b> codifica la geometría interna del lente: longitud focal (fx, fy)
    y punto principal (cx, cy). Una K incorrecta invalida toda la fotogrametría.<br>
    Podés obtenerla de tres formas: calibración con tablero de ajedrez (recomendado), ingreso manual, o estimación automática.
    </div>
    """, unsafe_allow_html=True)

    # ── BLOQUEO MODO PRO ──────────────────────────────────────────
    st.markdown("#### 🔒 Bloqueo de Configuración de Cámara (Modo Pro)")
    st.markdown("""
    <div class='warning-box'>
    ⚠️ <b>Consistencia obligatoria:</b> Una vez que bloquees estos parámetros, deben mantenerse
    <i>idénticos</i> durante <b>toda la sesión de captura</b>. Cualquier cambio en ISO, obturador,
    enfoque o balance de blancos invalida la matriz K y arruina la fotogrametría.
    </div>
    """, unsafe_allow_html=True)

    pm = st.session_state.pro_mode
    disabled = pm["locked"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        iso_opts = [50, 100, 200, 400, 800, 1600, 3200]
        pm["iso"] = st.selectbox("ISO", iso_opts,
                                  index=iso_opts.index(pm["iso"]) if pm["iso"] in iso_opts else 1,
                                  disabled=disabled, key="iso_sel")
    with col2:
        shut_opts = ["1/1000", "1/500", "1/250", "1/125", "1/60", "1/30"]
        pm["shutter"] = st.selectbox("Velocidad de obturación", shut_opts,
                                      index=shut_opts.index(pm["shutter"]) if pm["shutter"] in shut_opts else 2,
                                      disabled=disabled, key="shutter_sel")
    with col3:
        focus_opts = ["Manual — Infinity", "Manual — Fixed", "AF-S (no recomendado)"]
        pm["focus"] = st.selectbox("Modo de enfoque", focus_opts,
                                    index=focus_opts.index(pm["focus"]) if pm["focus"] in focus_opts else 0,
                                    disabled=disabled, key="focus_sel")
    with col4:
        wb_opts = [
            "Tungsten / Incandescente (2800K)",
            "Tungsten cálido (3000K)",
            "Fluorescente cálido (3200K)",
            "Fluorescente neutro (4000K)",
            "Fluorescente frío (4500K)",
            "Horizonte / Amanecer (5000K)",
            "Daylight / Sol directo (5500K)",
            "Flash de estudio (5500K)",
            "Nublado (6000K)",
            "Cloudy / Overcast (6500K)",
            "Sombra exterior (7500K)",
            "Cielo azul / Sombra profunda (9000K)",
            "Manual Kelvin",
        ]
        pm["wb"] = st.selectbox("Balance de blancos", wb_opts,
                                 index=wb_opts.index(pm["wb"]) if pm["wb"] in wb_opts else 0,
                                 disabled=disabled, key="wb_sel")

    col_lock, col_unlock = st.columns(2)
    with col_lock:
        if not pm["locked"]:
            if st.button("🔒 Bloquear configuración", use_container_width=True):
                pm["locked"] = True
                st.success("✅ Configuración bloqueada. No modifiques estos parámetros durante la captura.")
        else:
            st.success("🔒 Configuración bloqueada y activa.")
    with col_unlock:
        if pm["locked"]:
            if st.button("🔓 Desbloquear (nueva sesión)", use_container_width=True):
                pm["locked"] = False
                st.warning("⚠️ Desbloqueado. Si cambias parámetros deberás recalibrar K.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── CALIBRACIÓN CON TABLERO ───────────────────────────────────
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("♟️ Calibración Estricta con Patrón Chessboard")
    st.markdown("""
    <div class='info-box'>
    Capturá entre <b>15 y 20 imágenes</b> del tablero en diversas posiciones, inclinaciones y distancias.
    El algoritmo usa <code>cv2.calibrateCamera</code> para obtener la matriz K y el vector de distorsión.
    <br>Una vez calibrado, todas las imágenes de ensayo pasarán por <code>cv2.undistort</code> automáticamente
    antes de la proyección inversa.
    </div>
    """, unsafe_allow_html=True)

    col_cb1, col_cb2 = st.columns(2)
    with col_cb1:
        cb_cols = st.number_input("Esquinas internas — columnas", min_value=3, max_value=20, value=5,
                                   help="Número de esquinas internas horizontales del tablero (no cuadrados).")
        cb_rows = st.number_input("Esquinas internas — filas", min_value=3, max_value=20, value=6,
                                   help="Número de esquinas internas verticales del tablero.")
    with col_cb2:
        square_size = st.number_input("Tamaño de cuadrado (mm)", min_value=1.0, max_value=200.0, value=25.0, step=0.5,
                                       help="Medida real del lado de cada cuadrado del tablero en milímetros.")

    chessboard_files = st.file_uploader(
        "Subir imágenes de calibración (15-20 fotos del tablero)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="cb_uploader",
        help="Sube todas las imágenes del tablero de ajedrez para calibrar la cámara.",
    )

    if chessboard_files:
        st.info(f"📸 {len(chessboard_files)} imágenes cargadas. Mínimo recomendado: 15.")
        if st.button("🔬 Calibrar con Tablero (cv2.calibrateCamera)", use_container_width=True):
            with st.spinner("Detectando esquinas y calibrando..."):
                objp = np.zeros((cb_rows * cb_cols, 3), np.float32)
                objp[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2) * square_size

                obj_points_all = []
                img_points_all = []
                img_shape = None
                found_count = 0
                failed = []

                progress = st.progress(0)
                for i, f in enumerate(chessboard_files):
                    raw = np.frombuffer(f.read(), np.uint8)
                    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_shape = gray.shape[::-1]

                    ret, corners = cv2.findChessboardCorners(gray, (cb_cols, cb_rows), None)
                    if ret:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        obj_points_all.append(objp)
                        img_points_all.append(corners2)
                        found_count += 1
                    else:
                        failed.append(f.name)
                    progress.progress((i + 1) / len(chessboard_files))

                if found_count < 4:
                    st.error(f"❌ Solo se encontraron {found_count} tableros válidos. Se necesitan al menos 4. Revisá las imágenes.")
                else:
                    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                        obj_points_all, img_points_all, img_shape, None, None
                    )
                    st.session_state.K_matrix = K
                    st.session_state.dist_coeffs = dist
                    st.session_state.calibration_rms = rms
                    st.session_state.K_source = "chessboard"

                    if failed:
                        st.warning(f"⚠️ No se encontró tablero en {len(failed)} imágen(es): {', '.join(failed)}")
                    st.success(f"✅ Calibración exitosa con {found_count} imágenes válidas. RMS = {rms:.4f} px")
                    if rms > 1.0:
                        st.warning("⚠️ El error RMS es mayor a 1 px. Capturá más imágenes con mayor variedad de posiciones.")

    # Mostrar K actual
    if st.session_state.K_matrix is not None:
        st.markdown("##### Matriz K obtenida:")
        st.markdown(K_matrix_display(st.session_state.K_matrix))
        if st.session_state.dist_coeffs is not None:
            dc = st.session_state.dist_coeffs.flatten()
            st.markdown(f"**Coeficientes de distorsión:** k1={dc[0]:.5f}, k2={dc[1]:.5f}, p1={dc[2]:.5f}, p2={dc[3]:.5f}" +
                        (f", k3={dc[4]:.5f}" if len(dc) > 4 else ""))

    st.markdown("</div>", unsafe_allow_html=True)

    # ── INGRESO MANUAL DE K ───────────────────────────────────────
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("✏️ Ingreso Manual de Parámetros de Cámara")
    st.caption("Usá esto si ya tenés la especificación técnica de tu cámara/lente o un valor conocido.")

    col_fx, col_fy, col_cx, col_cy = st.columns(4)
    with col_fx:
        m_fx = st.number_input("fx (focal px horizontal)", value=float(st.session_state.K_matrix[0,0]) if st.session_state.K_matrix is not None else 3000.0, step=10.0, key="m_fx")
    with col_fy:
        m_fy = st.number_input("fy (focal px vertical)", value=float(st.session_state.K_matrix[1,1]) if st.session_state.K_matrix is not None else 3000.0, step=10.0, key="m_fy")
    with col_cx:
        m_cx = st.number_input("cx (centro óptico X)", value=float(st.session_state.K_matrix[0,2]) if st.session_state.K_matrix is not None else 2016.0, step=1.0, key="m_cx")
    with col_cy:
        m_cy = st.number_input("cy (centro óptico Y)", value=float(st.session_state.K_matrix[1,2]) if st.session_state.K_matrix is not None else 1512.0, step=1.0, key="m_cy")

    if st.button("💾 Guardar K manual", use_container_width=True):
        K_manual = np.array([[m_fx, 0, m_cx], [0, m_fy, m_cy], [0, 0, 1]], dtype=np.float64)
        st.session_state.K_matrix = K_manual
        st.session_state.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        st.session_state.K_source = "manual"
        st.success("✅ Matriz K guardada manualmente.")
        st.markdown(K_matrix_display(K_manual))

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  SECCIÓN 1 — CARGA DE DATOS
# ══════════════════════════════════════════
elif page == "📂 1. Carga de Modelo y Fotos":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📂 Carga del Modelo STL y Fotografías")

    col_stl, col_photos = st.columns(2)

    with col_stl:
        st.markdown("#### 1. Malla 3D (Modelo STL)")
        stl_file = st.file_uploader("Cargar archivo STL (.stl)", type=["stl"])
        if stl_file is not None:
            if st.session_state.stl_file_name != stl_file.name:
                st.session_state.stl_file_name = stl_file.name
                with st.spinner("Decodificando malla STL..."):
                    try:
                        v, f = parse_stl(stl_file.read())
                        st.session_state.vertices = v
                        st.session_state.faces = f
                        if "face_colors" in st.session_state:
                            del st.session_state.face_colors
                    except Exception as e:
                        st.error(f"Error al analizar STL: {e}")
            if st.session_state.vertices is not None:
                v = st.session_state.vertices
                f_arr = st.session_state.faces
                st.success(f"✅ `{stl_file.name}` | {len(f_arr)} caras | {len(v)} vértices")
        else:
            st.session_state.vertices = None
            st.session_state.faces = None

    with col_photos:
        st.markdown("#### 2. Imágenes de Ensayo (Oil Flow / Humo)")
        uploaded_files = st.file_uploader(
            "Cargar fotos (.png, .jpg, .jpeg)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            new_images = []
            for uf in uploaded_files:
                exists = next((img for img in st.session_state.images if img["name"] == uf.name), None)
                if exists:
                    new_images.append(exists)
                else:
                    try:
                        raw = uf.read()
                        pil_img = Image.open(io.BytesIO(raw))
                        np_arr = np.array(pil_img)
                        if len(np_arr.shape) == 2:
                            np_arr = cv2.cvtColor(np_arr, cv2.COLOR_GRAY2RGB)
                        elif np_arr.shape[2] == 4:
                            np_arr = cv2.cvtColor(np_arr, cv2.COLOR_RGBA2RGB)
                        new_images.append({"name": uf.name, "pil": pil_img, "np": np_arr, "raw": raw})
                        if uf.name not in st.session_state.points_data:
                            st.session_state.points_data[uf.name] = []
                    except Exception as e:
                        st.error(f"Error al cargar '{uf.name}': {e}")
            st.session_state.images = new_images
            st.success(f"✅ {len(st.session_state.images)} imágenes preparadas.")
        else:
            st.session_state.images = []
            st.session_state.points_data = {}
            st.session_state.last_clicks = {}
            st.session_state.calibrations = {}

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.vertices is not None:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🌐 Vista de Geometría STL")
        v = st.session_state.vertices
        f_arr = st.session_state.faces
        units = st.session_state.stl_units
        fig = go.Figure(data=[go.Mesh3d(
            x=v[:,0], y=v[:,1], z=v[:,2],
            i=f_arr[:,0], j=f_arr[:,1], k=f_arr[:,2],
            color='#3b82f6', opacity=0.9, flatshading=False,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.5)
        )])
        fig.update_layout(
            scene=dict(xaxis_title=f"X [{units}]", yaxis_title=f"Y [{units}]", zaxis_title=f"Z [{units}]", bgcolor="#0f172a", aspectmode="data"),
            margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="rgba(0,0,0,0)", height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  SECCIÓN 2 — MARCADOR 2D-3D (FIXED)
# ══════════════════════════════════════════
elif page == "🎯 2. Marcador 2D-3D":
    from streamlit_image_coordinates import streamlit_image_coordinates

    if not st.session_state.images:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.warning("⚠️ Sube al menos una imagen en la pestaña 1 para habilitar el marcador.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🎯 Marcado de Puntos Fiduciarios")

        # ── Nota sobre condición de colinealidad ─────────────────
        st.markdown("""
        <div class='info-box'>
        <b>Condición de colinealidad:</b> Cada punto marcado define una línea recta entre
        el centro óptico de la cámara, el punto 3D en el modelo y el píxel 2D en la imagen.
        Marcá puntos con coordenadas 3D bien conocidas (fiduciales físicos medidos).
        </div>
        """, unsafe_allow_html=True)

        imgs = st.session_state.images
        n_imgs = len(imgs)

        col_prev, col_title, col_next = st.columns([1, 4, 1])
        with col_prev:
            if st.button("◀ Anterior", use_container_width=True):
                st.session_state.current_image_idx = (st.session_state.current_image_idx - 1) % n_imgs
        with col_title:
            st.markdown(f"<h4 style='text-align:center;color:#a78bfa;'>Imagen {st.session_state.current_image_idx + 1} de {n_imgs}</h4>", unsafe_allow_html=True)
        with col_next:
            if st.button("Siguiente ▶", use_container_width=True):
                st.session_state.current_image_idx = (st.session_state.current_image_idx + 1) % n_imgs

        active_img = imgs[st.session_state.current_image_idx]
        image_name = active_img["name"]
        pil_img = active_img["pil"]
        orig_w, orig_h = pil_img.size

        # Redimensionar para visualización
        max_display_width = 1000
        if orig_w > max_display_width:
            scale_factor = max_display_width / orig_w
            cache_key_resize = f"resized_{image_name}"
            if cache_key_resize not in st.session_state:
                st.session_state[cache_key_resize] = pil_img.resize(
                    (max_display_width, int(orig_h * scale_factor)), Image.Resampling.LANCZOS
                )
            pil_display = st.session_state[cache_key_resize]
        else:
            scale_factor = 1.0
            pil_display = pil_img

        # Aplicar undistort si tenemos K calibrada
        if st.session_state.K_matrix is not None and st.session_state.dist_coeffs is not None:
            np_disp = np.array(pil_display)
            np_disp = cv2.undistort(np_disp, st.session_state.K_matrix, st.session_state.dist_coeffs)
            pil_display = Image.fromarray(np_disp)
            st.caption("🔧 Imagen con corrección de distorsión de lente aplicada (undistort).")

        st.markdown(f"**Activo:** `{image_name}` ({orig_w}×{orig_h} → display {pil_display.size[0]}×{pil_display.size[1]})")

        points = st.session_state.points_data.get(image_name, [])

        # Cache de imagen dibujada — se invalida solo cuando cambia la cantidad de puntos
        if "drawn_images" not in st.session_state:
            st.session_state.drawn_images = {}
        draw_cache_key = f"{image_name}_{len(points)}"
        if draw_cache_key not in st.session_state.drawn_images:
            st.session_state.drawn_images[draw_cache_key] = draw_points_on_image(pil_display, points, scale_factor)
        marked_pil = st.session_state.drawn_images[draw_cache_key]

        # ── CLICK HANDLER — FIX PRINCIPAL ──────────────────────
        # La clave del fix: comparamos (x, y) con el último click guardado.
        # Si son iguales NO agregamos punto, evitando duplicados en reruns.
        # NO llamamos st.rerun() manualmente — Streamlit lo hace solo al cambiar session_state.
        value = streamlit_image_coordinates(marked_pil, key=f"coords_clicker_{image_name}")

        if value is not None:
            click_xy = (value["x"], value["y"])
            last_xy = st.session_state.last_clicks.get(image_name)
            if click_xy != last_xy:
                # Nuevo click — guardar y agregar punto
                st.session_state.last_clicks[image_name] = click_xy
                orig_u = float(click_xy[0] / scale_factor)
                orig_v = float(click_xy[1] / scale_factor)
                st.session_state.points_data[image_name].append(
                    {"u": orig_u, "v": orig_v, "X": 0.0, "Y": 0.0, "Z": 0.0}
                )
                # Invalidar cache del dibujo para que se redibuje con el nuevo punto
                keys_to_del = [k for k in st.session_state.drawn_images if k.startswith(image_name)]
                for k in keys_to_del:
                    del st.session_state.drawn_images[k]
                # NO llamamos st.rerun() explícitamente —
                # Streamlit hace rerun automático al modificar session_state desde un widget.

        col_del, col_clear = st.columns(2)
        with col_del:
            if st.button("🗑️ Eliminar último punto", use_container_width=True) and points:
                st.session_state.points_data[image_name].pop()
                st.session_state.last_clicks.pop(image_name, None)
                keys_to_del = [k for k in st.session_state.drawn_images if k.startswith(image_name)]
                for k in keys_to_del:
                    del st.session_state.drawn_images[k]
        with col_clear:
            if st.button("🧹 Limpiar todos los puntos", use_container_width=True) and points:
                st.session_state.points_data[image_name] = []
                st.session_state.last_clicks.pop(image_name, None)
                keys_to_del = [k for k in st.session_state.drawn_images if k.startswith(image_name)]
                for k in keys_to_del:
                    del st.session_state.drawn_images[k]

        st.markdown("</div>", unsafe_allow_html=True)

        # ── TABLA DE COORDENADAS 3D — FIX data_editor ─────────────
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader(f"📍 Correspondencia 2D-3D: {image_name}")

        points = st.session_state.points_data.get(image_name, [])  # refrescar tras modificaciones

        if not points:
            st.info("Hacé click sobre la foto para agregar puntos fiduciarios.")
        else:
            units = st.session_state.stl_units
            st.markdown(f"Ingresá las coordenadas 3D reales de cada punto marcado (unidades: **{units}**).")

            snap_active = False
            if st.session_state.vertices is not None:
                snap_active = st.checkbox(
                    "⚙️ Ajustar automáticamente al vértice STL más cercano (Snapping)",
                    value=False,
                    help="Ajusta la coordenada 3D al vértice más cercano del STL.",
                )

            # ── ROTACIÓN DEL OBJETO ───────────────────────────────
            st.markdown("#### ⚙️ Configuración de rotación del objeto")
            st.markdown("""
            <div class='warning-box'>
            Si rotaste físicamente el cilindro alrededor del eje Z entre fotos, ingresá el ángulo 
            de rotación acumulado. Esto ajusta automáticamente las coordenadas al sistema de referencia 
            fijo del laboratorio.
            </div>
            """, unsafe_allow_html=True)
            
            rotation_key = f"rotation_{image_name}"
            if rotation_key not in st.session_state.object_rotation_angles:
                st.session_state.object_rotation_angles[rotation_key] = 0.0
            
            rotation_angle = st.number_input(
                f"Rotación del objeto alrededor de Z (grados)",
                min_value=-360.0,
                max_value=360.0,
                value=st.session_state.object_rotation_angles[rotation_key],
                step=1.0,
                key=f"rotation_input_{image_name}",
                help="Ángulo positivo = rotación antihoraria vista desde +Z. Si rotaste el cilindro 45° antihorario, ingresá 45.",
            )
            st.session_state.object_rotation_angles[rotation_key] = rotation_angle

            # ── SELECTOR DE SISTEMA DE COORDENADAS ───────────────
            coord_mode_key = f"coord_mode_{image_name}"
            if coord_mode_key not in st.session_state:
                st.session_state[coord_mode_key] = "Cartesiano (X, Y, Z)"

            coord_mode = st.radio(
                "Sistema de coordenadas de entrada",
                ["Cartesiano (X, Y, Z)", "Cilíndrico (Z, R, φ)"],
                index=0 if st.session_state[coord_mode_key] == "Cartesiano (X, Y, Z)" else 1,
                horizontal=True,
                key=f"coord_radio_{image_name}",
                help="Cilíndrico: Z = altura sobre el plano base, R = radio desde el eje, φ = ángulo en grados. Se convierten a X,Y,Z automáticamente.",
            )
            st.session_state[coord_mode_key] = coord_mode

            if coord_mode == "Cilíndrico (Z, R, φ)":
                st.markdown("""
                <div class='info-box'>
                <b>Conversión cilíndrica → cartesiana:</b><br>
                X = R · cos(φ) &nbsp;&nbsp; Y = R · sin(φ) &nbsp;&nbsp; Z = Z<br>
                El eje Z es el eje de rotación del objeto. φ se ingresa en <b>grados</b>.
                Los valores X,Y,Z resultantes se actualizan automáticamente.
                </div>
                """, unsafe_allow_html=True)

            # ── FIX DATA_EDITOR: key estable por imagen + cantidad de puntos + modo coord
            mode_tag = "cyl" if coord_mode == "Cilíndrico (Z, R, φ)" else "cart"
            if coord_mode == "Cartesiano (X, Y, Z)":
                col_labels = [f"X [{units}]", f"Y [{units}]", f"Z [{units}]"]
                
                with st.form(key=f"form_cart_{image_name}"):
                    st.write("Editá las coordenadas y luego presioná **Guardar Coordenadas**:")
                    hcols = st.columns([1, 1, 1, 2, 2, 2])
                    hcols[0].markdown("**Pto**")
                    hcols[1].markdown("**Pix U**")
                    hcols[2].markdown("**Pix V**")
                    hcols[3].markdown(f"**{col_labels[0]}**")
                    hcols[4].markdown(f"**{col_labels[1]}**")
                    hcols[5].markdown(f"**{col_labels[2]}**")
                    
                    new_coords = []
                    for i, pt in enumerate(points):
                        cols = st.columns([1, 1, 1, 2, 2, 2])
                        cols[0].write(f"{i + 1}")
                        cols[1].write(f"{int(pt['u'])}")
                        cols[2].write(f"{int(pt['v'])}")
                        
                        raw_x = cols[3].number_input("X", value=float(pt["X"]), step=0.001, format="%.4f", key=f"x_{image_name}_{i}", label_visibility="collapsed")
                        raw_y = cols[4].number_input("Y", value=float(pt["Y"]), step=0.001, format="%.4f", key=f"y_{image_name}_{i}", label_visibility="collapsed")
                        raw_z = cols[5].number_input("Z", value=float(pt["Z"]), step=0.001, format="%.4f", key=f"z_{image_name}_{i}", label_visibility="collapsed")
                        new_coords.append((raw_x, raw_y, raw_z))
                        
                    submitted = st.form_submit_button("💾 Guardar Coordenadas", use_container_width=True)
                    if submitted:
                        for idx in range(len(points)):
                            rx, ry, rz = new_coords[idx]
                            if snap_active and st.session_state.vertices is not None:
                                P_snap = snap_to_closest_vertex(np.array([rx, ry, rz]), st.session_state.vertices)
                                points[idx]["X"] = float(P_snap[0])
                                points[idx]["Y"] = float(P_snap[1])
                                points[idx]["Z"] = float(P_snap[2])
                            else:
                                points[idx]["X"] = rx
                                points[idx]["Y"] = ry
                                points[idx]["Z"] = rz
                        st.rerun()

            else:
                # ── MODO CILÍNDRICO ───────────────────────────────
                cyl_col_labels = [f"Z [{units}]", f"R [{units}]", "φ [°]"]

                with st.form(key=f"form_cyl_{image_name}"):
                    st.write("Editá las coordenadas y luego presioná **Guardar Coordenadas**:")
                    hcols = st.columns([1, 1, 1, 2, 2, 2])
                    hcols[0].markdown("**Pto**")
                    hcols[1].markdown("**Pix U**")
                    hcols[2].markdown("**Pix V**")
                    hcols[3].markdown(f"**{cyl_col_labels[0]}**")
                    hcols[4].markdown(f"**{cyl_col_labels[1]}**")
                    hcols[5].markdown(f"**{cyl_col_labels[2]}**")
                    
                    new_coords = []
                    for i, pt in enumerate(points):
                        cols = st.columns([1, 1, 1, 2, 2, 2])
                        cols[0].write(f"{i + 1}")
                        cols[1].write(f"{int(pt['u'])}")
                        cols[2].write(f"{int(pt['v'])}")
                        
                        r_z = cols[3].number_input("Z", value=float(pt["Z"]), step=0.001, format="%.4f", key=f"cz_{image_name}_{i}", label_visibility="collapsed")
                        
                        r_r = float(np.sqrt(pt["X"]**2 + pt["Y"]**2))
                        r_r_val = cols[4].number_input("R", value=r_r, step=0.001, format="%.4f", min_value=0.0, key=f"cr_{image_name}_{i}", label_visibility="collapsed")
                        
                        r_phi = float(np.degrees(np.arctan2(pt["Y"], pt["X"])))
                        r_phi_val = cols[5].number_input("phi", value=r_phi, step=1.0, format="%.2f", key=f"cphi_{image_name}_{i}", label_visibility="collapsed")
                        
                        new_coords.append((r_z, r_r_val, r_phi_val))
                        
                    submitted = st.form_submit_button("💾 Guardar Coordenadas", use_container_width=True)
                    if submitted:
                        for idx in range(len(points)):
                            Z_cyl, R_cyl, phi_deg = new_coords[idx]
                            phi_rad = np.radians(phi_deg)
                            raw_x = R_cyl * np.cos(phi_rad)
                            raw_y = R_cyl * np.sin(phi_rad)
                            raw_z = Z_cyl
                            if snap_active and st.session_state.vertices is not None:
                                P_snap = snap_to_closest_vertex(np.array([raw_x, raw_y, raw_z]), st.session_state.vertices)
                                points[idx]["X"] = float(P_snap[0])
                                points[idx]["Y"] = float(P_snap[1])
                                points[idx]["Z"] = float(P_snap[2])
                            else:
                                points[idx]["X"] = raw_x
                                points[idx]["Y"] = raw_y
                                points[idx]["Z"] = raw_z
                        st.rerun()

                # Preview de la conversión
                with st.expander("👁️ Ver coordenadas cartesianas resultantes (X, Y, Z)"):
                    preview_rows = [
                        {"Pto": i+1,
                         f"X [{units}]": round(pt["X"], 4),
                         f"Y [{units}]": round(pt["Y"], 4),
                         f"Z [{units}]": round(pt["Z"], 4)}
                        for i, pt in enumerate(points)
                    ]
                    st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)

            # ── ASIGNACIÓN MASIVA DE COORDENADAS ───────────────
            with st.expander("📝 Edición en lote (Asignación masiva por puntos)"):
                with st.form(key=f"bulk_form_{image_name}"):
                    b_cols = st.columns([2, 2, 2])
                    target_pts = b_cols[0].text_input("Puntos a editar (ej: 1, 3, 4-6)", help="Separá por comas y usa guiones para rangos.")
                    
                    if coord_mode == "Cartesiano (X, Y, Z)":
                        coord_to_edit = b_cols[1].selectbox("Coordenada", ["X", "Y", "Z"])
                    else:
                        coord_to_edit = b_cols[1].selectbox("Coordenada", ["Z", "R", "phi"])
                        
                    new_val = b_cols[2].number_input("Nuevo valor", format="%.4f")
                    
                    bulk_submit = st.form_submit_button("Aplicar a los puntos", use_container_width=True)
                    if bulk_submit and target_pts.strip():
                        pts_to_update = set()
                        for part in target_pts.split(","):
                            part = part.strip()
                            if "-" in part:
                                try:
                                    start_s, end_s = part.split("-")
                                    pts_to_update.update(range(int(start_s), int(end_s)+1))
                                except:
                                    pass
                            elif part.isdigit():
                                pts_to_update.add(int(part))
                                
                        updated_count = 0
                        for pt_idx in pts_to_update:
                            idx = pt_idx - 1
                            if 0 <= idx < len(points):
                                if coord_mode == "Cartesiano (X, Y, Z)":
                                    points[idx][coord_to_edit] = float(new_val)
                                else:
                                    Z_cyl = points[idx]["Z"]
                                    R_cyl = float(np.sqrt(points[idx]["X"]**2 + points[idx]["Y"]**2))
                                    phi_deg = float(np.degrees(np.arctan2(points[idx]["Y"], points[idx]["X"])))
                                    
                                    if coord_to_edit == "Z": Z_cyl = float(new_val)
                                    elif coord_to_edit == "R": R_cyl = float(new_val)
                                    elif coord_to_edit == "phi": phi_deg = float(new_val)
                                    
                                    phi_rad = np.radians(phi_deg)
                                    points[idx]["X"] = float(R_cyl * np.cos(phi_rad))
                                    points[idx]["Y"] = float(R_cyl * np.sin(phi_rad))
                                    points[idx]["Z"] = float(Z_cyl)
                                updated_count += 1
                                
                        if updated_count > 0:
                            st.rerun()

            # Alerta de colinealidad si hay >= 4 puntos y K disponible
            if len(points) >= 4 and st.session_state.K_matrix is not None:
                obj_pts = np.array([[p["X"], p["Y"], p["Z"]] for p in points], dtype=np.float64)
                # Chequeo básico: los puntos 3D no deben ser coplanares en exceso
                centroid = obj_pts.mean(axis=0)
                diffs = obj_pts - centroid
                _, s, _ = np.linalg.svd(diffs)
                if s[-1] < 1e-6:
                    st.warning("⚠️ Los puntos 3D son coplanares. Agregá puntos en distintas profundidades para mejor estabilidad numérica.")
                else:
                    st.success(f"✅ {len(points)} puntos listos para SolvePnP.")

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  SECCIÓN 3 — CALIBRACIÓN DE CÁMARAS
# ══════════════════════════════════════════
elif page == "🔭 3. Calibración de Cámaras":
    if st.session_state.vertices is None:
        st.warning("⚠️ Cargá un archivo STL en la sección 1.")
    elif not st.session_state.images:
        st.warning("⚠️ Subí fotos en la sección 1.")
    else:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🔭 Estimación de Pose de Cámaras (SolvePnP)")

        # Aviso sobre la K
        if st.session_state.K_source == "chessboard":
            st.success(f"✅ Usando K calibrada con tablero (RMS={st.session_state.calibration_rms:.4f}px). Las imágenes serán corregidas con undistort antes de la proyección.")
        elif st.session_state.K_source == "manual":
            st.info("ℹ️ Usando K ingresada manualmente.")
        else:
            st.warning("⚠️ Usando K estimada. Para resultados óptimos, calibrá con tablero en la sección 0.")

        ready_images = [
            img["name"] for img in st.session_state.images
            if len(st.session_state.points_data.get(img["name"], [])) >= 4
        ]

        if not ready_images:
            st.info("💡 Necesitás al menos **4 puntos con coordenadas 3D** por imagen para calibrar.")
        else:
            st.write(f"Imágenes listas: {', '.join([f'`{n}`' for n in ready_images])}")

            col_profiles, col_assignments = st.columns(2)
            with col_profiles:
                st.markdown("##### Perfiles de Cámara (factor zoom estimado)")
                updated_profiles = {}
                for cam_name, current_val in list(st.session_state.camera_profiles.items()):
                    val = st.slider(f"Zoom — {cam_name}", 0.5, 4.0, float(current_val), 0.05, key=f"prof_{cam_name}")
                    updated_profiles[cam_name] = val
                st.session_state.camera_profiles = updated_profiles

                with st.expander("➕ Nuevo perfil"):
                    new_name = st.text_input("Nombre", placeholder="Ej: Cámara Macro", key="new_cam_name")
                    new_val = st.slider("Zoom inicial", 0.5, 4.0, 1.5, 0.05, key="new_cam_val2")
                    if st.button("Añadir Cámara"):
                        if new_name.strip() and new_name.strip() not in st.session_state.camera_profiles:
                            st.session_state.camera_profiles[new_name.strip()] = new_val
                            st.success(f"Cámara '{new_name}' creada.")

            with col_assignments:
                st.markdown("##### Asignar cámara a cada foto")
                available_cams = list(st.session_state.camera_profiles.keys())
                for img_name in ready_images:
                    current = st.session_state.image_camera_assignments.get(img_name, available_cams[0])
                    if current not in available_cams:
                        current = available_cams[0]
                    sel = st.selectbox(f"`{img_name}`", available_cams, index=available_cams.index(current), key=f"assign_{img_name}")
                    st.session_state.image_camera_assignments[img_name] = sel

            if st.button("🚀 Calibrar poses de cámaras", use_container_width=True):
                with st.spinner("Resolviendo SolvePnP..."):
                    ok_count = 0
                    for img_name in ready_images:
                        pts = st.session_state.points_data[img_name]
                        img_dict = next(i for i in st.session_state.images if i["name"] == img_name)
                        img_arr = img_dict["np"]
                        H, W, _ = img_arr.shape

                        obj_pts = np.array([[p["X"], p["Y"], p["Z"]] for p in pts], dtype=np.float64)
                        img_pts = np.array([[p["u"], p["v"]] for p in pts], dtype=np.float64)

                        # Usar K calibrada si está disponible, sino estimar
                        if st.session_state.K_matrix is not None:
                            K = st.session_state.K_matrix
                        else:
                            assigned = st.session_state.image_camera_assignments.get(img_name, available_cams[0])
                            focal_factor = st.session_state.camera_profiles.get(assigned, 1.2)
                            K = estimate_camera_matrix(W, H, focal_factor)

                        rvec, tvec, C, pitch, roll, yaw, success = calibrate_camera(obj_pts, img_pts, K)
                        if success:
                            st.session_state.calibrations[img_name] = {
                                "rvec": rvec, "tvec": tvec, "K": K, "C": C,
                                "pitch": pitch, "roll": roll, "yaw": yaw,
                            }
                            ok_count += 1
                        else:
                            st.error(f"Fallo al calibrar `{img_name}`.")

                    if ok_count:
                        st.success(f"🎉 {ok_count} cámara(s) calibradas correctamente.")
                        if "face_colors" in st.session_state:
                            del st.session_state.face_colors

            # Tabla de resultados
            if st.session_state.calibrations:
                st.markdown("#### 📊 Parámetros resueltos")
                units = st.session_state.stl_units
                rows = []
                for name, cal in st.session_state.calibrations.items():
                    C = cal["C"]
                    rows.append({
                        "Archivo": name,
                        f"X [{units}]": round(C[0], 3),
                        f"Y [{units}]": round(C[1], 3),
                        f"Z [{units}]": round(C[2], 3),
                        "Pitch": f"{cal['pitch']:.1f}°",
                        "Roll": f"{cal['roll']:.1f}°",
                        "Yaw": f"{cal['yaw']:.1f}°",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Visor 3D con cámaras
                st.markdown("#### 🌐 Entorno 3D reconstruido")
                v = st.session_state.vertices
                f_arr = st.session_state.faces
                mesh_span = np.linalg.norm(v.max(axis=0) - v.min(axis=0))
                cam_scale = mesh_span * 0.12

                traces = [go.Mesh3d(
                    x=v[:,0], y=v[:,1], z=v[:,2],
                    i=f_arr[:,0], j=f_arr[:,1], k=f_arr[:,2],
                    color='#64748b', opacity=0.55, flatshading=False,
                    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.1)
                )]
                for name, cal in st.session_state.calibrations.items():
                    xs, ys, zs = get_camera_wireframe(cal["C"], cal["rvec"], cal["tvec"], cam_scale)
                    traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                                               line=dict(color='#8b5cf6', width=4), name=f"Cam: {name}"))
                    traces.append(go.Scatter3d(x=[cal["C"][0]], y=[cal["C"][1]], z=[cal["C"][2]],
                                               mode='markers+text', text=[name], textposition="top center",
                                               marker=dict(color='#f43f5e', size=8), name=f"Foco: {name}"))

                fig = go.Figure(data=traces)
                fig.update_layout(
                    scene=dict(xaxis_title=f"X [{units}]", yaxis_title=f"Y [{units}]",
                               zaxis_title=f"Z [{units}]", bgcolor="#0f172a", aspectmode="data"),
                    margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="rgba(0,0,0,0)", height=550
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  SECCIÓN 4 — TEXTURIZADO FINAL
# ══════════════════════════════════════════
elif page == "🎨 4. Texturizado Final":
    if st.session_state.vertices is None:
        st.warning("⚠️ Cargá el archivo STL en la sección 1.")
    elif not st.session_state.calibrations:
        st.warning("⚠️ Calibrá al menos una cámara en la sección 3.")
    else:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🎨 Proyección de Texturas — Backward Projection")
        st.markdown("""
        <div class='info-box'>
        El algoritmo evalúa el <b>producto punto entre la normal de cada cara STL y el vector de vista</b>
        de cada cámara. Solo se proyecta textura en superficies visibles (dot product &gt; 0),
        descartando caras que apuntan "de espaldas" a la cámara.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🎨 Generar Modelo Texturizado Completo", use_container_width=True):
            with st.spinner("Proyección inversa vectorizada..."):
                try:
                    v = st.session_state.vertices
                    f_arr = st.session_state.faces

                    # Aplicar undistort a imágenes si tenemos K calibrada
                    images_dict = {}
                    for img in st.session_state.images:
                        arr = img["np"].copy()
                        if (st.session_state.K_matrix is not None and
                                st.session_state.dist_coeffs is not None and
                                st.session_state.K_source != "estimada"):
                            arr = cv2.undistort(arr, st.session_state.K_matrix, st.session_state.dist_coeffs)
                        images_dict[img["name"]] = arr

                    colors = backward_projection(v, f_arr, images_dict, st.session_state.calibrations)
                    st.session_state.face_colors = colors
                    st.success("✅ Texturizado completado.")
                except Exception as e:
                    st.error(f"Error en proyección inversa: {e}")

        if "face_colors" in st.session_state:
            st.markdown("#### 🌐 Modelo STL Texturizado")
            v = st.session_state.vertices
            f_arr = st.session_state.faces
            units = st.session_state.stl_units
            fig = go.Figure(data=[go.Mesh3d(
                x=v[:,0], y=v[:,1], z=v[:,2],
                i=f_arr[:,0], j=f_arr[:,1], k=f_arr[:,2],
                facecolor=st.session_state.face_colors,
                opacity=1.0, flatshading=False,
                lighting=dict(ambient=0.55, diffuse=0.6, specular=0.1, roughness=0.8)
            )])
            fig.update_layout(
                scene=dict(xaxis_title=f"X [{units}]", yaxis_title=f"Y [{units}]",
                           zaxis_title=f"Z [{units}]", bgcolor="#0f172a", aspectmode="data"),
                margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="rgba(0,0,0,0)", height=650
            )
            st.plotly_chart(fig, use_container_width=True)
            st.balloons()

        st.markdown("</div>", unsafe_allow_html=True)