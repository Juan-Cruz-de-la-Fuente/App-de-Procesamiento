import sqlite3
import hashlib
import os

DB_NAME = os.path.join(os.path.dirname(__file__), '..', 'identificaciones_api', 'users.db')

def init_db():
    # Asegurar que el directorio de la base de datos exista (necesario en la nube)
    os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)

    from codigo_fuente import Drive_Connection as drive_api
    # Intentar descargar users.db desde Drive si existe en la carpeta raíz
    db_file_id = None
    files = drive_api.list_files(drive_api.ROOT_FOLDER_ID)
    for f in files:
        if f.get('name') == 'users.db':
            db_file_id = f['id']
            break

    if db_file_id:
        try:
            db_data = drive_api.download_file(db_file_id)
            with open(DB_NAME, 'wb') as f:
                f.write(db_data)
        except Exception as e:
            print("No se pudo descargar data DB de Drive. Usando el local si existe: ", e)

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        ret = True
    except sqlite3.IntegrityError:
        ret = False
    finally:
        conn.close()

    if ret:
        from codigo_fuente import Drive_Connection as drive_api
        try:
            with open(DB_NAME, 'rb') as f:
                db_data = f.read()
            drive_api.upload_file(db_data, 'users.db', drive_api.ROOT_FOLDER_ID, mimetype='application/x-sqlite3')
        except Exception as e:
            print("Error subiendo DB a Drive:", e)

    return ret

def verify_user(username, password):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        # Seleccionamos solo el password para comparar en Python
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        conn.close()
        
        if row:
            db_password = row[0]
            # Verificamos si la contraseña coincide
            hashed_input = hashlib.sha256(password.encode()).hexdigest()
            return db_password == hashed_input
        return False
    except Exception as e:
        import streamlit as st
        st.error(f"Error técnico en el login: {e}")
        return False

def check_db_status():
    """Devuelve True si la tabla de usuarios está vacía."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users")
        count = c.fetchone()[0]
        conn.close()
        return count == 0
    except Exception:
        return True

# Initialize DB on module load
init_db()

# --- DATA STORAGE FUNCTIONS VIA GOOGLE DRIVE ---
from codigo_fuente import Drive_Connection as drive_api
import json


def init_data_table():
    pass  # No longer uses SQLite for data


# ---------------------------------------------------------------------------
# SUPERFICIES  →  usuario / ENSAYO DE ESTELA / 3D /
# ---------------------------------------------------------------------------

def save_surface_data(username, filename, data_json):
    """Guarda una superficie 3D (JSON) en Drive → usuario/ENSAYO DE ESTELA/3D/.
    Formato: 3D____nombre.json  (sin posición X — el plano es independiente del espacio)
    """
    folder_id = drive_api.get_folder_3d(username)
    if not folder_id:
        return False

    safe_filename = str(filename).replace("/", "-").replace("____", "_")
    drive_filename = f"3D____{safe_filename}.json"

    file_id = drive_api.upload_file(data_json, drive_filename, folder_id)
    return file_id is not None


def save_surface_data_4d(username, filename, x_position, data_json):
    """Guarda un plano 4D (JSON) en Drive → usuario/ENSAYO DE ESTELA/4D/.
    Formato: 4D____nombre____xpos.json  (incluye posición X en el espacio)
    """
    folder_id = drive_api.get_folder_4d(username)
    if not folder_id:
        return False

    safe_filename = str(filename).replace("/", "-").replace("____", "_")
    drive_filename = f"4D____{safe_filename}____{x_position}.json"

    file_id = drive_api.upload_file(data_json, drive_filename, folder_id)
    return file_id is not None


def get_user_surfaces(username):
    """Devuelve lista de (id, filename, x_position, created_at, data_json) de la carpeta 3D.
    Compatible con archivos viejos (SURF____) y nuevos (3D____).
    """
    import streamlit as st
    folder_id = drive_api.get_folder_3d(username)
    if not folder_id:
        return []

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_all_surfaces(fid):
        all_res = []
        fs = drive_api.list_files(fid)
        for f in fs:
            name = f.get('name', '')
            # Nuevo formato: 3D____nombre.json
            # Nuevo formato: 3D____nombre.json
            if name.startswith('3D____'):
                parts = name.replace('.json', '').split('____')
                fname = parts[1] if len(parts) >= 2 else name
                all_res.append((f['id'], fname, 0.0, f.get('createdTime'), None))
            # Formato viejo: SURF____nombre.json (3D viejo sin xpos)
            elif name.startswith('SURF____'):
                parts = name.replace('.json', '').split('____')
                if len(parts) < 3: # Si es menor a 3, es un archivo 3D (no tiene Xpos)
                    fname = parts[1] if len(parts) >= 2 else name.replace('.json', '')
                    all_res.append((f['id'], fname, 0.0, f.get('createdTime'), None))
        return sorted(all_res, key=lambda x: x[3] or '', reverse=True)

    return fetch_all_surfaces(folder_id)


def get_user_surfaces_4d(username):
    """Devuelve lista de (id, filename, x_position, created_at, data_json) de la carpeta 4D."""
    import streamlit as st
    folder_id = drive_api.get_folder_4d(username)
    if not folder_id:
        return []

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_all_4d(fid):
        all_res = []
        fs = drive_api.list_files(fid)
        for f in fs:
            name = f.get('name', '')
            if name.startswith('4D____'):
                parts = name.replace('.json', '').split('____')
                if len(parts) >= 3:
                    fname = parts[1]
                    try:
                        x_pos = float(parts[2])
                    except Exception:
                        x_pos = 0.0
                    all_res.append((f['id'], fname, x_pos, f.get('createdTime'), None))
            elif name.startswith('SURF____'):
                parts = name.replace('.json', '').split('____')
                if len(parts) >= 3: # Si tiene 3 partes o más, es 4D (tiene Xpos)
                    fname = parts[1]
                    try:
                        x_pos = float(parts[2])
                    except Exception:
                        x_pos = 0.0
                    all_res.append((f['id'], fname, x_pos, f.get('createdTime'), None))
        return sorted(all_res, key=lambda x: x[2])  # ordenado por posición X

    return fetch_all_4d(folder_id)


def get_user_files_2d(username):
    """Lista archivos CSV de la carpeta 2D (guardados desde BETZ 2D).
    Devuelve lista de (id, name, created_at, data_bytes).
    """
    import streamlit as st
    folder_id = drive_api.get_folder_2d(username)
    if not folder_id:
        return []

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_all_2d(fid):
        all_res = []
        fs = drive_api.list_files(fid)
        for f in fs:
            name = f.get('name', '')
            if name.endswith('.csv'):
                all_res.append((f['id'], name, f.get('createdTime'), None))
        return sorted(all_res, key=lambda x: x[2] or '', reverse=True)

    return fetch_all_2d(folder_id)


def download_file_2d(file_id):
    """Descarga el contenido de un archivo 2D de Drive como bytes."""
    return drive_api.download_file(file_id)

def get_surface_data_string(file_id):
    """Descarga on-demand el string JSON de una superficie 3D o 4D."""
    data_bytes = drive_api.download_file(file_id)
    return data_bytes.decode('utf-8') if data_bytes else '{}'



def delete_user_surface(surface_id):
    drive_api.delete_file(surface_id)


def rename_user_surface(surface_id, new_filename):
    # El nombre en Drive incluye metadata codificada en el prefijo;
    # el renombrado simple ya funciona via drive_api.rename_file (cambia el nombre visible)
    drive_api.rename_file(surface_id, new_filename)


# ---------------------------------------------------------------------------
# OBJETOS 3D  →  usuario / MODELOS 3D /
# ---------------------------------------------------------------------------

def save_user_object(username, name, obj_type, data_json):
    """Guarda un objeto 3D (JSON) en Drive → usuario/MODELOS 3D/."""
    folder_id = drive_api.get_folder_modelos(username)
    if not folder_id:
        return False

    safe_name = str(name).replace("/", "-").replace("____", "_")
    drive_filename = f"OBJ____{safe_name}____{obj_type}.json"

    file_id = drive_api.upload_file(data_json, drive_filename, folder_id)
    return file_id is not None


def get_user_objects(username):
    """Devuelve lista de (id, name, obj_type, data_json, created_at)."""
    import streamlit as st
    folder_id = drive_api.get_folder_modelos(username)
    if not folder_id:
        return []

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_all_objs(fid):
        all_res = []
        fs = drive_api.list_files(fid)
        for f in fs:
            name_d = f.get('name', '')
            if name_d.startswith('OBJ____'):
                parts = name_d.replace('.json', '').split('____')
                if len(parts) >= 3:
                    obj_name = parts[1]
                    obj_type = parts[2]
                    data_bytes = drive_api.download_file(f['id'])
                    data_str = data_bytes.decode('utf-8') if data_bytes else "{}"
                    all_res.append((f['id'], obj_name, obj_type, data_str, f.get('createdTime')))
        return sorted(all_res, key=lambda x: x[4], reverse=True)

    return fetch_all_objs(folder_id)


def delete_user_object(object_id):
    drive_api.delete_file(object_id)


def rename_user_object(object_id, new_name):
    pass


# ---------------------------------------------------------------------------
# ANIMACION  →  usuario / ENSAYO DE ESTELA / ANIMACION / [nombre_sesion] /
# ---------------------------------------------------------------------------

def save_animation_session(username, session_name, frames_bytes_list, metadata_dict):
    """
    Guarda una sesión de animación en Drive.

    Estructura:
      ANIMACION / session_name /
          frame_000.png
          frame_001.png
          ...
          _metadata.json   ← planos base, AOAs, variable, fecha, etc.

    Parameters
    ----------
    username       : str
    session_name   : str  — nombre de la carpeta a crear
    frames_bytes_list : list[bytes]  — frames PNG en orden
    metadata_dict  : dict  — info de la interpolación (aoas, x_pos, variable, planos usados, etc.)

    Returns
    -------
    session_folder_id : str | None
    """
    session_folder_id = drive_api.get_or_create_session_folder(username, session_name)
    if not session_folder_id:
        return None

    # Subir metadata JSON primero
    meta_str = json.dumps(metadata_dict, ensure_ascii=False, indent=2)
    drive_api.upload_file(meta_str.encode('utf-8'), '_metadata.json',
                          session_folder_id, mimetype='application/json')

    # Subir frames PNG
    for idx, frame_bytes in enumerate(frames_bytes_list):
        fname = f"frame_{idx:04d}.png"
        drive_api.upload_file(frame_bytes, fname, session_folder_id, mimetype='image/png')

    return session_folder_id


def list_animation_sessions(username):
    """
    Lista las sesiones de animación guardadas del usuario.
    Devuelve lista de (folder_id, session_name, created_at).
    """
    return drive_api.list_animation_sessions(username)


def load_animation_session(session_folder_id):
    """
    Carga todos los frames y la metadata de una sesión.

    Returns
    -------
    frames : list[bytes]   — PNGs ordenados por nombre
    metadata : dict        — contenido de _metadata.json (o {} si no existe)
    """
    files = drive_api.list_session_frames(session_folder_id)

    # Separar metadata de frames
    meta_file = next((f for f in files if f['name'] == '_metadata.json'), None)
    frame_files = sorted(
        [f for f in files if f['name'].startswith('frame_') and f['name'].endswith('.png')],
        key=lambda f: f['name']
    )

    metadata = {}
    if meta_file:
        raw = drive_api.download_file(meta_file['id'])
        if raw:
            try:
                metadata = json.loads(raw.decode('utf-8'))
            except Exception:
                metadata = {}

    frames = []
    for f in frame_files:
        raw = drive_api.download_file(f['id'])
        if raw:
            frames.append(raw)

    return frames, metadata


# ---------------------------------------------------------------------------
# VTK  →  usuario / HERRAMIENTAS / ARCHIVOS VTK / PLANOS DE PRESION  o  SUPERFICIES 3D
# ---------------------------------------------------------------------------

def save_vtk_plano(username, filename, vtk_bytes):
    """Sube un archivo VTK de plano 2D a Drive → HERRAMIENTAS/ARCHIVOS VTK/PLANOS DE PRESION/."""
    folder_id = drive_api.get_folder_vtk_planos(username)
    if not folder_id:
        return False
    file_id = drive_api.upload_file(vtk_bytes, filename, folder_id, mimetype='application/octet-stream')
    return file_id is not None


def save_csv_1d(username, filename, csv_bytes):
    """Sube un sub-archivo CSV de 1D a Drive → usuario/ENSAYO DE ESTELA/1D/."""
    uid    = drive_api.get_user_root(username)
    eid    = drive_api.get_or_create_folder(drive_api.FOLDER_ESTELA, uid)
    fid    = drive_api.get_or_create_folder(drive_api.FOLDER_1D, eid)
    if not fid:
        return False
    file_id = drive_api.upload_file(csv_bytes, filename, fid, mimetype='text/csv')
    return file_id is not None


def save_csv_2d(username, filename, csv_bytes):
    """Sube un archivo CSV de matriz 2D a Drive → usuario/ENSAYO DE ESTELA/2D/."""
    uid    = drive_api.get_user_root(username)
    eid    = drive_api.get_or_create_folder(drive_api.FOLDER_ESTELA, uid)
    fid    = drive_api.get_or_create_folder(drive_api.FOLDER_2D, eid)
    if not fid:
        return False
    file_id = drive_api.upload_file(csv_bytes, filename, fid, mimetype='text/csv')
    return file_id is not None


def save_vtk_superficie(username, filename, vtk_bytes):
    """Sube un archivo VTK de superficie 3D a Drive → HERRAMIENTAS/ARCHIVOS VTK/SUPERFICIES 3D/."""
    folder_id = drive_api.get_folder_vtk_superf(username)
    if not folder_id:
        return False
    file_id = drive_api.upload_file(vtk_bytes, filename, folder_id, mimetype='application/octet-stream')
    return file_id is not None
