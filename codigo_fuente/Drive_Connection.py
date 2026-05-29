import os
import io
import json
import streamlit as st
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive']
# ID de la carpeta raíz en Google Drive donde se guardará todo
ROOT_FOLDER_ID = '1mkmYGx0uNmWxxvHTKKrUfGUDZlqTfeOs'

# --- Nombres de carpetas (MAYÚSCULAS) ---
FOLDER_ESTELA        = 'ENSAYO DE ESTELA'
FOLDER_1D            = '1D'
FOLDER_2D            = '2D'
FOLDER_3D            = '3D'
FOLDER_4D            = '4D'
FOLDER_ANIMACION     = 'ANIMACION'
FOLDER_HERRAMIENTAS  = 'HERRAMIENTAS'
FOLDER_VTK           = 'ARCHIVOS VTK'
FOLDER_VTK_PLANOS    = 'PLANOS DE PRESION'
FOLDER_VTK_SUPERF    = 'SUPERFICIES 3D'
FOLDER_MODELOS       = 'MODELOS 3D'


def get_service():
    creds = None
    token_path = os.path.join(os.path.dirname(__file__), '..', 'identificaciones_api', 'OAuth_Token.json')
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    else:
        try:
            import streamlit as st
            if "gcp_service_account" in st.secrets:
                secret_dict = dict(st.secrets["gcp_service_account"])
                creds = Credentials.from_authorized_user_info(secret_dict, SCOPES)
            elif "google_token_json" in st.secrets:
                try:
                    import json
                    token_str = st.secrets["google_token_json"]
                    secret_dict = json.loads(token_str)
                    creds = Credentials.from_authorized_user_info(secret_dict, SCOPES)
                except Exception as json_e:
                    print("Error interpretando el JSON pegado en secrets:", json_e)
        except Exception as st_e:
            print(f"Error cargando secretos: {st_e}")

    if not creds:
        print("No se encontraron credenciales de Google Drive (token.json). Ningun Token configurado.")
        return None

    return build('drive', 'v3', credentials=creds)


def get_or_create_folder(folder_name, parent_id=ROOT_FOLDER_ID):
    """Busca una carpeta por nombre. Si no existe, la crea y devuelve su ID."""
    service = get_service()
    if not service:
        return None

    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
    results = service.files().list(q=query, spaces='drive', supportsAllDrives=True,
                                   includeItemsFromAllDrives=True, fields='files(id, name)').execute()
    items = results.get('files', [])

    if not items:
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        folder = service.files().create(body=file_metadata, fields='id', supportsAllDrives=True).execute()
        return folder.get('id')
    return items[0].get('id')


def init_user_folders(username):
    """
    Crea toda la jerarquía de carpetas para un usuario si no existen.
    Estructura:
      root / username /
          ENSAYO DE ESTELA / 1D, 2D, 3D, 4D
          HERRAMIENTAS / ARCHIVOS VTK / PLANOS DE PRESION
                                      / SUPERFICIES 3D
          MODELOS 3D
    """
    user_id       = get_or_create_folder(username, ROOT_FOLDER_ID)
    if not user_id:
        return {}

    estela_id     = get_or_create_folder(FOLDER_ESTELA, user_id)
    herr_id       = get_or_create_folder(FOLDER_HERRAMIENTAS, user_id)
    modelos_id    = get_or_create_folder(FOLDER_MODELOS, user_id)

    folder_1d     = get_or_create_folder(FOLDER_1D, estela_id)
    folder_2d     = get_or_create_folder(FOLDER_2D, estela_id)
    folder_3d     = get_or_create_folder(FOLDER_3D, estela_id)
    folder_4d     = get_or_create_folder(FOLDER_4D, estela_id)
    folder_anim   = get_or_create_folder(FOLDER_ANIMACION, estela_id)

    vtk_id        = get_or_create_folder(FOLDER_VTK, herr_id)
    vtk_planos_id = get_or_create_folder(FOLDER_VTK_PLANOS, vtk_id)
    vtk_superf_id = get_or_create_folder(FOLDER_VTK_SUPERF, vtk_id)

    return {
        'user':         user_id,
        'estela':       estela_id,
        '1d':           folder_1d,
        '2d':           folder_2d,
        '3d':           folder_3d,
        '4d':           folder_4d,
        'animacion':    folder_anim,
        'herr':         herr_id,
        'vtk':          vtk_id,
        'vtk_planos':   vtk_planos_id,
        'vtk_superf':   vtk_superf_id,
        'modelos':      modelos_id,
    }


# --- Helpers de acceso rápido a carpetas específicas ---

def get_user_root(username):
    return get_or_create_folder(username, ROOT_FOLDER_ID)

def get_folder_3d(username):
    uid = get_user_root(username)
    eid = get_or_create_folder(FOLDER_ESTELA, uid)
    return get_or_create_folder(FOLDER_3D, eid)

def get_folder_2d(username):
    uid = get_user_root(username)
    eid = get_or_create_folder(FOLDER_ESTELA, uid)
    return get_or_create_folder(FOLDER_2D, eid)

def get_folder_modelos(username):
    uid = get_user_root(username)
    return get_or_create_folder(FOLDER_MODELOS, uid)

def get_folder_4d(username):
    uid = get_user_root(username)
    eid = get_or_create_folder(FOLDER_ESTELA, uid)
    return get_or_create_folder(FOLDER_4D, eid)

def get_folder_animacion(username):
    """Devuelve (creando si no existe) la carpeta ANIMACION del usuario.
    Ruta: root / username / ENSAYO DE ESTELA / ANIMACION
    """
    uid = get_user_root(username)
    eid = get_or_create_folder(FOLDER_ESTELA, uid)
    return get_or_create_folder(FOLDER_ANIMACION, eid)

def get_or_create_session_folder(username, session_name):
    """Devuelve (creando si no existe) la carpeta de una sesión de animación.
    Ruta: root / username / ENSAYO DE ESTELA / ANIMACION / session_name
    """
    anim_id = get_folder_animacion(username)
    if not anim_id:
        return None
    safe_name = session_name.strip().replace('/', '-').replace('\\', '-')
    return get_or_create_folder(safe_name, anim_id)

def list_animation_sessions(username):
    """Lista las sub-carpetas (sesiones) dentro de ANIMACION. Devuelve [(id, name)]."""
    anim_id = get_folder_animacion(username)
    if not anim_id:
        return []
    service = get_service()
    if not service:
        return []
    query = (f"'{anim_id}' in parents and "
             f"mimeType='application/vnd.google-apps.folder' and trashed=false")
    results = service.files().list(
        q=query, spaces='drive',
        supportsAllDrives=True, includeItemsFromAllDrives=True,
        fields='files(id, name, createdTime)', orderBy='name'
    ).execute()
    return [(f['id'], f['name'], f.get('createdTime','')) for f in results.get('files', [])]

def list_session_frames(session_folder_id):
    """Lista los archivos (frames PNG + metadata JSON) dentro de una carpeta de sesión."""
    service = get_service()
    if not service:
        return []
    query = f"'{session_folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query, spaces='drive',
        supportsAllDrives=True, includeItemsFromAllDrives=True,
        fields='files(id, name, createdTime)', orderBy='name'
    ).execute()
    return results.get('files', [])

def get_folder_vtk_planos(username):
    uid    = get_user_root(username)
    hid    = get_or_create_folder(FOLDER_HERRAMIENTAS, uid)
    vtk_id = get_or_create_folder(FOLDER_VTK, hid)
    return get_or_create_folder(FOLDER_VTK_PLANOS, vtk_id)

def get_folder_vtk_superf(username):
    uid    = get_user_root(username)
    hid    = get_or_create_folder(FOLDER_HERRAMIENTAS, uid)
    vtk_id = get_or_create_folder(FOLDER_VTK, hid)
    return get_or_create_folder(FOLDER_VTK_SUPERF, vtk_id)


# --- Operaciones de archivo ---

def upload_file(file_content, filename, parent_id, mimetype='application/json'):
    """Sube un archivo a Google Drive dentro de un parent_id específico."""
    service = get_service()
    if not service:
        return None

    file_metadata = {'name': filename, 'parents': [parent_id]}

    if isinstance(file_content, str):
        file_content_io = io.BytesIO(file_content.encode('utf-8'))
    else:
        file_content_io = io.BytesIO(file_content)

    media = MediaIoBaseUpload(file_content_io, mimetype=mimetype, resumable=True)

    # Sobrescribir si ya existe
    query = f"name='{filename}' and '{parent_id}' in parents and trashed=false"
    results = service.files().list(q=query, spaces='drive', supportsAllDrives=True,
                                   includeItemsFromAllDrives=True, fields='files(id, name)').execute()
    items = results.get('files', [])

    if items:
        file_id = items[0].get('id')
        service.files().update(fileId=file_id, media_body=media, supportsAllDrives=True).execute()
        return file_id
    else:
        created_file = service.files().create(body=file_metadata, media_body=media,
                                              fields='id', supportsAllDrives=True).execute()
        return created_file.get('id')


def list_files(parent_id):
    """Devuelve la lista de archivos (sin carpetas) dentro de una carpeta."""
    service = get_service()
    if not service:
        return []
    query = f"'{parent_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', supportsAllDrives=True,
                                   includeItemsFromAllDrives=True,
                                   fields='files(id, name, createdTime)').execute()
    return results.get('files', [])


def list_folder_contents(parent_id):
    """Devuelve carpetas y archivos dentro de parent_id, ordenado: carpetas primero."""
    service = get_service()
    if not service:
        return []
    query = f"'{parent_id}' in parents and trashed=false"
    results = service.files().list(
        q=query, spaces='drive',
        supportsAllDrives=True, includeItemsFromAllDrives=True,
        fields='files(id, name, mimeType, createdTime)',
        orderBy='folder,name'
    ).execute()
    return results.get('files', [])


def get_folder_info(folder_id):
    """Devuelve el nombre de una carpeta dado su ID (para breadcrumb)."""
    service = get_service()
    if not service:
        return None
    try:
        file_meta = service.files().get(
            fileId=folder_id, fields='id, name',
            supportsAllDrives=True
        ).execute()
        return file_meta
    except Exception as e:
        print(f"Error obteniendo info de carpeta: {e}")
        return None


def download_file(file_id):
    """Descarga un archivo dado su ID y lo devuelve como bytes."""
    service = get_service()
    if not service:
        return None
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    return fh.getvalue()


def delete_file(file_id):
    """Mueve un archivo a la papelera."""
    service = get_service()
    if not service:
        return False
    try:
        service.files().update(
            fileId=file_id,
            body={'trashed': True},
            supportsAllDrives=True
        ).execute()
        return True
    except Exception as e:
        print(f"Error borrando archivo: {e}")
        return False


def rename_file(file_id, new_name):
    """Renombra un archivo manteniendo su ID intacto."""
    service = get_service()
    if not service:
        return False
    try:
        service.files().update(
            fileId=file_id,
            body={'name': new_name},
            supportsAllDrives=True
        ).execute()
        return True
    except Exception as e:
        print(f"Error renombrando archivo: {e}")
        return False
