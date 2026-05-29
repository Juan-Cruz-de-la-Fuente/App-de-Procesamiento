import streamlit as st
import os
from datetime import datetime
from codigo_fuente import Auth_Manager as auth
from codigo_fuente import Drive_Connection as _dapi

def show_configuracion():
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(180deg, #000 0%, #111 100%); margin-bottom: 3rem;">
        <h1 style="font-size: 3.5rem; margin-bottom: 1rem; letter-spacing: 4px; color: #fff; text-transform: uppercase;">Configuración</h1>
        <p style="color: #666; font-size: 1.2rem; max-width: 600px; margin: 0 auto;">Estado del sistema y gestión de parámetros operativos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔌 Estado del Sistema")
    
    db_ok = os.path.exists("users.db")
    db_size = os.path.getsize("users.db") / 1024 if db_ok else 0
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="section-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 10px;">💾</div>
            <h4 style="margin:0; color:white;">Base de Datos</h4>
            <div style="font-size: 1.2rem; font-weight:bold; color: {'#4ade80' if db_ok else '#f87171'}; margin: 10px 0;">
                {' ONLINE' if db_ok else ' OFFLINE'}
            </div>
            <p style="color: grey; font-size: 0.8rem; margin:0;">users.db ({db_size:.1f} KB)</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="section-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 10px;">👤</div>
            <h4 style="margin:0; color:white;">Sesión Activa</h4>
            <div style="font-size: 1.2rem; font-weight:bold; color: #60a5fa; margin: 10px 0;">
                {st.session_state.username}
            </div>
            <p style="color: grey; font-size: 0.8rem; margin:0;">Privilegios: {'Admin' if st.session_state.username=='admin' else 'Usuario'}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="section-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🚀</div>
            <h4 style="margin:0; color:white;">Versión App</h4>
            <div style="font-size: 1.2rem; font-weight:bold; color: #a78bfa; margin: 10px 0;">
                v2.1.0 (Modular)
            </div>
            <p style="color: grey; font-size: 0.8rem; margin:0;">Build: {datetime.now().strftime('%Y.%m.%d')}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗃 Explorador de Archivos en Drive")
    st.info("Navegá entre carpetas, renombrá o eliminá archivos de tu cuenta en Google Drive.")

    if 'drive_folder_path' not in st.session_state:
        st.session_state.drive_folder_path = []
    if 'drive_current_folder_id' not in st.session_state:
        st.session_state.drive_current_folder_id = None
    if 'drive_rename_file_id' not in st.session_state:
        st.session_state.drive_rename_file_id = None
    if 'drive_confirm_delete_id' not in st.session_state:
        st.session_state.drive_confirm_delete_id = None

    if st.session_state.drive_current_folder_id is None:
        with st.spinner("Conectando con Google Drive..."):
            user_root_id = _dapi.get_user_root(st.session_state.username)
        if user_root_id:
            st.session_state.drive_current_folder_id = user_root_id
            st.session_state.drive_folder_path = [(user_root_id, f" {st.session_state.username}")]
        else:
            st.error("  No se pudo conectar con Google Drive. Verificá las credenciales.")
            user_root_id = None

    current_folder_id = st.session_state.drive_current_folder_id

    if current_folder_id:
        breadcrumb_cols = st.columns(len(st.session_state.drive_folder_path) * 2)
        for i, (fid, fname) in enumerate(st.session_state.drive_folder_path):
            with breadcrumb_cols[i * 2]:
                is_last = (i == len(st.session_state.drive_folder_path) - 1)
                if is_last:
                    st.markdown(f"<span style='color:#60a5fa; font-weight:bold;'>{fname}</span>", unsafe_allow_html=True)
                else:
                    if st.button(fname, key=f"bread_{fid}"):
                        idx = next((j for j, (x, _) in enumerate(st.session_state.drive_folder_path) if x == fid), None)
                        if idx is not None:
                            st.session_state.drive_folder_path = st.session_state.drive_folder_path[:idx + 1]
                            st.session_state.drive_current_folder_id = fid
                            st.session_state.drive_rename_file_id = None
                            st.session_state.drive_confirm_delete_id = None
                            st.rerun()
            if i < len(st.session_state.drive_folder_path) - 1:
                with breadcrumb_cols[i * 2 + 1]:
                    st.markdown("<span style='color:#555;'> › </span>", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#222; margin: 0.5rem 0;'>", unsafe_allow_html=True)

        with st.spinner("Cargando contenido..."):
            contenido = _dapi.list_folder_contents(current_folder_id)

        FOLDER_MIME = 'application/vnd.google-apps.folder'
        carpetas = [f for f in contenido if f.get('mimeType') == FOLDER_MIME]
        archivos = [f for f in contenido if f.get('mimeType') != FOLDER_MIME]

        if not contenido:
            st.markdown("<p style='color:#666; font-style:italic;'>Esta carpeta está vacía.</p>", unsafe_allow_html=True)
        else:
            for carpeta in carpetas:
                c_icon, c_name, c_btn = st.columns([0.05, 0.75, 0.2])
                with c_icon:
                    st.markdown(" 📁")
                with c_name:
                    st.markdown(f"<span style='color:#fbbf24;'>{carpeta['name']}</span>", unsafe_allow_html=True)
                with c_btn:
                    if st.button("Abrir →", key=f"open_{carpeta['id']}"):
                        st.session_state.drive_folder_path.append((carpeta['id'], f" {carpeta['name']}"))
                        st.session_state.drive_current_folder_id = carpeta['id']
                        st.session_state.drive_rename_file_id = None
                        st.session_state.drive_confirm_delete_id = None
                        st.rerun()

            if carpetas and archivos:
                st.markdown("<div style='border-top: 1px solid #222; margin: 0.3rem 0;'></div>", unsafe_allow_html=True)

            for archivo in archivos:
                fid  = archivo['id']
                fname = archivo['name']
                created = archivo.get('createdTime', '')[:10] if archivo.get('createdTime') else ''

                is_renaming = (st.session_state.drive_rename_file_id == fid)
                is_confirming_delete = (st.session_state.drive_confirm_delete_id == fid)

                if is_renaming:
                    r_col1, r_col2, r_col3 = st.columns([0.6, 0.2, 0.2])
                    with r_col1:
                        nuevo_nombre = st.text_input("Nuevo nombre:", value=fname, key=f"inp_rename_{fid}", label_visibility="collapsed")
                    with r_col2:
                        if st.button("✅ Guardar", key=f"confirm_rename_{fid}"):
                            with st.spinner("Renombrando..."):
                                ok = _dapi.rename_file(fid, nuevo_nombre)
                            if ok:
                                st.success(f"✅ Renombrado a '{nuevo_nombre}'")
                            else:
                                st.error("  Error al renombrar.")
                            st.session_state.drive_rename_file_id = None
                            st.rerun()
                    with r_col3:
                        if st.button("✖ Cancelar", key=f"cancel_rename_{fid}"):
                            st.session_state.drive_rename_file_id = None
                            st.rerun()

                elif is_confirming_delete:
                    st.warning(f"⚠️ ¿Seguro que querés eliminar **{fname}**? Esta acción es irreversible.")
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        if st.button("🗑️ Sí, eliminar", type="primary", key=f"confirm_del_{fid}"):
                            with st.spinner("Eliminando..."):
                                ok = _dapi.delete_file(fid)
                            if ok:
                                st.success(f"✅ '{fname}' eliminado.")
                            else:
                                st.error("  Error al eliminar.")
                            st.session_state.drive_confirm_delete_id = None
                            st.rerun()
                    with d_col2:
                        if st.button("Cancelar", key=f"cancel_del_{fid}"):
                            st.session_state.drive_confirm_delete_id = None
                            st.rerun()

                else:
                    f_col1, f_col2, f_col3, f_col4 = st.columns([0.05, 0.65, 0.15, 0.15])
                    with f_col1:
                        st.markdown("📄")
                    with f_col2:
                        st.markdown(f"<span style='color:#e5e7eb;'>{fname}</span>"
                                    f"<br><span style='color:#555; font-size:0.75rem;'>{created}</span>",
                                    unsafe_allow_html=True)
                    with f_col3:
                        if st.button(" Renombrar", key=f"ren_{fid}"):
                            st.session_state.drive_rename_file_id = fid
                            st.session_state.drive_confirm_delete_id = None
                            st.rerun()
                    with f_col4:
                        if st.button("🗑️ Eliminar", key=f"del_{fid}"):
                            st.session_state.drive_confirm_delete_id = fid
                            st.session_state.drive_rename_file_id = None
                            st.rerun()

    st.markdown("---")
    st.markdown("### 👥 Gestión de Usuarios")
    
    if st.session_state.username == 'admin':
        st.success("✅ Acceso de Administrador - Panel de Gestión de Usuarios")
        
        with st.expander("➕ Crear Nuevo Usuario", expanded=False):
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                new_username = st.text_input("Nombre de Usuario", key="admin_new_user")
            with col_u2:
                new_password = st.text_input("Contraseña", type="password", key="admin_new_pass")
            
            if st.button("Crear Usuario", type="primary"):
                if not new_username or not new_password:
                    st.error("Complete todos los campos")
                elif len(new_password) < 4:
                    st.error("La contraseña debe tener al menos 4 caracteres")
                else:
                    if auth.create_user(new_username, new_password):
                        st.success(f"✅ Usuario '{new_username}' creado exitosamente")
                    else:
                        st.error(f"  El usuario '{new_username}' ya existe")
        
        st.info("💡 Los usuarios creados podrán acceder inmediatamente con sus credenciales.")
    else:
        st.warning("⚠️ Solo el administrador puede gestionar usuarios. Contacte al admin para solicitar acceso.")
