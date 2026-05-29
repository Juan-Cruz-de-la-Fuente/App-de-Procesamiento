import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from codigo_fuente.Calculations_Core import (
    unir_archivos_incertidumbre,
    extraer_matriz_presiones_completa,
    crear_vtk_plano_presion_2d,
    crear_vtk_superficie_3d_delaunay
)
from codigo_fuente import Auth_Manager as auth

from codigo_fuente.Graficos_Comunes import mostrar_configuracion_sensores

def show_herramientas():
    st.markdown("""
    <div class="header-container">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🔧 HERRAMIENTAS DE PROCESAMIENTO
        </h1>
        <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Herramientas avanzadas para el procesamiento y análisis de datos aerodinámicos
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    if 'archivos_unidos' not in st.session_state:
        st.session_state.archivos_unidos = None
    if 'matriz_presiones' not in st.session_state:
        st.session_state.matriz_presiones = None
    if 'archivo_vtk' not in st.session_state:
        st.session_state.archivo_vtk = None
    
    # --- HERRAMIENTA 1: Unir Archivos de Incertidumbre ---
    with st.container():
        st.markdown("""
        <div class="section-card" style="border-left: 5px solid #0ea5e9; margin-bottom: 10px;">
            <h3 style="color: #0ea5e9; margin: 0;"> 01. UNIÓN DE ARCHIVOS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        c_desc, c_func = st.columns([1, 2])
        
        with c_desc:
            st.markdown("""
            <p style="color: #ccc; font-size: 0.95rem;">
                Utilidad para combinar múltiples archivos CSV de incertidumbre en un único conjunto de datos.
                <br><br>
                El sistema detectará automáticamente si hay puntos temporales sobrepuestos y generará una alerta.
            </p>
            """, unsafe_allow_html=True)

        with c_func:
            archivos_union = st.file_uploader(
                "Seleccionar archivos CSV:",
                type=['csv'],
                accept_multiple_files=True,
                key="union_archivos",
                label_visibility="collapsed"
            )
            
            c_input, c_btn = st.columns([2, 1])
            with c_input:
                 nombre_archivo_union = st.text_input("Nombre saliente:", value="archivos_unidos", key="nombre_union")
            with c_btn:
                 st.write("") # Spacer
                 st.write("")
                 btn_unir = st.button("🔗 Unir Ahora", key="btn_unir", type="primary", use_container_width=True)

            if btn_unir:
                if archivos_union and len(archivos_union) > 1:
                    with st.spinner("Uniendo archivos..."):
                        contenido_unido, puntos_sobrepuestos = unir_archivos_incertidumbre(
                            archivos_union, nombre_archivo_union
                        )
                        
                        if contenido_unido:
                            st.session_state.archivos_unidos = {
                                'contenido': contenido_unido,
                                'nombre': nombre_archivo_union,
                                'puntos_sobrepuestos': puntos_sobrepuestos
                            }
                            
                            st.success(f"✅ {len(archivos_union)} archivos unidos correctamente")
                            
                            if puntos_sobrepuestos:
                                st.warning(f"⚠️ Se detectaron {len(puntos_sobrepuestos)} puntos sobrepuestos")
                                with st.expander("Ver puntos sobrepuestos"):
                                    for punto in puntos_sobrepuestos:
                                        st.write(f"Y={punto[0]}, Z={punto[1]}, Tiempo={punto[2]}s")
                            
                            st.download_button(
                                label="📥 Descargar CSV Unido",
                                data=contenido_unido.encode('utf-8-sig'),
                                file_name=f"{nombre_archivo_union}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("  Seleccione min. 2 archivos")

    st.markdown("---")

    # --- HERRAMIENTA 2: Extraer matriz de presiones ---
    with st.container():
        st.markdown("""
        <div class="section-card" style="border-left: 5px solid #f59e0b; margin-bottom: 10px;">
            <h3 style="color: #f59e0b; margin: 0;">📊 02. MATRIZ DE PRESIONES</h3>
        </div>
        """, unsafe_allow_html=True)
        
        c_desc, c_func = st.columns([1, 2])
        
        with c_desc:
            st.markdown("""
            <p style="color: #ccc; font-size: 0.95rem;">
                Extrae una matriz estructurada (Filas=Y, Columnas=Z) de presiones a partir de datos crudos.
                <br><br>
                Ideal para análisis numérico posterior o verificación manual de campos.
            </p>
            """, unsafe_allow_html=True)
            
        with c_func:
            archivo_matriz = st.file_uploader(
                "Cargar archivo de incertidumbre:",
                type=['csv'],
                key="archivo_matriz",
                 label_visibility="collapsed"
            )
            
            c_input, c_btn = st.columns([2, 1])
            with c_input:
                nombre_matriz = st.text_input("Nombre matriz:", value="matriz_presiones", key="nombre_matriz")
            with c_btn:
                st.write("")
                st.write("")
                btn_matriz = st.button("📊 Extraer", key="btn_matriz", type="primary", use_container_width=True)

            with st.expander("Configuración de Sensores y Atmósfera (Avanzado)"):
                configuracion_matriz = mostrar_configuracion_sensores("herramienta2")
                upl_inf_vtk = st.file_uploader("🔗 'Valores en el infinito.txt' para normalización VTK:", type=['txt', 'csv'], key="upl_inf_vtk")

            if btn_matriz:
                if archivo_matriz:
                    with st.spinner("Procesando..."):
                        matriz = extraer_matriz_presiones_completa(archivo_matriz, configuracion_matriz, upl_inf_vtk)

                        if matriz is not None and not matriz.empty:
                            st.session_state.matriz_presiones = {
                                'matriz': matriz,
                                'nombre': nombre_matriz
                            }

                            st.success("✅ Matriz extraída")
                            st.dataframe(matriz.head(), use_container_width=True)

                            df_matriz = pd.DataFrame(matriz, columns=["Y", "Z", "Presion"])
                            csv_matriz = df_matriz.to_csv(sep=';', decimal=',', index=False)
                            st.download_button(
                                label="📥 Descargar Matriz CSV",
                                data=csv_matriz.encode('utf-8-sig'),
                                file_name=f"{nombre_matriz}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("  Falta archivo")

    st.markdown("---")

    # --- HERRAMIENTA 3: Generador VTK ---
    with st.container():
        st.markdown("""
        <div class="section-card" style="border-left: 5px solid #10b981; margin-bottom: 10px;">
            <h3 style="color: #10b981; margin: 0;">🎯 03. GENERADOR VTK (CFD)</h3>
            <p style="color:#aaa; margin: 6px 0 0 0; font-size:0.9rem;">
                Convierte datos de presión en archivos <b>.VTK</b> compatibles con ParaView / Salome.
                Elegí el tipo de archivo (2D, 3D o 4D) y la fuente de datos.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🗺 VTK 2D — Plano de Presión  |  Plano YZ (X fijo). Presión como color.", expanded=True):

            fuente_2d = st.radio(
                "Fuente de datos:",
                ["📂 Drive 2D (base de datos)", "💾 Memoria (sesión actual)", " Subir CSV nuevo"],
                key="fuente_vtk2d", horizontal=True
            )

            df_vtk2d = None
            fname_2d_drive = None

            if fuente_2d == "📂 Drive 2D (base de datos)":
                archivos_2d = auth.get_user_files_2d(st.session_state.username)
                if archivos_2d:
                    dict_2d = {f"{a[1]} [{a[2][:10] if a[2] else ''}]": a for a in archivos_2d}
                    sel_2d = st.selectbox("Seleccionar archivo 2D:", list(dict_2d.keys()), key="sel_drive2d_vtk")
                    if sel_2d:
                        fid_2d = dict_2d[sel_2d][0]
                        fname_2d_drive = dict_2d[sel_2d][1]
                        raw_2d = auth.download_file_2d(fid_2d)
                        if raw_2d:
                            import io
                            df_vtk2d = pd.read_csv(io.BytesIO(raw_2d), sep=';', decimal=',')
                            st.success(f"✅ Cargado desde Drive: **{fname_2d_drive}**")
                else:
                    st.info("No hay archivos 2D en Drive. Guardá desde BETZ 2D → Paso 4.")

            elif fuente_2d == "💾 Memoria (sesión actual)":
                mat_disp = st.session_state.get('matriz_presiones')
                if mat_disp:
                    st.success(f"✅ Usando: {mat_disp['nombre']}")
                    df_vtk2d = mat_disp['matriz']
                else:
                    st.warning("No hay matriz en sesión. Usá Herramienta 02 o cargá desde Drive.")

            else:
                csv_new = st.file_uploader("CSV Matriz (sep=;, dec=,):", type=['csv'], key="up_vtk2d")
                if csv_new:
                    try:
                        df_vtk2d = pd.read_csv(csv_new, sep=';', decimal=',')
                    except Exception as e:
                        st.error(f"Error leyendo CSV: {e}")

            if df_vtk2d is not None:
                x_vtk2d = st.number_input(" Posición X [mm]:", value=0.0, step=10.0, key="x_vtk2d")
                res_vtk2d = st.slider("Suavizado:", 1, 5, 2, key="res_vtk2d")

                if fname_2d_drive:
                    stem_2d = os.path.splitext(fname_2d_drive)[0]
                    nombre_auto_vtk2d = "VTK-" + stem_2d[stem_2d.index("-")+1:] if "-" in stem_2d else f"VTK-{stem_2d}"
                else:
                    nombre_auto_vtk2d = f"VTK-X{int(x_vtk2d)}-2D"

                c2d_chk, c2d_nom = st.columns([0.18, 0.82])
                if c2d_chk.checkbox("Nombre libre", key="chk_vtk2d"):
                    nombre_vtk2d = c2d_nom.text_input("Nombre:", placeholder=nombre_auto_vtk2d, key="nom_vtk2d")
                    if not nombre_vtk2d: nombre_vtk2d = nombre_auto_vtk2d
                else:
                    nombre_vtk2d = nombre_auto_vtk2d
                    c2d_nom.code(f"{nombre_vtk2d}.vtk")

                if st.button("🗺 Generar VTK 2D", key="btn_gen_vtk2d", type="primary"):
                    resultado_2d = crear_vtk_plano_presion_2d(df_vtk2d, nombre_vtk2d, x_vtk2d)
                    if resultado_2d:
                        vtk_path_2d, vtk_bytes_2d = resultado_2d
                        c_dl1, c_dl2 = st.columns(2)
                        with c_dl1:
                            st.download_button("📥 Descargar VTK 2D", vtk_bytes_2d,
                                               file_name=os.path.basename(vtk_path_2d),
                                               mime="application/octet-stream", key="dl_vtk2d")
                        with c_dl2:
                            if st.button(" Guardar en Drive", key="save_vtk2d_drive"):
                                if auth.save_vtk_plano(st.session_state.username,
                                                       os.path.basename(vtk_path_2d), vtk_bytes_2d):
                                    st.success("✅ Subido → HERRAMIENTAS/ARCHIVOS VTK/PLANOS DE PRESION")
                                else:
                                    st.error("Error al subir a Drive")
                    else:
                        st.error("  No se pudo generar el VTK.")

        with st.expander("🕸 VTK 3D — Malla Delaunay  |  Triangulación 3D fiel a los datos. Ideal para CFD.", expanded=True):

            fuente_3d = st.radio(
                "Fuente de datos:",
                ["📂 Drive 3D (base de datos)", "💾 Memoria (sesión actual)", " Subir CSV nuevo"],
                key="fuente_vtk3d", horizontal=True
            )

            df_vtk3d = None
            fname_3d_drive = None

            if fuente_3d == "📂 Drive 3D (base de datos)":
                archivos_3d = auth.get_user_surfaces(st.session_state.username)
                if archivos_3d:
                    dict_3d = {f"{s[1]} [{s[3][:10] if s[3] else ''}]": s for s in archivos_3d}
                    sel_3d = st.selectbox("Seleccionar plano 3D:", list(dict_3d.keys()), key="sel_drive3d_vtk")
                    if sel_3d:
                        import json as _json
                        fname_3d_drive = dict_3d[sel_3d][1]
                        data_str_3d = dict_3d[sel_3d][4]
                        df_vtk3d = pd.DataFrame(_json.loads(data_str_3d))
                        st.success(f"✅ Cargado: **{fname_3d_drive}**")
                else:
                    st.info("No hay planos 3D en Drive. Guardá desde BETZ 3D → Paso 5.")

            elif fuente_3d == "💾 Memoria (sesión actual)":
                mat_disp3d = st.session_state.get('matriz_presiones')
                if mat_disp3d:
                    st.success(f"✅ Usando: {mat_disp3d['nombre']}")
                    df_vtk3d = mat_disp3d['matriz']
                else:
                    st.warning("No hay matriz en sesión.")

            else:
                csv_new3d = st.file_uploader("CSV Matriz:", type=['csv'], key="up_vtk3d")
                if csv_new3d:
                    try:
                        df_vtk3d = pd.read_csv(csv_new3d, sep=';', decimal=',')
                    except Exception as e:
                        st.error(f"Error: {e}")

            if df_vtk3d is not None:
                x_vtk3d = st.number_input(" Posición X [mm]:", value=0.0, step=10.0, key="x_vtk3d")

                if fname_3d_drive:
                    stem_3d = os.path.splitext(fname_3d_drive)[0]
                    nombre_auto_vtk3d = "VTK-" + stem_3d[stem_3d.index("-")+1:] if "-" in stem_3d else f"VTK-{stem_3d}"
                else:
                    nombre_auto_vtk3d = f"VTK-X{int(x_vtk3d)}-3D"

                c3d_chk, c3d_nom = st.columns([0.18, 0.82])
                if c3d_chk.checkbox("Nombre libre", key="chk_vtk3d"):
                    nombre_vtk3d = c3d_nom.text_input("Nombre:", placeholder=nombre_auto_vtk3d, key="nom_vtk3d")
                    if not nombre_vtk3d: nombre_vtk3d = nombre_auto_vtk3d
                else:
                    nombre_vtk3d = nombre_auto_vtk3d
                    c3d_nom.code(f"{nombre_vtk3d}.vtk")

                if st.button("🕸 Generar VTK 3D Delaunay", key="btn_gen_vtk3d", type="primary"):
                    res_3d = crear_vtk_superficie_3d_delaunay(df_vtk3d, nombre_vtk3d, x_vtk3d)
                    if res_3d:
                        with open(res_3d, "rb") as f3d:
                            vtk_bytes_3d = f3d.read()
                        c_dl3, c_dl4 = st.columns(2)
                        with c_dl3:
                            st.download_button("📥 Descargar VTK 3D", vtk_bytes_3d,
                                               file_name=f"{nombre_vtk3d}.vtk",
                                               mime="application/octet-stream", key="dl_vtk3d")
                        with c_dl4:
                            if st.button(" Guardar en Drive", key="save_vtk3d_drive"):
                                if auth.save_vtk_superficie(st.session_state.username,
                                                            f"{nombre_vtk3d}.vtk", vtk_bytes_3d):
                                    st.success("✅ Subido → HERRAMIENTAS/ARCHIVOS VTK/SUPERFICIES 3D")
                                else:
                                    st.error("Error al subir a Drive")
                    else:
                        st.error("  No se pudo generar el VTK.")

        with st.expander("🌌 VTK 4D — Multi-plano  |  Genera un VTK Delaunay por cada plano 4D, en su estación X.", expanded=True):

            archivos_4d_vtk = auth.get_user_surfaces_4d(st.session_state.username)
            if not archivos_4d_vtk:
                st.info("No hay planos 4D en Drive. Guardá desde BETZ 4D → Paso 1.")
            else:
                dict_4d_vtk = {f"{s[1]} (X={s[2]}mm) [{s[3][:10] if s[3] else ''}]": s for s in archivos_4d_vtk}
                sels_4d = st.multiselect("Seleccionar planos 4D:", list(dict_4d_vtk.keys()), key="sels_4d_vtk")

                if sels_4d:
                    rename_4d = st.checkbox("Personalizar nombres de salida", key="chk_rename_4d")

                    if rename_4d:
                        custom_names_4d = {}
                        for lab in sels_4d:
                            s4 = dict_4d_vtk[lab]
                            stem_4d = s4[1]
                            auto_4d = "VTK-" + stem_4d[stem_4d.index("-")+1:] if "-" in stem_4d else f"VTK-{stem_4d}"
                            custom_names_4d[lab] = st.text_input(
                                f"  Nombre para {s4[1]}:", value=auto_4d, key=f"nom4d_{s4[1]}"
                            )
                    else:
                        for lab in sels_4d:
                            s4 = dict_4d_vtk[lab]
                            stem_4d = s4[1]
                            auto_4d = "VTK-" + stem_4d[stem_4d.index("-")+1:] if "-" in stem_4d else f"VTK-{stem_4d}"
                            st.code(f"{auto_4d}.vtk", language=None)

                    if st.button("🌌 Generar VTK por cada plano", key="btn_gen_vtk4d", type="primary"):
                        import json as _json4
                        for lab in sels_4d:
                            s4 = dict_4d_vtk[lab]
                            df_s4 = pd.DataFrame(_json4.loads(s4[4]))
                            x_s4  = s4[2]
                            stem_4d = s4[1]
                            auto_4d = "VTK-" + stem_4d[stem_4d.index("-")+1:] if "-" in stem_4d else f"VTK-{stem_4d}"
                            nom_s4 = custom_names_4d.get(lab, auto_4d) if rename_4d else auto_4d
                            with st.spinner(f"Generando {nom_s4}.vtk..."):
                                res_s4 = crear_vtk_superficie_3d_delaunay(df_s4, nom_s4, x_s4)
                            if res_s4:
                                with open(res_s4, "rb") as f4:
                                    bytes_s4 = f4.read()
                                st.download_button(
                                    f"📥 {nom_s4}.vtk",
                                    bytes_s4,
                                    file_name=f"{nom_s4}.vtk",
                                    mime="application/octet-stream",
                                    key=f"dl_vtk4d_{s4[1]}"
                                )
                                st.success(f"✅ {nom_s4}.vtk generado")
                            else:
                                st.error(f"Error generando VTK para {s4[1]}")
