# 🚀 AeroAnalysis Pro (BETZ APP)
### Laboratorio de Aerodinámica y Fluidos — UTN HAEDO

AeroAnalysis Pro es una plataforma avanzada de procesamiento de datos aerodinámicos y visualización de estelas, diseñada específicamente para ensayos en túnel de viento. La aplicación permite realizar análisis detallados en varias dimensiones, interpolación de campos fluidodinámicos, detección de vórtices y generación de animaciones tridimensionales complejas.

---

## 🌌 Características Principales

*   **📈 Ensayo Estela 1D y 2D:** Visualización de perfiles de velocidad/presión en la estela y cálculo de pérdida de cantidad de movimiento mediante el método de Betz.
*   **🌪️ Superficies 3D:** Reconstrucción tridimensional mediante triangulación de Delaunay y campos volumétricos de presión.
*   **🌌 Ensayo Estela 4D:** Análisis de múltiples planos espaciales en el tiempo.
*   **🌀 Análisis de Vórtices:** Detección automática de núcleos de vórtices y análisis cinemático del campo rotacional.
*   **🎬 Animación 4D:** Renderizado secuencial de fotogramas y reconstrucción de la dinámica temporal de la estela.
*   **📊 Interpolación:** Herramientas avanzadas para remuestreo y regularización de mallas de datos dispersos.
*   **📦 Modelos 3D y Herramientas:** Exportación y visualización en formatos estándares como VTK y CSV.

---

## 🛠️ Arquitectura y Tecnologías

*   **Core:** Python 3.x
*   **Frontend / UI:** [Streamlit](https://streamlit.io/) con estilos personalizados oscuros estilo "SpaceX" (fuentes Orbitron e Inter).
*   **Procesamiento Numérico:** NumPy, SciPy, Pandas.
*   **Visualización:** Plotly, PyVista / VTK, Matplotlib.
*   **Persistencia:** Google Drive API (Sincronización remota automática) y base de datos SQLite descentralizada.

---

## 💻 Ejecución Local

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/Juan-Cruz-de-la-Fuente/App-de-Procesamiento.git
    cd App-de-Procesamiento
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Correr la aplicación:**
    ```bash
    streamlit run Main_App.py
    ```

---

## ☁️ Despliegue en la Nube (Streamlit Cloud)

Para que la aplicación funcione correctamente en Streamlit Community Cloud, se debe configurar el secreto de autenticación de Google Drive en las opciones avanzadas de Streamlit (`Secrets`):

```toml
google_token_json = '''
{
  "token": "...",
  "refresh_token": "...",
  "token_uri": "https://oauth2.googleapis.com/token",
  "client_id": "...",
  "client_secret": "...",
  "scopes": ["https://www.googleapis.com/auth/drive"],
  "expiry": "..."
}
'''
```

---

## 🔒 Seguridad y Privacidad

Esta aplicación maneja credenciales sensibles para la conexión con Google Drive. El archivo `.gitignore` está configurado para **excluir** estrictamente:
*   La carpeta de tokens y credenciales (`identificaciones_api/`).
*   Los tokens OAuth locales (`OAuth_Token.json`).
*   La base de datos local de usuarios (`users.db`).
*   Archivos temporales de compilación de Python (`__pycache__/`).

*Desarrollado para el Laboratorio de Aerodinámica y Fluidos de la Universidad Tecnológica Nacional, Facultad Regional Haedo.*
