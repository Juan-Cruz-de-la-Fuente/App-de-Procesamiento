@echo off
title Servidor Local - BETZ APP
echo =======================================================
echo Iniciando el Servidor Local de la Aplicacion (Streamlit)
echo =======================================================
echo.
echo Por favor, NO cierre esta ventana negra de consola mientras usa el programa.
echo Para apagar el servidor, cierre esta ventana.
echo.
python -m streamlit run Main_App.py
pause
