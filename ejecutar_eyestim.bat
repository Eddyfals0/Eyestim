@echo off
title EyeStim - Detector de Mirada y Pupila
cd /d "%~dp0"
echo ============================================================
echo   Iniciando EyeStim con Camara Web y Redes Neuronales...
echo ============================================================
echo.
echo Presiona 'q' en la ventana del video para salir y guardar reporte.
echo Presiona 't' para ver el panel de entrenamiento.
echo.
".\.venv\Scripts\python.exe" "src\show_eyes.py"
if %errorlevel% neq 0 (
    echo.
    echo [AVISO] El programa se cerro con codigo de salida %errorlevel%.
    pause
)
