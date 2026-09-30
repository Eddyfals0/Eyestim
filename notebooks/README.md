# Reproducción de la auditoría de datasets

Desde la raíz del repositorio, en PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe scripts\download_validation_data.py eyedentify lpw-center swirski nemar --eyedentify-participants 1-51 --frames-per-session 4
.\.venv\Scripts\python.exe src\train_center.py
.\.venv\Scripts\python.exe src\train_diameter.py --augment
.\.venv\Scripts\python.exe research\final_validation.py
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name eyestim-research --display-name "EyeStim Research"
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 notebooks\01_bioid_fitness_audit.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 notebooks\02_pupilometry_benchmark.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 notebooks\03_attention_construct_validation.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 notebooks\04_final_training_and_validation.ipynb
.\.venv\Scripts\python.exe research\paper_figures.py
.\.venv\Scripts\python.exe scripts\build_academic_paper.py
```

Los notebooks contienen sus salidas ejecutadas. Las figuras consolidadas se guardan en `docs/research/` y los datos externos permanecen fuera de Git bajo `data/external/`.

El notebook 04 audita todas las imágenes EyeDentify descargadas, evalúa los
checkpoints congelados y regenera `docs/training/project_dashboard.png`.

Los dos últimos comandos generan las figuras académicas y el paper final en
`output/pdf/eyestim_paper_academico_2026.pdf`.
