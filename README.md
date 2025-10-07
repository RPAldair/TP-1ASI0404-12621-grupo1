# Asistente de Pictogramas — Prototipo

Instrucciones rápidas:

- Crear y activar el entorno virtual (ya se incluyó `.venv` en el repo por el asistente):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
```

- Instalar dependencias (si falta algo):

```powershell
python -m pip install -r requirements.txt
```

- Generar dataset sintético y dividir:

```powershell
python scripts/preprocess.py
```

- Entrenar Naive Bayes:

```powershell
python scripts/train_nb.py
```

Los modelos entrenados se guardarán en `models/`.
