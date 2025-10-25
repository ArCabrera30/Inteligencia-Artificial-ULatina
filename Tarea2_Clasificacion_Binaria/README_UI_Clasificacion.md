
# UI de Clasificación Binaria (Gradio) — Hábitos de Sueño (genérico)

Archivos:
- `app_gradio_clasificacion.py`: Interfaz Gradio que lee `train.csv`, detecta la columna objetivo binaria, divide 80/20 de forma estratificada, entrena 5 modelos (LogReg, KNN, Árbol, RF, SVM), calcula métricas (Accuracy, Precision, Recall, Specificity, F1), guarda `metricas_clasificacion.csv`, elige el **mejor por F1** y permite predecir con entradas dinámicas.
- `requirements_clasificacion.txt`: dependencias mínimas.

## Cómo ejecutar en Google Colab
```bash
!pip install -r requirements_clasificacion.txt
!python app_gradio_clasificacion.py
```

> Debes tener `train.csv` en la misma carpeta. La columna objetivo puede llamarse `label`, `target`, `y`, `clase`, `objetivo` o cualquier columna binaria (0/1); el script la detecta automáticamente.
