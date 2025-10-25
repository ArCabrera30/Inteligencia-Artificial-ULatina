
# UI de Regresión (Gradio) — Costo de Seguro Médico

Archivos:
- `app_gradio_regresion.py`: interfaz Gradio. Entrena 4 modelos, calcula MAPE, MSE, RMSE y MAE, elige el mejor por RMSE y permite predecir.
- `requirements_regresion.txt`: dependencias mínimas para ejecutar en Colab o Hugging Face Spaces.

## Cómo ejecutar en Google Colab
1. Sube `app_gradio_regresion.py`, `requirements_regresion.txt` y (opcional) `insurance.csv` a Colab.
2. Instala dependencias y ejecuta la app:
   ```bash
   !pip install -r requirements_regresion.txt
   !python app_gradio_regresion.py
   ```
3. Colab mostrará un enlace público temporal para usar la UI.

> Si `insurance.csv` no está presente, la app genera un dataset sintético compatible automáticamente.
