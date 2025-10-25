
# 🧠 Tarea 2 – Clasificación Binaria: Análisis de Hábitos de Sueño

**Autor:** Armando Cabrera  
**Materia:** Inteligencia Artificial  
**Universidad Latina de Panamá – 2025**

---

## 🎯 Objetivo
Realizar un **análisis exploratorio de datos (EDA)** y un **modelo de clasificación binaria** para detectar patrones en los **hábitos de sueño y niveles de estrés**.  
El proyecto aplica técnicas de inteligencia artificial y aprendizaje supervisado.

---

## 📂 Contenido del repositorio

| Archivo | Descripción |
|----------|-------------|
| `analisis_binario.ipynb` | Notebook principal con el análisis exploratorio (EDA), creación de variable binaria, filtrado, correlaciones y división de datos (train/test). |
| `train.csv` / `test.csv` | Datasets generados para el entrenamiento y validación del modelo. |
| `app_gradio_clasificacion.py` | Interfaz Gradio que entrena 5 modelos de clasificación, calcula métricas y permite realizar predicciones interactivas. |
| `metricas_clasificacion.csv` | Tabla con los resultados (Accuracy, Precision, Recall, Specificity y F1) de los modelos evaluados. |
| `requirements_clasificacion.txt` | Dependencias necesarias para ejecutar la aplicación en Google Colab o Hugging Face Spaces. |
| `README_UI_Clasificacion.md` | Instrucciones específicas para correr la interfaz gráfica. |

---

## 🔍 Metodología aplicada

### 1️⃣ **Análisis exploratorio (EDA)**
- Carga y descripción del dataset de Kaggle:  
  [Lifestyle and Sleep Patterns Dataset](https://www.kaggle.com/datasets/minahilfatima12328/lifestyle-and-sleep-patterns)
- Limpieza y tratamiento de valores vacíos.  
- Análisis **univariante** y **bivariante** (variables numéricas y categóricas).  
- Filtrado de **outliers** mediante el método de **IQR (intercuartiles)**.  
- Creación de la **variable binaria** `estresado`:
  - Estrés moderado: valores entre 3 y 6.  
  - Estrés alto: valores ≥ 7.
- Eliminación de la variable numérica original de estrés (por redundancia).  
- Cálculo de la **matriz de correlación** para identificar relaciones significativas.  
- División del dataset en `train.csv` y `test.csv` con proporción **80/20 estratificada**.

---

### 2️⃣ **Modelado de Clasificación**
- Modelos entrenados:
  1. **Logistic Regression**
  2. **K-Nearest Neighbors (KNN)**
  3. **Decision Tree**
  4. **Random Forest**
  5. **Support Vector Machine (SVM)**
- Métricas calculadas:
  - **Accuracy**
  - **Precision**
  - **Recall (Sensibilidad)**
  - **Specificity**
  - **F1-Score**
- El mejor modelo se selecciona automáticamente según el **mayor F1-Score**.

---

### 3️⃣ **Interfaz de usuario (Gradio UI)**
- La app **`app_gradio_clasificacion.py`** permite ingresar características (edad, sueño, hábitos, etc.) y obtener una predicción del nivel de estrés.
- Se genera un enlace público desde Colab o Hugging Face Spaces para interactuar con el modelo sin código.

---

## 📊 Resultados esperados
- Análisis visual y estadístico de las variables del dataset.  
- Reducción de ruido mediante filtrado de outliers.  
- Comparación de **5 modelos de clasificación** con sus métricas respectivas.  
- Identificación del **mejor modelo** y justificación de la métrica elegida.  
- Implementación de una **UI funcional** accesible vía navegador.

---

## 🌐 Integración con la página web
El link de este proyecto debe agregarse en la sección “Tarea 2 – Clasificación Binaria” de tu GitHub Pages.  
Ejemplo:

```markdown
[Ver análisis de clasificación binaria](https://arcabrera30.github.io/Tarea2_Clasificacion_Binaria/)
```

---

## ✅ Conclusión
Este proyecto cumple con **todos los puntos requeridos** en la guía de Moodle:
- Análisis exploratorio de datos  
- Filtrado y correlación de variables  
- Creación de variable binaria  
- Entrenamiento de modelos de clasificación  
- Evaluación de métricas  
- Análisis de errores  
- Interfaz Gradio interactiva  

> Con este trabajo, se logra un flujo completo de **inteligencia artificial aplicada a un problema de salud y bienestar**, desde la exploración de datos hasta la implementación de una interfaz funcional.
