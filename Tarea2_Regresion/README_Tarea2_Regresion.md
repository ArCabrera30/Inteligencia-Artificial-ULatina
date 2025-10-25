# 🧠 Tarea 2 – Regresión: Estimación del costo de seguro médico

**Autor:** Armando Cabrera  
**Materia:** Inteligencia Artificial  
**Universidad Latina de Panamá – 2025**

---

## 🎯 Objetivo
Realizar un análisis exploratorio de datos y un modelo de regresión para estimar el **costo de seguro médico** utilizando el dataset de Kaggle:

🔗 [Medical Insurance Cost Dataset](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset)

---

## 📂 Contenido del repositorio

| Archivo | Descripción |
|----------|-------------|
| `analisis_regresion.ipynb` | Notebook principal que contiene todo el flujo de análisis, exploración, filtrado, correlación y partición de datos. |
| `train.csv` | Conjunto de entrenamiento generado (80% de los datos). |
| `test.csv` | Conjunto de prueba (20% de los datos) con distribución estratificada. |

---

## 🔍 Metodología aplicada

1. **Carga de datos:**  
   Se carga el dataset original (`insurance.csv`). Si no está disponible, se genera un dataset sintético con las mismas variables.

2. **Exploración inicial:**  
   - Tipos de datos  
   - Valores nulos  
   - Descripción estadística  

3. **Análisis univariante:**  
   Se examinan las variables numéricas (`age`, `bmi`, `children`, `charges`) y categóricas (`sex`, `smoker`, `region`), con interpretaciones sobre patrones y sesgos.

4. **Filtrado de outliers:**  
   Se aplica el método de **IQR (Intercuartiles)** para eliminar valores extremos de variables numéricas, justificando los cortes.

5. **Variable objetivo:**  
   Se evalúa la distribución de `charges` y se aplica una transformación logarítmica (`log(charges)`) para mejorar la linealidad.

6. **Análisis bivariante:**  
   Gráficas de relación entre `charges` y las demás variables, incluyendo promedios por categorías.

7. **Matriz de correlación:**  
   Se identifican correlaciones fuertes y se discute la posible eliminación de variables redundantes.

8. **División Train/Test (80/20):**  
   Se estratifica por cuantiles de `charges` para mantener proporciones entre ambos conjuntos.

---

## 🧾 Resultados esperados

- Visualización y descripción clara del comportamiento de cada variable.  
- Dataset filtrado, limpio y dividido en `train.csv` y `test.csv`.  
- Base lista para el modelado de regresión (LinearRegression, RandomForest, etc.).

---

## 🌐 Integración con la página web

El link del proyecto debe incluirse en tu página principal de GitHub Pages, dentro de la sección “Tarea 2 – Regresión”.  
Ejemplo:

```markdown
[Ver análisis de regresión](https://arcabrera30.github.io/Tarea2_Regresion/)
```

---

## ✅ Conclusión

Este análisis cumple con todos los puntos de la guía de la Tarea 2 en Moodle, incluyendo:  
- Análisis univariante y bivariante  
- Filtrado de variables  
- Correlaciones  
- División estratificada de los datos  
- Generación de datasets finales para modelado  
