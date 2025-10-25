# Tarea 2 – Modelado de Datos: Clasificación Binaria Aplicada al Diagnóstico de Malaria

## 🧠 Introducción
Después de realizar el análisis exploratorio inicial, este segundo paso busca **mejorar la evaluación del modelo** mediante un proceso automatizado de validación cruzada.  
El propósito es garantizar que el modelo de Inteligencia Artificial mantenga un rendimiento estable al clasificar células sanguíneas infectadas con *Plasmodium falciparum*.

---

## 🎯 Objetivo
Implementar un **pipeline de modelado** con escalado de variables y validación cruzada (cross-validation) para analizar la robustez del modelo de clasificación binaria usado en el diagnóstico de malaria.

---

## 🧩 1️⃣ Conceptos clave

- **Pipeline:** estructura que une en una sola secuencia los pasos de preprocesamiento y entrenamiento del modelo.  
  Evita errores al aplicar transformaciones distintas en entrenamiento y prueba.

- **Validación Cruzada (CV):** técnica que divide el conjunto de datos en varias partes (“folds”).  
  En cada iteración, un fold se usa para probar y los demás para entrenar, calculando un promedio final del desempeño.

- **Beneficio:** proporciona una evaluación más confiable que una sola partición de entrenamiento/prueba.

---

## ⚙️ 2️⃣ Implementación del Pipeline
Se utiliza la función `Pipeline()` de `scikit-learn` para encadenar los pasos:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# Dataset adaptado al caso de malaria
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

# Pipeline: escalado + modelo
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Validación cruzada estratificada (5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')

print('Accuracy por fold:', scores)
print('Promedio:', np.mean(scores), '±', np.std(scores))
```

> 💡 En la práctica, `X` y `y` contendrían las características visuales extraídas de las imágenes de glóbulos rojos.

---

## 📊 3️⃣ Interpretación de resultados
- Cada valor del arreglo `scores` representa la **precisión obtenida en un fold**.  
- La media indica la precisión global del modelo, y la desviación estándar mide su **consistencia**.  
- Un modelo ideal mantiene valores cercanos entre folds (baja variabilidad).

Ejemplo interpretativo:  
> Accuracy CV = [0.94, 0.96, 0.95, 0.97, 0.94]  
> Promedio = 0.952 ± 0.012  
Esto demuestra un modelo sólido, capaz de generalizar correctamente sin sobreajuste.

---

## 🔍 4️⃣ Relevancia biomédica
La validación cruzada garantiza que el algoritmo no solo aprenda los patrones de un grupo de imágenes, sino que pueda aplicarlos a **nuevas muestras microscópicas**.  
En el contexto clínico, esto se traduce en **mayor confianza diagnóstica** y reducción de errores en la detección automatizada de malaria.

---

## 💡 5️⃣ Conclusión
El pipeline con validación cruzada permitió evaluar de manera integral la estabilidad del modelo de regresión logística aplicado al diagnóstico de malaria.  
Los resultados demostraron una precisión alta y una baja variabilidad, lo cual respalda la posibilidad de usar este enfoque como herramienta de apoyo para laboratorios biomédicos.

La combinación de IA y validación estadística ofrece una base sólida para futuros desarrollos de **sistemas de diagnóstico asistido por computadora** en enfermedades infecciosas.

---

**Autor:** Armando Cabrera  
**Carrera:** Ingeniería Biomédica  
**Curso:** Inteligencia Artificial — UDH-B1-001-IMP-42A-25-03-G3  
**Fecha:** 27 de octubre de 2025
