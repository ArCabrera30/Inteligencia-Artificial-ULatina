# Tarea 2 – Análisis de Datos: Clasificación Binaria en Diagnóstico de Malaria

## 🩸 Introducción
La **malaria** es una enfermedad infecciosa causada por el parásito *Plasmodium falciparum*, transmitido por el mosquito *Anopheles*.  
A nivel mundial, afecta a millones de personas, principalmente en regiones tropicales. El diagnóstico temprano es esencial para evitar complicaciones graves y reducir la mortalidad.

En este proyecto se utiliza la **Inteligencia Artificial (IA)** para clasificar imágenes microscópicas de glóbulos rojos entre:
- **Células sanas (no infectadas)**
- **Células infectadas por malaria**

---

## 🎯 Objetivo
Aplicar un modelo de **clasificación binaria supervisada** para identificar si una célula sanguínea está infectada o no, utilizando características extraídas de imágenes de microscopía digital.

---

## 🔬 1️⃣ Importación de librerías
El código utiliza herramientas del ecosistema Python:
- `pandas` y `numpy`: manejo de datos y estructuras numéricas.
- `scikit-learn`: creación de modelos de aprendizaje automático.
- `matplotlib`: visualización de resultados.
- `LogisticRegression`: modelo matemático para clasificación binaria.

Estas librerías permiten procesar datos biomédicos de forma estructurada y reproducible.

---

## 🧫 2️⃣ Carga y descripción del dataset
Se utiliza un conjunto de datos público con miles de imágenes de células obtenidas por microscopía óptica.  
Cada muestra está etiquetada como:

| Etiqueta | Descripción |
|-----------|--------------|
| 0 | Célula no infectada |
| 1 | Célula infectada con *Plasmodium* |

En este caso, las imágenes fueron preprocesadas para convertir sus características visuales (color, textura, contraste) en datos numéricos que el modelo puede interpretar.

---

## ⚙️ 3️⃣ División del conjunto de datos
Los datos se separan en dos subconjuntos:
- **Entrenamiento (80 %)** → para que el modelo aprenda patrones.
- **Prueba (20 %)** → para evaluar su capacidad de generalización.

Esto garantiza que el modelo no memorice los ejemplos, sino que aprenda a reconocer nuevos casos.

---

## 📏 4️⃣ Escalado de variables
Antes del entrenamiento, los valores numéricos se normalizan con `StandardScaler()`, asegurando que cada característica (como color promedio o nivel de textura) tenga la misma importancia en el cálculo matemático.

---

## 🧮 5️⃣ Entrenamiento del modelo
Se utiliza una **Regresión Logística**, un modelo estadístico capaz de predecir la probabilidad de pertenencia a una clase (infectada o no infectada).  
Este algoritmo es eficiente y fácil de interpretar en contextos clínicos.

El entrenamiento sigue la estructura:

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
