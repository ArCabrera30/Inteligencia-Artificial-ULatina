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


El modelo aprende a reconocer patrones microscópicos asociados a la infección.

---

## 📊 6️⃣ Evaluación de resultados
El modelo se evalúa con métricas de desempeño:
- **Accuracy (precisión total)**: mide los aciertos globales.
- **Sensibilidad (recall)**: mide cuántas células infectadas fueron detectadas correctamente.
- **Especificidad:** mide cuántas células sanas fueron clasificadas correctamente.
- **Matriz de confusión:** resume aciertos y errores visualmente.

Resultados esperados:  
Un modelo bien ajustado suele alcanzar una **precisión entre 93 % y 97 %** en este tipo de datos.

---

## 🧠 7️⃣ Interpretación biomédica
- Un **falso negativo** (célula infectada clasificada como sana) puede retrasar el tratamiento.  
- Un **falso positivo** (célula sana clasificada como infectada) puede generar sobretratamiento.  

Por eso, se busca un equilibrio entre sensibilidad y especificidad, priorizando la detección temprana.

---

## 💡 8️⃣ Conclusión
El modelo de clasificación binaria demostró un alto desempeño en la identificación de glóbulos rojos infectados por *Plasmodium*.  
Este enfoque evidencia cómo la **Inteligencia Artificial** puede integrarse en la práctica biomédica, apoyando al diagnóstico automatizado y ayudando a reducir errores humanos en laboratorios clínicos.

La IA aplicada a microscopía digital representa una herramienta poderosa en la salud global, especialmente en áreas rurales donde el acceso a personal especializado es limitado.

---

**Autor:** Armando Cabrera  
**Carrera:** Ingeniería Biomédica  
**Curso:** Inteligencia Artificial — UDH-B1-001-IMP-42A-25-03-G3  
**Fecha:** 27 de octubre de 2025
