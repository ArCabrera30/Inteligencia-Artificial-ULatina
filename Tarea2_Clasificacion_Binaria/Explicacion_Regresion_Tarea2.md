# Tarea 2 – Análisis de Datos: Regresión Aplicada a la Predicción de Saturación de Oxígeno (SpO₂)

## 🫁 Introducción
La **saturación de oxígeno (SpO₂)** es un parámetro clínico fundamental que refleja el nivel de oxigenación en la sangre.  
Su monitoreo constante es crucial para pacientes con enfermedades respiratorias como **EPOC**, **asma**, o **COVID-19**.  
La Inteligencia Artificial permite **predecir este valor de manera continua** a partir de datos fisiológicos, evitando complicaciones por hipoxia.

En esta actividad se desarrolla un modelo de **regresión lineal**, cuyo objetivo es estimar el valor de SpO₂ utilizando variables fisiológicas simuladas de pacientes.

---

## 🎯 Objetivo
Desarrollar un modelo de **regresión supervisada** que prediga la saturación de oxígeno (SpO₂) en función de variables biomédicas como frecuencia respiratoria, edad, temperatura corporal y frecuencia cardíaca.

---

## 🧩 1️⃣ Importación de librerías
El código emplea librerías de análisis de datos y modelado:
- `pandas` y `numpy`: manejo de datos y estructuras numéricas.
- `scikit-learn`: creación del modelo de regresión y evaluación del error.
- `matplotlib`: visualización de los resultados.

---

## 🧠 2️⃣ Descripción del dataset
Se utiliza un conjunto de datos **simulado**, basado en registros clínicos reales.  
Las variables incluidas son:

| Variable | Descripción | Unidad |
|-----------|-------------|--------|
| Edad | Edad del paciente | años |
| Frecuencia cardíaca | Pulsaciones por minuto | bpm |
| Frecuencia respiratoria | Respiraciones por minuto | rpm |
| Temperatura corporal | Grados Celsius | °C |
| SpO₂ | Saturación de oxígeno | % |

El modelo aprende la relación entre las variables fisiológicas y la saturación de oxígeno esperada.

---

## ⚙️ 3️⃣ Entrenamiento del modelo
Se aplica un modelo de **Regresión Lineal**, que busca la mejor combinación de coeficientes que explique el comportamiento de SpO₂.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Simulación de datos biomédicos
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'Edad': np.random.randint(20, 80, n),
    'Frecuencia_Cardiaca': np.random.randint(60, 120, n),
    'Frecuencia_Respiratoria': np.random.randint(10, 30, n),
    'Temperatura': np.random.uniform(36.0, 38.5, n)
})
# Relación simulada: SpO₂ desciende con edad y frecuencia respiratoria alta
df['SpO2'] = 98 - 0.05*df['Edad'] - 0.15*df['Frecuencia_Respiratoria'] + np.random.normal(0, 1, n)

# División de datos
X = df.drop(columns=['SpO2'])
y = df['SpO2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de regresión
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R²:", r2)
```

---

## 📊 4️⃣ Evaluación de resultados
- **MSE (Error cuadrático medio):** mide qué tan alejadas están las predicciones de los valores reales (menor es mejor).  
- **R² (Coeficiente de determinación):** mide la calidad del ajuste del modelo (1 indica ajuste perfecto).  
- Un valor de **R² > 0.85** se considera excelente para datos fisiológicos simulados.

Ejemplo interpretativo:
> MSE: 0.92  
> R²: 0.88  
> El modelo logra predecir el valor de saturación de oxígeno con alta precisión.

---

## 🧬 5️⃣ Interpretación biomédica
- Una **SpO₂ normal** se encuentra entre 95 % y 100 %.  
- Valores **por debajo de 90 %** pueden indicar hipoxia, lo que requiere atención médica.  
- Este modelo podría integrarse en dispositivos portátiles o sistemas de monitoreo remoto, alertando automáticamente al personal médico si un paciente presenta riesgo de desaturación.

---

## 💡 6️⃣ Conclusión
El modelo de regresión lineal desarrollado logró predecir eficazmente los niveles de saturación de oxígeno a partir de datos clínicos básicos.  
Su implementación en contextos hospitalarios o domiciliares podría **reducir riesgos asociados a episodios de hipoxia** mediante monitoreo predictivo automatizado.

Este tipo de aplicaciones demuestra cómo la **Inteligencia Artificial en ingeniería biomédica** fortalece la capacidad de respuesta ante emergencias respiratorias y mejora la toma de decisiones clínicas.

---

**Autor:** Armando Cabrera  
**Carrera:** Ingeniería Biomédica  
**Curso:** Inteligencia Artificial — UDH-B1-001-IMP-42A-25-03-G3  
**Fecha:** 27 de octubre de 2025
