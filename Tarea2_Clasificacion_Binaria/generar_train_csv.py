# generar_train_csv.py
# ---------------------------------------
# Generador de dataset simulado para el proyecto de clasificación binaria (Malaria)
# Autor: Armando Cabrera
# Universidad Latina de Panamá
# Curso: Inteligencia Artificial
# ---------------------------------------

import pandas as pd
import numpy as np

# Fijar semilla para que los datos sean reproducibles
np.random.seed(42)

# Cantidad de registros (100 sanos y 100 infectados)
n = 200

# ============================
# Simulación de características celulares
# ============================

# Intensidad promedio del color (0 a 255)
color_avg = np.random.normal(150, 25, n)

# Nivel de textura (0 = lisa, 1 = rugosa)
texture_level = np.random.uniform(0.2, 0.9, n)

# Grado de redondez de la célula (0 = irregular, 1 = redonda)
roundness = np.random.uniform(0.5, 1.0, n)

# Tamaño promedio del área celular (en micras²)
area_size = np.random.normal(300, 50, n)

# Etiquetas binarias: 0 = no infectada, 1 = infectada
label = np.concatenate([np.zeros(100), np.ones(100)])
np.random.shuffle(label)

# Crear DataFrame
df = pd.DataFrame({
    "color_avg": color_avg,
    "texture_level": texture_level,
    "roundness": roundness,
    "area_size": area_size,
    "label": label.astype(int)
})

# Guardar el archivo CSV
df.to_csv("train.csv", index=False)

print("✅ Archivo 'train.csv' generado correctamente con", len(df), "registros.")
print(df.head())
