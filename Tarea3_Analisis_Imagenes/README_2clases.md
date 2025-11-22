# 🧠 Proyecto de IA – Análisis y Preparación de Imágenes (2 Clases)

Este proyecto realiza el **análisis exploratorio (EDA)**, **preprocesamiento**, **detección de imágenes defectuosas** y **separación estratificada** del dataset de dos clases:

- **Natalie Portman**
- **Scarlett Johansson**

El trabajo cumple con las indicaciones del curso:
- ❌ No usa *scikit-learn*
- ✔ Usa **PyTorch, Torchvision, PIL, OpenCV, ImageHash**
- ✔ Incluye control de calidad del dataset
- ✔ EDA completo (estadísticas RGB, resoluciones, corruptas, duplicadas, oscuras, borrosas)
- ✔ Preprocesamiento y data augmentation
- ✔ Split estratificado hecho manualmente (sin sklearn)

---

## 📂 Estructura del proyecto

```
Tarea3_Analisis_Imagenes/
│
├── EDA_2clases_celebs.ipynb
├── requirements_2clases.txt
├── README_2clases.md   ← este archivo
│
└── data/
    └── raw/
        ├── Natalie_Portman/
        └── Scarlett_Johansson/
```

Coloca las imágenes de cada clase dentro de sus carpetas correspondientes.

---

## 🔎 Contenido del Notebook

El notebook `EDA_2clases_celebs.ipynb` incluye:

### **1. Exploración estructural**
- Conteo de imágenes por clase
- Identificación de formatos válidos
- Estadísticas de resolución (ancho, alto)
- Detección de imágenes corruptas o ilegibles

### **2. Análisis estadístico RGB**
- Media y desviación estándar por canal (R, G, B)
- Detección de valores atípicos
- Rango global (mínimos y máximos)
- Brillo promedio por imagen

### **3. Detección de imágenes defectuosas**
- Imágenes **oscuras**  
- Imágenes **borrosas** usando varianza del Laplaciano (OpenCV)
- Archivos corruptos
- Duplicados usando **perceptual hash**

### **4. Preprocesamiento**
Transformaciones definidas con Torchvision:
- Redimensionado
- Normalización estilo ImageNet
- Conversión de color
- Tensores listos para modelos

### **5. Data Augmentation**
Aumentos incluidos:
- ColorJitter
- Rotaciones
- Flips horizontales
- RandomResizedCrop

### **6. División estratificada sin sklearn**
Split 80/20 reproducible usando NumPy.

Archivos generados:
```
data/qc/inventory_full.csv
data/qc/train_list.csv
data/qc/val_list.csv
data/qc/to_review_flags.csv
data/qc/possible_duplicates.csv
```

---

## ▶ ¿Cómo ejecutar el proyecto?

### **En Google Colab**
1. Sube la carpeta `data/raw` con las dos clases.
2. Abre el notebook `EDA_2clases_celebs.ipynb`.
3. Ejecuta la primera celda (instala dependencias automáticamente).
4. Ejecuta todo el notebook.

### **En local**
```bash
pip install -r requirements_2clases.txt
python -m ipykernel install --user
```

Luego abre el notebook desde Jupyter o VSCode.

---

## 🏁 Resultado final

Al ejecutar este proyecto tendrás:

✔ Dataset verificado y limpio  
✔ Imágenes defectuosas identificadas  
✔ CSVs para control de calidad  
✔ Dataset listo para entrenamiento en PyTorch  
✔ Flujo completamente compatible con proyectos de IA del curso  

---

## 📌 Autor

**Armando Cabrera**  
Universidad Latina de Panamá  
Materia: *Inteligencia Artificial*
