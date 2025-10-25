
# app_gradio_regresion.py
# --------------------------------------------------
# UI (Gradio) para predecir el costo de seguro médico (Regresión)
# Cumple con:
# - Carga de insurance.csv (o genera datos sintéticos compatibles)
# - División 80/20 estratificada por cuantiles del objetivo
# - Entrenamiento y comparación de 4 modelos (Linear, Ridge, Lasso, RandomForest)
# - Cálculo de métricas: MAPE, MSE, RMSE y MAE
# - Selección automática del mejor modelo por RMSE
# - Interfaz para ingresar variables y obtener la predicción
# --------------------------------------------------
import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# 1) Cargar datos
# =========================
path = Path("insurance.csv")
if path.exists():
    df = pd.read_csv(path)
    origen = "original Kaggle"
else:
    # Generar datos sintéticos compatibles si no existe el archivo
    np.random.seed(42)
    n = 1338
    df = pd.DataFrame({
        'age': np.random.randint(18, 65, n),
        'sex': np.random.choice(['male','female'], n),
        'bmi': np.round(np.random.normal(30, 6, n), 2),
        'children': np.random.randint(0, 5, n),
        'smoker': np.random.choice(['yes','no'], n, p=[0.2,0.8]),
        'region': np.random.choice(['southwest','southeast','northwest','northeast'], n),
    })
    base = 2000 + df['age']*50 + (df['bmi']-25)*100 + df['children']*400
    fum = np.where(df['smoker']=='yes', 12000, 0)
    region_effect = df['region'].map({
        'southwest': -200,
        'southeast': 300,
        'northwest': -100,
        'northeast': 0
    }).values
    ruido = np.random.normal(0, 2000, n)
    df['charges'] = np.maximum(1000, base + fum + region_effect + ruido)
    origen = "sintético (auto-generado)"

# =========================
# 2) División 80/20 estratificada
# =========================
df_split = df.copy()
bins = pd.qcut(df_split['charges'], q=5, duplicates='drop')
df_split['charges_bin'] = bins.cat.codes

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(df_split, df_split['charges_bin']):
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

# =========================
# 3) Preprocesamiento y modelos
# =========================
num_features = ['age','bmi','children']
cat_features = ['sex','smoker','region']

preproc = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

modelos = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001, max_iter=10000),
    "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42)
}

X_train, y_train = train.drop(columns=['charges']), train['charges']
X_test, y_test = test.drop(columns=['charges']), test['charges']

resultados = []
pipes = {}

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100.0

for nombre, modelo in modelos.items():
    pipe = Pipeline([("prep", preproc), ("model", modelo)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape_v = mape(y_test.values, y_pred)
    resultados.append([nombre, mape_v, mse, rmse, mae])
    pipes[nombre] = pipe

tabla = pd.DataFrame(resultados, columns=["Modelo","MAPE","MSE","RMSE","MAE"]).sort_values("RMSE")
mejor_nombre = tabla.iloc[0]["Modelo"]
mejor_pipe = pipes[mejor_nombre]

# Guardar la tabla como referencia (opcional)
tabla.to_csv("metricas_regresion.csv", index=False)

# =========================
# 4) Interfaz Gradio
# =========================
def predecir(age, sex, bmi, children, smoker, region):
    data = pd.DataFrame([{
        "age": int(age),
        "sex": sex,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker,
        "region": region
    }])
    pred = float(mejor_pipe.predict(data)[0])
    return (f"Predicción de costo (USD): {pred:,.2f}\n"
            f"Modelo seleccionado: {mejor_nombre}\n"
            f"Fuente de datos: {origen}")

inputs = [
    gr.Number(label="Edad (años)", value=35),
    gr.Dropdown(["male","female"], label="Sexo", value="male"),
    gr.Number(label="BMI (Índice de Masa Corporal)", value=29.5),
    gr.Slider(0, 5, step=1, label="Hijos", value=1),
    gr.Dropdown(["yes","no"], label="Fumador", value="no"),
    gr.Dropdown(["southwest","southeast","northwest","northeast"], label="Región", value="northeast")
]

demo = gr.Interface(
    fn=predecir,
    inputs=inputs,
    outputs=gr.Textbox(label="Resultado"),
    title="Regresión — Costo de Seguro Médico",
    description=(
        "Este demo entrena 4 modelos (Linear, Ridge, Lasso, RandomForest), calcula MAPE, MSE, RMSE y MAE, "
        "y selecciona automáticamente el **mejor por RMSE**. "
        "Luego, puedes ingresar características para estimar el costo de seguro."
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
