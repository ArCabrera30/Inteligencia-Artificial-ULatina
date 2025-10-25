
# Guardar métricas
tabla_sorted = tabla.sort_values(by="F1", ascending=False).reset_index(drop=True)
tabla_sorted.to_csv("metricas_clasificacion.csv", index=False)

best_name = tabla_sorted.iloc[0]["Modelo"]
best_pipe = pipes[best_name]

# ----------------------------
# Construcción de UI dinámica
# ----------------------------
def predict_fn(*vals):
    data = {}
    for c, v in zip(X.columns, vals):
        data[c] = v
    row = pd.DataFrame([data])
    proba = float(best_pipe.predict_proba(row)[0,1])
    pred = int(proba >= 0.5)
    label = "Positivo (1)" if pred==1 else "Negativo (0)"
    return (f"Predicción: {label}\n"
            f"Probabilidad clase 1: {proba:.3f}\n"
            f"Modelo seleccionado: {best_name}\n"
            f"Columna objetivo detectada: {target}")

# Crear entradas según tipo
inputs = []
for c in X.columns:
    if c in num_cols:
        default = float(X_train[c].median()) if c in X_train else 0.0
        inputs.append(gr.Number(label=f"{c}", value=default))
    else:
        # valores vistos en train (si son muchos, usa los más frecuentes)
        vals = X_train[c].astype(str).value_counts().index.tolist() if c in X_train else X[c].astype(str).unique().tolist()
        if len(vals) > 25:
            vals = vals[:25]
        inputs.append(gr.Dropdown(choices=vals, value=vals[0] if vals else None, label=f"{c}"))

demo = gr.Interface(
    fn=predict_fn,
    inputs=inputs,
    outputs=gr.Textbox(label="Resultado"),
    title="Clasificación Binaria — UI (Gradio)",
    description=(
        "Este demo entrena 5 modelos (LogReg, KNN, Árbol, RandomForest, SVM), calcula Accuracy, Precision, Recall, Specificity y F1, "
        "y selecciona automáticamente el **mejor por F1**. Introduce características para obtener la predicción."
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
