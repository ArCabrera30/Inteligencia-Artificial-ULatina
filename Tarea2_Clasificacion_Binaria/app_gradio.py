
    # app_gradio.py
    # --------------------------------------------------
    # UI (Gradio) para clasificar células (malaria) con un modelo entrenado
    # Lee 'train.csv' de la misma carpeta, entrena RandomForest y
    # crea una interfaz con campos numéricos para predecir.
    # --------------------------------------------------

    import gradio as gr
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    # 1) Cargar datos
    df = pd.read_csv("train.csv")
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    # 2) Entrenar modelo (pipeline: escalado + Random Forest)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

    # 3) Función de predicción
    feature_names = list(X.columns)

    def predecir(*vals):
        # Convertir tupla de inputs a DataFrame con nombres de columnas
        row = pd.DataFrame([vals], columns=feature_names)
        prob = float(pipe.predict_proba(row)[0, 1])
        pred = int(prob >= 0.5)
        etiqueta = "Infectada" if pred == 1 else "No infectada"
        return f"Predicción: {etiqueta}  |  Probabilidad clase 'infectada': {prob:.3f}"

    # 4) Construir entradas dinámicas según columnas de X
    inputs = [gr.Number(label=col) for col in feature_names]

    demo = gr.Interface(
        fn=predecir,
        inputs=inputs,
        outputs=gr.Textbox(label="Resultado"),
        title="Clasificación binaria (Malaria) — Demo",
        description=(
            "Introduce valores numéricos para las características y obtén la predicción del modelo.
"
            "El modelo (RandomForest + StandardScaler) se entrena automáticamente con 'train.csv'."
        )
    )

    if __name__ == "__main__":
        # Ejecuta en local o en Colab. En Colab mostrará un enlace público temporal.
        demo.launch()
