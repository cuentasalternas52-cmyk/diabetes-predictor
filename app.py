import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# =====================================================
# CONFIGURACIÓN DE LA PÁGINA
# =====================================================

st.set_page_config(
    page_title="Predicción de Diabetes",
    page_icon="🩺",
    layout="centered"
)

# =====================================================
# CARGAR MODELOS
# =====================================================

log_model = joblib.load('log_model.pkl')
nn_model = joblib.load('nn_model.pkl')
scaler = joblib.load('scaler_diabetes.pkl')

# =====================================================
# TÍTULO
# =====================================================

st.title("🩺 Predicción de Diabetes")

st.write(
    """
    Sistema inteligente de predicción de diabetes utilizando
    Machine Learning y modelos de inteligencia artificial.
    """
)

# =====================================================
# SELECCIÓN DEL MODELO
# =====================================================

modelo = st.selectbox(
    "Seleccione el modelo de predicción",
    ["Regresión Logística", "Red Neuronal"]
)

if modelo == "Regresión Logística":
    modelo_seleccionado = log_model
else:
    modelo_seleccionado = nn_model

# =====================================================
# MENÚ LATERAL
# =====================================================

opcion = st.sidebar.radio(
    "Seleccione una opción",
    [
        "Predicción Individual",
        "Predicción por Lotes"
    ]
)

# =====================================================
# PREDICCIÓN INDIVIDUAL
# =====================================================

if opcion == "Predicción Individual":

    st.header("Ingrese los datos del paciente")

    col1, col2 = st.columns(2)

    with col1:

        embarazos = st.number_input(
            "Número de embarazos",
            min_value=0,
            max_value=20,
            value=1
        )

        glucosa = st.number_input(
            "Nivel de glucosa",
            min_value=0,
            max_value=300,
            value=120
        )

        presion = st.number_input(
            "Presión arterial",
            min_value=0,
            max_value=200,
            value=70
        )

        piel = st.number_input(
            "Grosor de piel",
            min_value=0,
            max_value=100,
            value=20
        )

    with col2:

        insulina = st.number_input(
            "Nivel de insulina",
            min_value=0,
            max_value=900,
            value=80
        )

        imc = st.number_input(
            "Índice de masa corporal (IMC)",
            min_value=0.0,
            max_value=70.0,
            value=25.0
        )

        herencia = st.number_input(
            "Factor hereditario de diabetes",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        edad = st.number_input(
            "Edad",
            min_value=1,
            max_value=120,
            value=30
        )

    # =================================================
    # BOTÓN PREDECIR
    # =================================================

    if st.button("🔍 Realizar Predicción"):

        datos = pd.DataFrame([[
            embarazos,
            glucosa,
            presion,
            piel,
            insulina,
            imc,
            herencia,
            edad
        ]], columns=[
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age'
        ])

        # Escalar datos
        datos_scaled = scaler.transform(datos)

        # Predicción
        prediccion = modelo_seleccionado.predict(datos_scaled)

        # Resultado
        st.subheader("Resultado de la Predicción")

        if prediccion[0] == 1:

            st.error(
                "⚠️ El modelo indica probabilidad de diabetes."
            )

        else:

            st.success(
                "✅ El modelo indica que NO hay probabilidad de diabetes."
            )

# =====================================================
# PREDICCIÓN POR LOTES
# =====================================================

elif opcion == "Predicción por Lotes":

    st.header("Predicción mediante archivo CSV")

    st.write(
        """
        Suba un archivo CSV con información de pacientes
        para realizar predicciones masivas.
        """
    )

    archivo = st.file_uploader(
        "Seleccione un archivo CSV",
        type=['csv']
    )

    if archivo is not None:

        # =============================================
        # LEER CSV
        # =============================================

        df = pd.read_csv(archivo)

        # Renombrar columnas automáticamente
        df.columns = [
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age',
            'Outcome'
        ]

        # =============================================
        # MOSTRAR DATOS
        # =============================================

        st.subheader("Datos cargados")

        st.dataframe(df)

        # =============================================
        # VARIABLES
        # =============================================

        X = df.drop('Outcome', axis=1)

        y_real = df['Outcome']

        # =============================================
        # ESCALAR DATOS
        # =============================================

        X_scaled = scaler.transform(X)

        # =============================================
        # PREDICCIONES
        # =============================================

        predicciones = modelo_seleccionado.predict(X_scaled)

        # =============================================
        # MOSTRAR RESULTADOS
        # =============================================

        df['Predicción'] = predicciones

        st.subheader("Resultados de Predicción")

        st.dataframe(df)

        # =============================================
        # MATRIZ DE CONFUSIÓN
        # =============================================

        cm = confusion_matrix(
            y_real,
            predicciones
        )

        st.subheader("Matriz de Confusión")

        st.write(cm)

        # =============================================
        # ACCURACY
        # =============================================

        accuracy = accuracy_score(
            y_real,
            predicciones
        )

        st.subheader("Precisión del Modelo")

        st.write(
            f"{round(accuracy * 100, 2)} %"
        )

        # =============================================
        # REPORTE DE CLASIFICACIÓN
        # =============================================

        reporte = classification_report(
            y_real,
            predicciones
        )

        st.subheader("Reporte de Clasificación")

        st.text(reporte)