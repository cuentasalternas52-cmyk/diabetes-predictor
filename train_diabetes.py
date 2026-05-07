import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ====================================
# NOMBRES DE LAS COLUMNAS
# ====================================

columnas = [
    'Embarazos',
    'Glucosa',
    'PresionArterial',
    'GrosorPiel',
    'Insulina',
    'IMC',
    'FactorHereditario',
    'Edad',
    'Resultado'
]

# ====================================
# CARGAR DATASET
# ====================================

df = pd.read_csv(
    'diabetes.csv',
    header=None,
    names=columnas
)

print("\nDATASET CARGADO CORRECTAMENTE\n")
print(df.head())

# ====================================
# VARIABLES
# ====================================

X = df.drop('Resultado', axis=1)
y = df['Resultado']

# ====================================
# DIVIDIR DATOS
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ====================================
# ESCALAR DATOS
# ====================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================================
# REGRESIÓN LOGÍSTICA
# ====================================

print("\nENTRENANDO MODELO DE REGRESIÓN LOGÍSTICA...\n")

modelo_logistico = LogisticRegression(max_iter=1000)

modelo_logistico.fit(X_train_scaled, y_train)

# ====================================
# RED NEURONAL
# ====================================

print("ENTRENANDO MODELO DE RED NEURONAL...\n")

modelo_red_neuronal = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    max_iter=1000,
    random_state=42
)

modelo_red_neuronal.fit(X_train_scaled, y_train)

# ====================================
# PREDICCIONES
# ====================================

pred_logistica = modelo_logistico.predict(X_test_scaled)

pred_red_neuronal = modelo_red_neuronal.predict(X_test_scaled)

# ====================================
# ACCURACY
# ====================================

accuracy_logistica = accuracy_score(
    y_test,
    pred_logistica
)

accuracy_red = accuracy_score(
    y_test,
    pred_red_neuronal
)

print("===================================")
print("RESULTADOS")
print("===================================\n")

print(
    f"Precisión Regresión Logística: "
    f"{accuracy_logistica:.2%}"
)

print(
    f"Precisión Red Neuronal: "
    f"{accuracy_red:.2%}"
)

# ====================================
# MATRIZ DE CONFUSIÓN
# ====================================

print("\nMATRIZ DE CONFUSIÓN - RED NEURONAL\n")

matriz = confusion_matrix(
    y_test,
    pred_red_neuronal
)

print(matriz)

# ====================================
# REPORTE DE CLASIFICACIÓN
# ====================================

print("\nREPORTE DE CLASIFICACIÓN\n")

print(
    classification_report(
        y_test,
        pred_red_neuronal
    )
)

# ====================================
# GUARDAR MODELOS
# ====================================

joblib.dump(
    modelo_logistico,
    'log_model.pkl'
)

joblib.dump(
    modelo_red_neuronal,
    'nn_model.pkl'
)

joblib.dump(
    scaler,
    'scaler_diabetes.pkl'
)

print("\nMODELOS GUARDADOS CORRECTAMENTE")