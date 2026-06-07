import streamlit as st
import numpy as np
import os


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Mensaje en la carga inicial <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
modelo_ya_estaba = "modelo_cargado" in st.session_state
titulo_carga = st.empty()
if not modelo_ya_estaba:
    titulo_carga.title("⏳ Cargando... un momento.")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Quitar warnings en consola <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
import joblib
import random as rn
import pandas as pd

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo():
    scaler = joblib.load("iris_scaler.pkl")
    model = tf.keras.models.load_model("iris_model.keras")
    return scaler, model

scaler, model = cargar_modelo()
st.session_state.modelo_cargado = True
titulo_carga.empty()

clases = ["setosa", "versicolor", "virginica"]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# -------------------
# Funciones
# -------------------

def rnd(minimo, maximo):
    return round(rn.uniform(minimo, maximo), 1)

# -------------------
# Título
# -------------------

st.title("Clasificador de Flores Iris")

# -------------------
# Botón generar flor
# -------------------

if st.button("🌸 Generar flor aleatoria"):
    st.session_state.sl = rnd(4.1, 8.1)
    st.session_state.sw = rnd(1.8, 4.6)
    st.session_state.pl = rnd(0.8, 7.1)
    st.session_state.pw = rnd(0.1, 2.7)

# -------------------
# Inputs
# -------------------


col_sepal, col_petal = st.columns(2)

with col_sepal:
    sl = st.number_input("Sepal Length", min_value=4.1, max_value=8.1, step=0.1,
                         value=st.session_state.get("sl", 5.1))
    sw = st.number_input("Sepal Width",  min_value=1.8, max_value=4.6, step=0.1,
                         value=st.session_state.get("sw", 3.5))

with col_petal:
    pl = st.number_input("Petal Length", min_value=0.8, max_value=7.1, step=0.1,
                         value=st.session_state.get("pl", 1.4))
    pw = st.number_input("Petal Width",  min_value=0.1, max_value=2.7, step=0.1,
                         value=st.session_state.get("pw", 0.2))


# -------------------
# Predicción
# -------------------

if st.button("🔍 Evaluar flor"):

    flor = np.array([[sl, sw, pl, pw]])

    flor_scaled = scaler.transform(flor)

    probs = model.predict(flor_scaled, verbose=0)

    pred = probs.argmax(axis=1)[0]

    st.success(
        f"Especie: {clases[pred]}"
    )

    st.write(
        f"Confianza: {probs[0][pred] * 100:.2f}%"
    )

    df = pd.DataFrame({
        "Especie": clases,
        "Probabilidad": probs[0]
    })

    st.bar_chart(df.set_index("Especie"))