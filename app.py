import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuración inicial de la interfaz
st.set_page_config(page_title="Clasificador de Flores IA", layout="centered")

# Encabezado personalizado con tus datos para la entrega formal de la tarea
st.title("🌻 Modelo predictivo de Flores - Clase de Inteligencia Artificial")
st.subheader("Estudiante: Edy Yandel Martinez Tejeda - Cuenta: 20221900004")
st.write("Suba una imagen para clasificar las flores con el modelo MobileNetV2 entrenado")

# Dimensiones estándar requeridas por MobileNetV2
IMG_SIZE = (224, 224)

# Configuración de las rutas relativas en la raíz de tu repositorio de GitHub
MODEL_PATH = Path("flower_model.h5")
CLASS_PATH = Path("clases.txt")

# Diccionario de traducción para las 5 categorías del dataset Flowers Recognition de Kaggle
LABELS_ES = {
    "daisy": "Margarita 🌼",
    "dandelion": "Diente de León 🌾",
    "rose": "Rosa 🌹",
    "sunflower": "Girasol 🌻",
    "tulip": "Tulipán 🌷"
}

@st.cache_resource
def cargar_modelo():
    if MODEL_PATH.exists():
        # Cargamos el archivo .h5 generado en el entrenamiento
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.error("No se encontró el modelo. Asegúrate de tener 'flower_model.h5' en la misma carpeta que app.py.")
    st.stop()

@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        # Leer el archivo clases.txt línea por línea (daisy, dandelion, rose, sunflower, tulip)
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return [line.strip().lower() for line in f.readlines()]
    # Clases por defecto (en minúsculas según la estructura de carpetas de Kaggle)
    return ["daisy", "dandelion", "rose", "sunflower", "tulip"]

def preparar_imagen(img):
    # Convertir a RGB por si tiene transparencias (PNG) y redimensionar a 224x224
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # Escalado de píxeles dividiendo entre 255.0, idéntico al ImageDataGenerator del entrenamiento
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    
    # Ordenar las predicciones de mayor a menor probabilidad
    top_indices = np.argsort(preds)[::-1]
    
    return [
        (LABELS_ES.get(clases[i], clases[i].capitalize()), float(preds[i]) * 100)
        for i in top_indices
    ]

# Cargar el modelo y las clases en memoria utilizando la caché de Streamlit
modelo = cargar_modelo()
clases = cargar_clases()

# Componente interactivo para cargar archivos en la web
archivo = st.file_uploader("Seleccione una imagen de una flor", type=["jpg", "jpeg", "png"])

if archivo:
    # Cargar y desplegar la imagen en la interfaz gráfica
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen analizada", use_container_width=True)

    # Realizar el análisis con la red neuronal
    resultados = predecir(imagen)
    
    st.subheader("Resultado del Análisis")
    # Mostrar la predicción ganadora con un recuadro verde de éxito (Tu clase de mayor porcentaje)
    st.success(f"Predicción principal: **{resultados[0][0]}** ({resultados[0][1]:.2f}%)")

    # Mostrar la tabla completa comparativa de probabilidades de la IA
    st.write("### Probabilidades del modelo:")
    for clase, prob in resultados:
        # Barra de progreso visual para mejorar el diseño de la entrega de cara al docente
        st.write(f"**{clase}**: {prob:.2f}%")
        st.progress(int(prob) / 100)
else:
    st.info("Cargue una imagen de una flor (Margarita, Diente de león, Rosa, Girasol o Tulipán) para iniciar la clasificación.")
