# ================================================================
# Implementación del modelo BART:
# Denoising Sequence-to-Sequence Pre-training for Natural Language
# Generation, Translation, and Comprehension
#
# ASIGNATURA:
# Procesamiento de Datos Secuenciales con Deep Learning
#
# ESTUDIANTES:
# - Juan Camilo Perez
# - Carlos Eduardo Trujillo
# - Osvaldo Marin
#
# DESCRIPCIÓN:
# Aplicación web desarrollada con Streamlit para generar resúmenes
# abstractivos utilizando el modelo preentrenado facebook/bart-large-cnn disponible en Hugging Face.
#
# REFERENCIA: 
# Este codigo fue desarrollado tomando como referencia el articulo base de BART
# y utilizando ChatGPT como principal herramienta de apoyo.

# [1] M. Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training
#     for Natural Language Generation, Translation, and Comprehension,"
#     ACL, 2020.
#
# [2] OpenAI, "ChatGPT," 2026. Utilizado como herramienta de apoyo para
#     desarrollo y adaptacion del codigo.
# ================================================================


# 1. IMPORTACIÓN DE LIBRERÍAS
# Se realiza la importacion de las librerias, Streamlit se utiliza para 
# la creacion de la interfaz web interactiva, Pytorch se utilizacomo framework 
# para implementar el modelo y Transformers carga el modelo pre-entrenado y el tokenizer 
# de hugging face.

import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# 2. CONFIGURACION DEL MODELO
#  se carga el modelo facebook/bart-large-cnn el cual esta ajustado
# específicamente para tareas de resumen abstractivo de noticias.

MODEL_NAME = "facebook/bart-large-cnn"

#3. CARGA DEL TOKENIZER Y DEL MODELO
# El cache permite almacenar el modelo en memoria para no tener que
# cargarlo cada vez que el usuario genere un resumen.

@st.cache_resource
def cargar_modelo():

    #Carga del tokenizer, convierte el texto en vectores
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    # Carga del modelo, se cargan los pesos pre-entrenados
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Selección del dispositivo, se selecciona CPU o GPU dependiendo de la disponibilidad
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device

# 4. GENERAR EL RESUMEN

def generar_resumen(texto, palabras_objetivo):

    # Se obtienen el tokenizer, el modelo y el dispositivo.
    tokenizer, model, device = cargar_modelo()

    # BART controla la longitud en tokens, no en palabras por lo cual se debe realizar la transformacion.
    # por esto se construye un rango aproximado alrededor de la cantidad de palabras seleccionadas por el 
    # usuario: 
    # Mínimo: aproximadamente el 80 %.
    # Máximo: aproximadamente el 130 %.
    min_tokens = max(10, int(palabras_objetivo * 0.8))
    max_tokens = max(min_tokens + 5, int(palabras_objetivo * 1.3))

    # El tokenizer transforma el texto en una representación
    # numérica que puede procesar el Transformer.
    inputs = tokenizer(
        texto,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(device)

    with torch.inference_mode():
        summary_ids = model.generate(
            # Tokens correspondientes al texto original.
            inputs["input_ids"], 
            # Máscara utilizada por el mecanismo de atención, indica posiciones con informacion relevante. 
            attention_mask=inputs["attention_mask"],
            # Longitud máxima aproximada del resumen.
            max_length=max_tokens,
            # Longitud mínima aproximada del resumen.
            min_length=min_tokens,
            # algoritmo para definir el numero de salidas parciales que genera el modelo antes de elegir una final.
            num_beams=6,
            # Ayuda a controlar el tamaño de las secuencias seleccionadas durante el Beam Search
            length_penalty=2.0,
            # Evita repetir secuencias idénticas
            no_repeat_ngram_size=3,
            # Detiene Beam Search cuando las secuencias candidatas han alcanzado una condición optima.
            early_stopping=True
        )
    # Decodificacion de la salida, es decir pasa de vectores nuevamente a textounicamente la secuencia final seleccionada en el Beam Search
    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

# 5. CONFIGURACIÓN DE LA INTERFAZ STREAMLIT

# Configuracion de caracteristicas generales
# Nombre, Icono, Diseño
st.set_page_config(
    page_title="Generador de Resúmenes con BART",
    page_icon="📝",
    layout="centered"
)

# 6. ENCABEZADO DE LA APLICACIÓN

st.title("📝 Generador de Resúmenes con BART")
st.caption("Modelo: facebook/bart-large-cnn")

# 7. FORMULARIO DE ENTRADA
# Se crea un formulario que agrupa el texto a resumir, tamaño del resumen y boton para ejecutar la inferencia
with st.form("form_resumen"):
    texto = st.text_area(
        "Texto original",
        placeholder="Ingrese el texto en inglés que desea resumir...",
        height=300
    )

    palabras_objetivo = st.slider(
        "Tamaño aproximado del resumen (palabras)",
        min_value=20,
        max_value=120,
        value=50,
        step=5
    )

    generar = st.form_submit_button(
        "Generar resumen",
        use_container_width=True
    )

# 8. VALIDACIÓN DE LA ENTRADA Y GENERACIÓN DEL RESUMEN
# Se ejecuta al presionar el boton de generar resumen, se encarga de: 
# Validar que haya texto de entrada.
# Que el resumen solicitado sea mas corto que el texto.
# Crea un mensaje de espera mientras el modelo esta corriendo.
# Muestra el resumen generado.
# Muestra Estadisticas de el numero original de palabras vs el numero de palabras del resumen.

if generar:
    if not texto.strip():
        st.warning("Ingrese un texto para generar el resumen.")

    elif palabras_objetivo >= len(texto.split()):
        st.warning("El resumen debe ser más corto que el texto original.")

    else:
        with st.spinner("Generando resumen..."):
            resumen = generar_resumen(texto, palabras_objetivo)

        st.subheader("Resumen generado")
        st.write(resumen)

        col1, col2 = st.columns(2)
        col1.metric("Palabras originales", len(texto.split()))
        col2.metric("Palabras del resumen", len(resumen.split()))
