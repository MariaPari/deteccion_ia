#-----------------------------------------------------------------------------------------
#*****************************************************************************************
#--> SERVIDOR CENTRAL DE RECEPCION Y ANALISIS DE IMAGENES (FASTAPI)
#*****************************************************************************************
# Descripcion:
# Servidor central encargado de recibir imagenes enviadas desde multiples dispositivos
# ESP32-CAM mediante solicitudes HTTP.
#
# El sistema permite conexiones concurrentes y controla la cantidad maxima de solicitudes
# simultaneas mediante un semaforo asincrono, el servidor procesa hasta 10 solicitudes al
# mismo tiempo, las solicitudes adicionales permanecen en espera hasta que existan recursos
# disponibles.
#
# Cada solicitud incluye:
#   - Imagen capturada por el dispositivo
#   - Identificador del dispositivo
#   - Timestamp de envio
#
# Funciones principales:
# 1. Recepcion de imagenes desde multiples ESP32-CAM
# 2. Registro de metadata asociada al dispositivo
# 3. Deteccion de rostros en la imagen
# 4. Analisis de atributos faciales:
#       - Numero de personas detectadas
#       - Genero
#       - Rango de edad
#       - Emocion predominante
# 5. Generacion de respuesta en formato JSON
# 6. Envio de datos procesados hacia el siguiente servidor  
#-----------------------------------------------------------------------------------------

#**************
#--> LIBRERIAS
#**************
from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from insightface.model_zoo import get_model
from typing import List 
import onnxruntime as ort
import os
import urllib.request

app = FastAPI()
@app.get("/")
def inicio():
    return {"mensaje": "API funcionando correctamente."}

detector_rostros = None
emotion_session = None
age_gender_session = None
emotion_input_name = None
age_gender_input_name = None

@app.on_event("startup")
async def cargar_modelos():

    global detector_rostros
    global emotion_session
    global age_gender_session
    global emotion_input_name
    global age_gender_input_name

    opciones = ort.SessionOptions()
    opciones.intra_op_num_threads = 1
    opciones.inter_op_num_threads = 1

    # SCRFD
    detector_rostros = get_model(
        "scrfd_2.5g_bnkps.onnx",
        providers=['CPUExecutionProvider']
    )

    detector_rostros.prepare(
        ctx_id=0,
        input_size=(416,416)
    )

    # EMOCIONES ONNX
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    emotion_dir = os.path.join(
        BASE_DIR,
        "emotion_onnx"
    )

    os.makedirs(
        emotion_dir,
        exist_ok=True
    )

    emotion_model_path = os.path.join(
        emotion_dir,
        "model.onnx"
    )

    # DESCARGAR MODELO SI NO EXISTE
    if not os.path.exists(emotion_model_path):

        print("Descargando modelo de emociones...")

        urllib.request.urlretrieve(
            "https://huggingface.co/stephaniePP/emotion-model/resolve/main/model.onnx",
            emotion_model_path
        )

        print("Modelo descargado")

    emotion_session = ort.InferenceSession(
        emotion_model_path,
        sess_options=opciones,
        providers=["CPUExecutionProvider"]
    )

    # EDAD Y GENERO ONNX
    age_gender_model_path = os.path.join(
        BASE_DIR,
        "age_gender_single.onnx"
    )

    # DESCARGAR SI NO EXISTE
    if not os.path.exists(age_gender_model_path):

        print("Descargando modelo edad/genero...")

        urllib.request.urlretrieve(
            "https://drive.google.com/uc?id=13bZAey4vWefYmNFWC8hD5F7cqSYFe3bK",
            age_gender_model_path
        )

        print("Modelo edad/genero descargado")

    age_gender_session = ort.InferenceSession(
        age_gender_model_path,
        sess_options=opciones,
        providers=["CPUExecutionProvider"]
    )

    emotion_input_name = emotion_session.get_inputs()[0].name

    age_gender_input_name = age_gender_session.get_inputs()[0].name
    print("MODELOS ONNX CARGADOS")

emociones_es = {
    "happy": "feliz",
    "sad": "triste",
    "neutral": "neutral",
    "angry": "enojado",
    "fear": "miedo",
    "surprise": "sorpresa",
    "disgust": "asco"
}

#--> CLASES
clase_genero = ['Masculino', 'Femenino']

clase_edad = [
    '0-2',
    '3-9',
    '10-19',
    '20-29',
    '30-39',
    '40-49',
    '50-59',
    '60-69',
    '70+'
]

def preprocess_image(img_bgr):

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img_rgb,
        (224,224)
    )

    img = img.astype(np.float32) / 255.0

    mean = np.array(
        [0.485,0.456,0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229,0.224,0.225],
        dtype=np.float32
    )

    img = (img - mean) / std

    img = np.transpose(
        img,
        (2,0,1)
    )

    img = np.expand_dims(
        img,
        axis=0
    )

    return img

#**********************************
#--> FUNCION PARA DETECTAR ROSTROS 
#**********************************
def detectar_rostros(imagen):

    bboxes, kpss = detector_rostros.detect(
        imagen,
        max_num=0
    )

    #--> LISTA DE ROSTROS
    rostros_recortados = []

    # Tamaño uniforme
    TAMANO = (224, 224)

    if bboxes is None:
        return [], 0
    
    #--> RECORRER ROSTROS
    for bbox in bboxes:

        x1, y1, x2, y2 = bbox[:4].astype(int)
        score = bbox[4]

        # Filtrar detecciones débiles
        if score < 0.5:
            continue

        # Verificar límites
        if x1 < 0 or y1 < 0 or x2 > imagen.shape[1] or y2 > imagen.shape[0]:
            continue

        # Recortar rostro
        rostro = imagen[y1:y2, x1:x2]

        # Evitar errores
        if rostro.size == 0:
            continue

        # Redimensionar
        rostro = cv2.resize(rostro, TAMANO)

        # Guardar rostro
        rostros_recortados.append({
            "rostro": rostro,
            "coords": (x1, y1, x2, y2)
        })

    cantidad = len(rostros_recortados)

    return rostros_recortados, cantidad
    
#*****************************************************
#--> FUNCION PARA DETECTAR LA EMOCION DE CADA PERSONA
#*****************************************************
def detectar_emocion(rostro_bgr):

    try:

        rostro = preprocess_image(rostro_bgr)

        input_name = emotion_input_name

        outputs = emotion_session.run(
            None,
            {
                input_name: rostro
            }
        )

        logits = outputs[0]

        predicted_class = np.argmax(
            logits,
            axis=-1
        )[0]

        labels = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise"
        ]

        emocion = labels[predicted_class]

        return emocion

    except:
        return "neutral"

#********************************************************************
#--> FUNCION PARA DETECTAR LA EDAD Y GENERO DE CADA ROSTRO DETECTADO
#********************************************************************
def detectar_edad_genero(rostros_recortados):
    resultados = []
    for dato in rostros_recortados:
        rostro = dato["rostro"]
        emocion_en = detectar_emocion(rostro)
        emocion = emociones_es.get(emocion_en, emocion_en)

        img = preprocess_image(rostro)

        # PREDICCION --> EDAD Y GENERO
        input_name = age_gender_input_name

        outputs = age_gender_session.run(
            None,
            {
                input_name: img
            }
        )[0]

        # GENERO
        salida_genero = outputs[:, 7:9]
        genero_predecido = np.argmax(salida_genero, axis=1)
        genero = clase_genero[genero_predecido.item()]

        # EDAD
        salida_edad = outputs[:, 9:18]
        edad_predecida = np.argmax(salida_edad, axis=1)
        edad = clase_edad[edad_predecida.item()]

        resultados.append({
            "genero": genero,
            "edad": edad,
            "emocion": emocion
        })
    return resultados

#************************
#--> PROGRAMA PRINCIPAL 
#************************
@app.post("/analizar")
async def analizar(files: List[UploadFile] = File(...)):

    resultados_totales = []

    for file in files:

        contents = await file.read()

        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            continue

        MAX_SIZE = 1280
        h, w = img.shape[:2]
        if max(h, w) > MAX_SIZE:
            scale = MAX_SIZE / max(h, w)

            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale))
            )

        rostros_recortados, cantidad = detectar_rostros(img)

        resultados = detectar_edad_genero(rostros_recortados)

        if cantidad == 0:
            resultados_totales.append({
                "imagen": file.filename,
                "cantidad_personas": 0,
                "mensaje": "No se detectaron rostros",
                "resultados": []
            })
        else:
            resultados_totales.append({
                "imagen": file.filename,
                "cantidad_personas": cantidad,
                "resultados": resultados
            })
    return resultados_totales