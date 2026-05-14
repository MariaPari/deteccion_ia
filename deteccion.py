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
from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import cv2
from retinaface import RetinaFace
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from PIL import Image
from transformers import pipeline
from typing import List
import asyncio

app = FastAPI()
@app.get("/")
def inicio():
    return {"mensaje": "API funcionando correctamente"}
# Maximo 10 solicitudes simultaneas
semaforo = asyncio.Semaphore(10)

#*********************
# MODELO DE EMOCIONES
#*********************
emotion_pipe = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression"
)
emociones_es = {
    "happy": "feliz",
    "sad": "triste",
    "neutral": "neutral",
    "angry": "enojado",
    "fear": "miedo",
    "surprise": "sorpresa",
    "disgust": "asco"
}

#*************************
# MODELO DE EDAD Y GENERO
#*************************
model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 18)
model.load_state_dict(torch.load('res34_fair_align_multi_7_20190809.pt',map_location=torch.device('cpu')))
model.eval()

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

#--> TRANSFORMACIONES
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#**********************************
#--> FUNCION PARA DETECTAR ROSTROS 
#**********************************
def detectar_rostros(imagen):
    resultados = RetinaFace.detect_faces(imagen)

    #--> LISTA DE ROSTROS
    rostros_recortados = []

    # Tamano uniforme
    TAMANO = (224, 224)

    #--> RECORRER ROSTROS
    if isinstance(resultados, dict):       # detectamos si hay rostros o no dentro de la imagen

        for clave in resultados:

            # Obtener coordenadas
            x1, y1, x2, y2 = resultados[clave]['facial_area']

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

    return rostros_recortados,cantidad
    
#*****************************************************
#--> FUNCION PARA DETECTAR LA EMOCION DE CADA PERSONA
#*****************************************************
def detectar_emocion(rostro_bgr):
    rostro_rgb = cv2.cvtColor(rostro_bgr, cv2.COLOR_BGR2RGB)
    rostro_pil = Image.fromarray(rostro_rgb)

    result = emotion_pipe(rostro_pil)

    emocion = max(result, key=lambda x: x["score"])["label"]

    return emocion

#********************************************************************
#--> FUNCION PARA DETECTAR LA EDAD Y GENERO DE CADA ROSTRO DETECTADO
#********************************************************************
def detectar_edad_genero(rostros_recortados):
    resultados = []
    for dato in rostros_recortados:
        rostro = dato["rostro"]
        emocion_en = detectar_emocion(rostro)
        emocion = emociones_es.get(emocion_en, emocion_en)

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)

        # PIL
        img = Image.fromarray(frame_rgb)

        # Transformar
        img = transform(img)
        img = img.unsqueeze(0)

        # PREDICCION --> EDAD Y GENERO
        with torch.no_grad():

            outputs = model(img)

            # GENERO
            salida_genero = outputs[:, 7:9]
            genero_predecido = torch.argmax(salida_genero, dim=1)
            genero = clase_genero[genero_predecido.item()]

            # EDAD
            salida_edad = outputs[:, 9:18]
            edad_predecida = torch.argmax(salida_edad, dim=1)
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
async def analizar(files: List[UploadFile] = File(...),dispositivo: str = Form(...),timestamp: str = Form(...),):

    async with semaforo:
        resultados_totales = []

        for file in files:

            contents = await file.read()

            npimg = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if img is None:
                continue

            rostros_recortados, cantidad = detectar_rostros(img)

            resultados = detectar_edad_genero(rostros_recortados)

            if cantidad == 0:
                resultados_totales.append({
                    "imagen": file.filename,
                    "cantidad_personas": 0,
                    "mensaje": "No se detectaron rostros",
                    "resultados": [],
                    "dispositivo": dispositivo,
                    "timestamp":timestamp
                })
            else:
                resultados_totales.append({
                    "imagen": file.filename,
                    "cantidad_personas": cantidad,
                    "resultados": resultados,
                    "dispositivo": dispositivo,
                    "timestamp":timestamp
                })
        return resultados_totales