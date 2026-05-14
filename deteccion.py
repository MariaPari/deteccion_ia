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
#--> LIBRERÍAS
#**************
from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from PIL import Image
from transformers import pipeline
from typing import List
import asyncio
import mediapipe as mp

#*********************
#--> FASTAPI APP
#*********************
app = FastAPI()

@app.get("/")
def inicio():
    return {"mensaje": "API funcionando correctamente"}

#********************************
#--> CONTROL DE CONCURRENCIA
#********************************
semaforo = asyncio.Semaphore(10)

#********************************
#--> MODELO DE EMOCIONES
#********************************
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

#********************************
#--> MODELO EDAD Y GÉNERO
#********************************
model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 18)
model.load_state_dict(torch.load(
    'res34_fair_align_multi_7_20190809.pt',
    map_location=torch.device('cpu')
))
model.eval()

clase_genero = ['Masculino', 'Femenino']

clase_edad = [
    '0-2','3-9','10-19','20-29','30-39',
    '40-49','50-59','60-69','70+'
]

#********************************
#--> TRANSFORMACIONES
#********************************
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#********************************
#--> MEDIAPIPE FACE DETECTION
#********************************
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

#********************************
#--> DETECCIÓN DE ROSTROS (MEDIA PIPE)
#********************************
def detectar_rostros(imagen):
    rostros_recortados = []
    TAMANO = (224, 224)

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = detector.process(imagen_rgb)

    if resultados.detections:

        h, w, _ = imagen.shape

        for det in resultados.detections:
            bbox = det.location_data.relative_bounding_box

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            # límites
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            rostro = imagen[y1:y2, x1:x2]

            if rostro.size == 0:
                continue

            rostro = cv2.resize(rostro, TAMANO)

            rostros_recortados.append({
                "rostro": rostro,
                "coords": (x1, y1, x2, y2)
            })

    return rostros_recortados, len(rostros_recortados)

#********************************
#--> EMOCIÓN
#********************************
def detectar_emocion(rostro_bgr):
    rostro_rgb = cv2.cvtColor(rostro_bgr, cv2.COLOR_BGR2RGB)
    rostro_pil = Image.fromarray(rostro_rgb)

    result = emotion_pipe(rostro_pil)

    emocion = max(result, key=lambda x: x["score"])["label"]

    return emocion

#********************************
#--> EDAD Y GÉNERO
#********************************
def detectar_edad_genero(rostros_recortados):
    resultados = []

    for dato in rostros_recortados:
        rostro = dato["rostro"]

        emocion_en = detectar_emocion(rostro)
        emocion = emociones_es.get(emocion_en, emocion_en)

        frame_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        img = transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)

            # GÉNERO
            salida_genero = outputs[:, 7:9]
            genero_idx = torch.argmax(salida_genero, dim=1).item()
            genero = clase_genero[genero_idx]

            # EDAD
            salida_edad = outputs[:, 9:18]
            edad_idx = torch.argmax(salida_edad, dim=1).item()
            edad = clase_edad[edad_idx]

        resultados.append({
            "genero": genero,
            "edad": edad,
            "emocion": emocion
        })

    return resultados

#********************************
#--> ENDPOINT PRINCIPAL
#********************************
@app.post("/analizar")
async def analizar(
    files: List[UploadFile] = File(...),
    dispositivo: str = Form(...),
    timestamp: str = Form(...)
):

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
                    "timestamp": timestamp
                })
            else:
                resultados_totales.append({
                    "imagen": file.filename,
                    "cantidad_personas": cantidad,
                    "resultados": resultados,
                    "dispositivo": dispositivo,
                    "timestamp": timestamp
                })

        return resultados_totales

#********************************
#--> START SERVER (RENDER FIX)
#********************************
if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "deteccion:app",
        host="0.0.0.0",
        port=port
    )