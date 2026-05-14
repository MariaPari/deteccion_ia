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

from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import cv2
import mediapipe as mp
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

semaforo = asyncio.Semaphore(5)  # menos carga = más estable

# =====================
# EMOCIONES
# =====================
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

# =====================
# MODELO EDAD / GENERO
# =====================
model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 18)
model.load_state_dict(torch.load(
    "res34_fair_align_multi_7_20190809.pt",
    map_location="cpu"
))
model.eval()

clase_genero = ['Masculino', 'Femenino']

clase_edad = [
    '0-2','3-9','10-19','20-29','30-39',
    '40-49','50-59','60-69','70+'
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# MEDIAPIPE FACE DETECTION
# =====================
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)


def detectar_rostros(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.process(img_rgb)

    rostros = []

    if results.detections:
        h, w, _ = img.shape

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(w,x2), min(h,y2)

            rostro = img[y1:y2, x1:x2]

            if rostro.size == 0:
                continue

            rostro = cv2.resize(rostro, (224,224))

            rostros.append({
                "rostro": rostro,
                "coords": (x1,y1,x2,y2)
            })

    return rostros, len(rostros)


def detectar_emocion(rostro):
    rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    result = emotion_pipe(pil)
    label = max(result, key=lambda x: x["score"])["label"]

    return emociones_es.get(label, label)


def analizar_rostros(rostros):
    resultados = []

    for r in rostros:
        rostro = r["rostro"]

        emocion = detectar_emocion(rostro)

        img = Image.fromarray(cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB))
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)

            genero = clase_genero[torch.argmax(outputs[:,7:9]).item()]
            edad = clase_edad[torch.argmax(outputs[:,9:18]).item()]

        resultados.append({
            "genero": genero,
            "edad": edad,
            "emocion": emocion
        })

    return resultados


@app.post("/analizar")
async def analizar(
    files: List[UploadFile] = File(...),
    dispositivo: str = Form(...),
    timestamp: str = Form(...)
):

    async with semaforo:
        salida = []

        for file in files:
            contents = await file.read()
            npimg = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if img is None:
                continue

            rostros, cantidad = detectar_rostros(img)
            resultados = analizar_rostros(rostros)

            salida.append({
                "imagen": file.filename,
                "cantidad_personas": cantidad,
                "resultados": resultados,
                "dispositivo": dispositivo,
                "timestamp": timestamp
            })

        return salida