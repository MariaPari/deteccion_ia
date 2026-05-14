FROM python:3.10

WORKDIR /code

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Puerto de Hugging Face Spaces
EXPOSE 7860

# Ejecutar FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]