# Usa una imagen base de Python y agrega ffmpeg
FROM python:3.12.10-slim
WORKDIR /app
RUN apt-get clean && apt-get update &&  \ 
apt-get install -y build-essential libglib2.0-0 libgl1-mesa-glx libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 && rm -rf /var/lib/apt/lists/* &&\
    pip install --upgrade pip
 
 
COPY /requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade -r /requirements.txt
 
COPY . .
 
CMD [ "uvicorn","main:app","--reload","--port","80","--host","0.0.0.0"]
 
