from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse,HTMLResponse
from fastapi.staticfiles import StaticFiles
import uuid, os
import threading
import time
import warnings
warnings.filterwarnings("ignore")

from pydub import AudioSegment
from speechbrain.pretrained import SpeakerRecognition
from azure.storage.blob import BlobServiceClient

# Leer configuración desde variables de entorno
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY") 
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER") 

if not AZURE_STORAGE_ACCOUNT or not AZURE_STORAGE_KEY or not AZURE_STORAGE_CONTAINER:
    raise RuntimeError("Faltan variables de entorno de Azure Storage.")

# Crear el cliente usando la cuenta y la clave
blob_service = BlobServiceClient(
    account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
    credential=AZURE_STORAGE_KEY
)
container_client = blob_service.get_container_client(AZURE_STORAGE_CONTAINER)

# Crear la aplicación FastAPI
app = FastAPI()

# Montar la carpeta estática
app.mount("/static", StaticFiles(directory="static"), name="static")

def convert_to_wav(input_path):
    if input_path.lower().endswith((".wav", ".wave")):
        return input_path
    audio = AudioSegment.from_file(input_path)
    wav_path = input_path + ".temp.wav"
    audio.export(wav_path, format="wav")
    return wav_path

def compare_blobs(blob_url1, blob_url2, account, key, container):
    # Construye el connection string usando los parámetros
    conn_str = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={account};"
        f"AccountKey={key};"
        f"EndpointSuffix=core.windows.net"
    )
    # Descarga ambos blobs a archivos temporales
    local1 = "temp_voice1"
    local2 = "temp_voice2"
    from urllib.parse import urlparse
    def get_blob_name(blob_url):
        parsed = urlparse(blob_url)
        return '/'.join(parsed.path.split('/')[2:])
    blob_name1 = get_blob_name(blob_url1)
    blob_name2 = get_blob_name(blob_url2)
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    blob_client1 = blob_service.get_blob_client(container=container, blob=blob_name1)
    blob_client2 = blob_service.get_blob_client(container=container, blob=blob_name2)
    with open(local1, "wb") as f:
        f.write(blob_client1.download_blob().readall())
    with open(local2, "wb") as f:
        f.write(blob_client2.download_blob().readall())
    wav1 = convert_to_wav(local1)
    wav2 = convert_to_wav(local2)

    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    score, prediction = verification.verify_files(wav1, wav2)

    # Limpieza
    for f in [local1, local2]:
        if os.path.exists(f):
            os.remove(f)
    for f in [wav1, wav2]:
        if f not in [local1, local2] and os.path.exists(f):
            os.remove(f)
    return float(score), bool(prediction)

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("audio_front.html", media_type="text/html")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload_audio")
async def upload_audio(audio: UploadFile = File(...)):
    blob_name = f"{uuid.uuid4()}.wav"
    blob_client = container_client.get_blob_client(blob_name)
    # Leer el archivo en memoria y subirlo
    content = await audio.read()
    blob_client.upload_blob(content, overwrite=True)
    blob_url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{AZURE_STORAGE_CONTAINER}/{blob_name}"
    return {"blob_url": blob_url}

def borrar_blobs_despues_de_delay(delay=10):
    time.sleep(delay)
    for blob in container_client.list_blobs():
        try:
            container_client.delete_blob(blob.name)
        except Exception as e:
            print(f"Error borrando blob {blob.name}: {e}")

@app.post("/compare")
async def compare(
    request: Request,
    background_tasks: BackgroundTasks
):
    try:
        data = await request.json()
        # Verifica que los parámetros existen
        if not data or 'blob_url1' not in data or 'blob_url2' not in data:
            raise HTTPException(status_code=400, detail="Faltan parámetros blob_url1 o blob_url2")
        # Llama a compare_blobs con los 5 parámetros
        score, same = compare_blobs(
            data['blob_url1'],
            data['blob_url2'],
            AZURE_STORAGE_ACCOUNT,
            AZURE_STORAGE_KEY,
            AZURE_STORAGE_CONTAINER
        )
        # Convierte tensores a valores nativos si es necesario
        if hasattr(score, 'item'):
            score = float(score.item())
        if hasattr(same, 'item'):
            same = bool(same.item())
        # Lanzar el borrado en segundo plano
        background_tasks.add_task(borrar_blobs_despues_de_delay, 10)
        return {"score": score, "same_speaker": same}
    except Exception as e:
        print("Error en /compare:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/Picture1.png")
