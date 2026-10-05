"""REST API: uvicorn tibetan_spellcheck.api:app"""
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from .correct import correct
from .model import DEVICE, MODEL_ID, get_model

_infer_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    get_model()
    yield


app = FastAPI(title="Tibetan spell checker", lifespan=lifespan)


class CorrectRequest(BaseModel):
    text: str
    add_shad_space: bool = False


class CorrectResponse(BaseModel):
    corrected: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.post("/correct", response_model=CorrectResponse)
def correct_endpoint(req: CorrectRequest):
    with _infer_lock:
        corrected = correct([req.text], add_shad_space=req.add_shad_space)[0]
    return CorrectResponse(corrected=corrected)
