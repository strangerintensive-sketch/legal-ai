import logging
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from dotenv import load_dotenv

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress ChromaDB telemetry capture warnings  
logging.getLogger("chromadb").setLevel(logging.WARNING)

# Load environment variables before importing routes
load_dotenv()
logger.info("✅ Environment variables loaded")

from app.api.routes import router
from app.models.database import init_db

app = FastAPI(
    title="Legal AI — Pearson Specter Litt",
    description="Document ingestion, grounded drafting, and improvement from operator edits.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
def on_startup():
    logger.info("Starting application...")
    init_db()
    logger.info("✅ Database initialized and ready")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")
