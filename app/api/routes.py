from pathlib import Path
import hashlib
import hmac
import logging
import os
import traceback
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import has_ingested_hash, write_audit, write_ingest_audit

router = APIRouter(prefix="/api")
settings = get_settings()
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
        write_audit(payload.question, result["source_used"], result.get("trace", []))

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),

        }

    except Exception as exc:
        error_id = uuid.uuid4().hex[:12]
        frames = traceback.extract_tb(exc.__traceback__)
        chain = []
        current = exc
        while current is not None and len(chain) < 5:
            chain.append(type(current).__name__)
            current = current.__cause__ or current.__context__
        locations = ";".join(
            f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
            for frame in frames[-6:]
        )
        logger.error(
            "Chat request failed error_id=%s exception_type=%s http_status=%s "
            "error_code=%s traceback_locations=%s",
            error_id,
            type(exc).__name__,
            getattr(exc, "status_code", None),
            getattr(exc, "code", None),
            locations,
        )
        logger.error(
            "Chat request diagnostics error_id=%s exception_chain=%s "
            "http_proxy_present=%s https_proxy_present=%s openai_base_url_present=%s",
            error_id,
            ">".join(chain),
            bool(os.getenv("HTTP_PROXY") or os.getenv("http_proxy")),
            bool(os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")),
            bool(os.getenv("OPENAI_BASE_URL") or os.getenv("openai_base_url")),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Chat request failed; reference={error_id}",
        ) from exc



@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if not hmac.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")
    original_name = Path(file.filename or "").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the upload size limit")
    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    content_sha256 = hashlib.sha256(content).hexdigest()
    if has_ingested_hash(content_sha256):
        raise HTTPException(status_code=409, detail="This document has already been indexed")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    try:
        dest.write_bytes(content)
        docs = load_file(dest)
        if not docs or not any(doc.page_content.strip() for doc in docs):
            raise HTTPException(status_code=400, detail="Uploaded file contains no readable text")
        chunks = chunk_documents(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail="Uploaded file produced no indexable content")
        ids = add_documents(chunks)
        write_ingest_audit(content_sha256, original_name or dest.name, len(chunks), len(ids))
        return {"message": "Document indexed", "file": original_name or dest.name, "chunks": len(chunks), "ids_created": len(ids)}
    except HTTPException:
        raise
    except Exception as exc:
        error_id = uuid.uuid4().hex[:12]
        frames = traceback.extract_tb(exc.__traceback__)
        locations = ";".join(
            f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
            for frame in frames[-6:]
        )
        logger.error(
            "Ingest request failed error_id=%s exception_type=%s traceback_locations=%s",
            error_id,
            type(exc).__name__,
            locations,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed; reference={error_id}",
        ) from exc
    finally:
        dest.unlink(missing_ok=True)
