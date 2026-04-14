"""
FastAPI Application Entry Point.
Serves REST API, manages MCP tool calls, handles WebSocket streaming.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import settings
from app.api.v1.routes.auth import router as auth_router
from app.api.v1.routes.email import router as email_router
from app.api.v1.routes.agent import router as agent_router
from app.middleware.rate_limit import RateLimitMiddleware

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("gmail-mcp-assistant")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} [{settings.APP_ENV}]")

    # Init database tables
    try:
        from app.db.session import engine, Base
        from app.models import user  # noqa: trigger model registration
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized")
    except Exception as e:
        logger.warning(f"DB init warning: {e}")

    # Init ChromaDB
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        client.get_or_create_collection(settings.CHROMA_COLLECTION_NAME)
        logger.info("ChromaDB vector store initialized")
    except Exception as e:
        logger.warning(f"ChromaDB init warning: {e}")

    logger.info(f"API: http://{settings.APP_HOST}:{settings.APP_PORT}")
    logger.info(f"Docs: http://{settings.APP_HOST}:{settings.APP_PORT}/docs")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered Gmail intelligence platform on MCP",
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,
    lifespan=lifespan,
)

# Middleware stack
app.add_middleware(CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RateLimitMiddleware,
    calls_per_minute=settings.RATE_LIMIT_PER_MINUTE)

# Versioned routes
app.include_router(auth_router, prefix="/api/v1")
app.include_router(email_router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": "1.0.0",
        "environment": settings.APP_ENV,
    })


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    checks = {"api": True, "database": False, "vector_db": False}
    try:
        from app.db.session import AsyncSessionLocal
        from sqlalchemy import text
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        pass
    try:
        import chromadb
        chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        checks["vector_db"] = True
    except Exception:
        pass
    all_ok = all(checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "not_ready", "checks": checks},
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.is_development)