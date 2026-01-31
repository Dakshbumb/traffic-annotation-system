from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


from database import Base, engine
from routers import annotated_frames
from routers import videos, annotations, export, autolabel, frames, analytics, monitoring, lanes, safety
from autolabel_worker import start_worker

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart Traffic Annotation System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API routers FIRST (before static files)
app.include_router(videos.router)
app.include_router(annotations.router)
app.include_router(export.router)
app.include_router(frames.router)
app.include_router(annotated_frames.router)
app.include_router(autolabel.router)
app.include_router(analytics.router)
app.include_router(monitoring.router)
app.include_router(lanes.router)
app.include_router(safety.router)

# Static files mount LAST (catches all remaining routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

@app.on_event("startup")
def startup_event():
    start_worker()
