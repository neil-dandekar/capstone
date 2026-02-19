from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# allow your GitHub Pages origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chguerra15.github.io",          # your GH pages domain
        "http://localhost:8000",                # optional
        "http://127.0.0.1:5500",                # optional (live server)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    text: str
    mode: str  # "classification" or "generation"
    interventions: dict = {}  # your payload (concept ids, deltas, etc.)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/baseline")
def baseline(req: RunRequest):
    # TODO: run model without intervention
    return {"output": "...", "top_concepts": []}

@app.post("/intervene")
def intervene(req: RunRequest):
    # TODO: run model with interventions applied
    return {"output": "...", "top_concepts": [], "delta": {}}

@app.get("/concepts")
def concepts():
    # TODO: return list of concepts + ids + stats
    return {"concepts": []}
