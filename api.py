from fastapi import FastAPI
from pydantic import BaseModel
from moderation_engine import ModerationEngine

app = FastAPI()
engine = ModerationEngine("moderation_model.pkl")

class Message(BaseModel):
    text: str

@app.post("/moderate")
def moderate(msg: Message):
    return engine.moderate(msg.text)