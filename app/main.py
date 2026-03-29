from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "QoSBuddy-5G API is running!"}