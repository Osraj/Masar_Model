from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import ngrok
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

groq_api = os.getenv("GROQ_API_KEY")

# Import functions from the uploaded script
from testing_rag_v7 import generate_response_v2, db_chroma

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for input validation
class QueryRequest(BaseModel):
    question: str
    chat_history: list = []
    return_chat_name: bool = False

# API route to handle queries
@app.post("/query")
def handle_query(request: QueryRequest):
    query = request.question
    chat_history = request.chat_history
    return_chat_name = request.return_chat_name
    _, answer, name = generate_response_v2(query, chat_history, return_chat_name)
    return {"answer": answer, "chat_name": name if return_chat_name else ""}

# Start ngrok tunnel
ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))  # Ensure this token is in your .env file
tunnel = ngrok.connect(8000)
print(f"Public URL: {tunnel.public_url}")

# Run FastAPI with Uvicorn (manually start this in terminal if needed)
# uvicorn.run(app, host="0.0.0.0", port=8000)