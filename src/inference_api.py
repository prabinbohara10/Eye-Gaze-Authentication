import os
import pickle
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from torch.nn.functional import cosine_similarity

from src.models import GazeEmbeddingModel


# Path for embeddings and trained model's weights
embedding_file_path = "models/user_embeddings.pkl"
gaze_embedding_model_weights_path = "models/gaze_embedding_model.pth"

# FastAPI app
app = FastAPI()

# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GazeEmbeddingModel().to(device)
model.load_state_dict(torch.load(gaze_embedding_model_weights_path,  weights_only=True))  # Loading trained models' weights
model.eval()  # Set model to evaluation mode

# Pydantic models
class GazeDataPoint(BaseModel):
    X: float
    Y: float

class GazeRequest(BaseModel):
    data_points: List[GazeDataPoint]

class EnrollRequest(BaseModel):
    user_id: str
    data_points: List[GazeDataPoint]

# Function to load embeddings from the file
def load_embeddings():
    if os.path.exists(embedding_file_path):
        with open(embedding_file_path, 'rb') as f:
            return pickle.load(f)
    return {}

# Function to save embeddings to the file
def save_embeddings(embeddings):
    with open(embedding_file_path, 'wb') as f:
        pickle.dump(embeddings, f)


# Inference function to get embeddings from gaze data points
def get_embeddings(data_points: List[GazeDataPoint]):
    points = np.array([[dp.X, dp.Y] for dp in data_points], dtype=np.float32)
    points_tensor = torch.tensor(points).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        embeddings = model(points_tensor)
    
    # Convert embeddings to a list of floats for easy JSON response
    embeddings_list = embeddings.squeeze(0).cpu().numpy().tolist()
    return embeddings_list

# Enroll a new user by storing their gaze embedding
def enroll_new_user(user_id: str, data_points: List[GazeDataPoint]):
    embeddings = get_embeddings(data_points)
    # Load existing embeddings
    all_embeddings = load_embeddings()
    
    if user_id in all_embeddings:
        raise HTTPException(status_code=400, detail=f"User {user_id} is already enrolled.")
    
    # Store the new user's embedding
    all_embeddings[user_id] = embeddings
    save_embeddings(all_embeddings)
    print(f"User {user_id} enrolled successfully!")

# Authenticate a user by comparing their gaze data with stored embeddings
def authenticate_user(data_points: List[GazeDataPoint], threshold=0.8):
    embeddings = get_embeddings(data_points)
    all_embeddings = load_embeddings()
    
    for user_id, stored_embedding in all_embeddings.items():
        similarity = cosine_similarity(torch.tensor(embeddings).unsqueeze(0).to(device), torch.tensor(stored_embedding).unsqueeze(0).to(device)).item()
        if similarity >= threshold:
            return True, user_id
    return False, None



# API Endpoints
@app.get("/all_users/")
async def get_all_users():
    """Retrieve all enrolled users."""
    try:
        # Load all stored embeddings and get user IDs
        all_embeddings = load_embeddings()
        users = list(all_embeddings.keys())
        return {"users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/enroll_user/")
async def enroll_user(request: EnrollRequest):
    """Enroll a new user by storing their gaze embedding."""
    try:
        enroll_new_user(request.user_id, request.data_points)
        return {"message": f"User {request.user_id} enrolled successfully."}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/authenticate/")
async def authenticate(request: GazeRequest):
    """Authenticate a user based on gaze data."""
    try:
        is_authenticated, user_id = authenticate_user(request.data_points)
        if is_authenticated:
            return {"message": f"User authenticated successfully as {user_id}."}
        else:
            return {"message": "Authentication failed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference/")
async def inference(request: GazeRequest):
    """Generate gaze embeddings from the provided data points."""
    try:
        embeddings = get_embeddings(request.data_points)
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))