from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel, EmailStr
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from code.app.database import database, engine, metadata
from code.app.models import predictions, users
from code.app.auth import get_password_hash, verify_password, create_access_token, verify_token
from joblib import load
from typing import List, Union, Optional
from fastapi import Query
from sqlalchemy.sql import and_
from datetime import date, datetime, timedelta
import sys
import os

# Add the environment module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../environment'))
from environment_config import env  # This will set the DATABASE_URL


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metadata.create_all(engine)


class ModelInput(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# Get the base directory for models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = load(os.path.join(MODELS_DIR, "grid_search.joblib"))


@app.on_event("startup")
async def startup():
    await database.connect()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


@app.post("/predict")
async def predict(
    input_data_list: Union[ModelInput, List[ModelInput]], request: Request
):  # Accept a list of input data
    source_prediction = "Scheduled Predictions"  # Default for Airflow
    if request.headers.get("X-Source") == "streamlit":
        source_prediction = "Web Application"

    # print(input_data_list)
    if isinstance(input_data_list, list):
        input_data_dicts = [item.dict() for item in input_data_list]
        input_df = pd.DataFrame(input_data_dicts)

        customer_ids = input_df["customerID"].tolist()

        # Query to check if these CustomerIDs already exist
        # Create placeholders for the IN clause
        placeholders = ','.join([':param_' + str(i) for i in range(len(customer_ids))])
        query = f"""
            SELECT * FROM past_predictions WHERE customerID IN ({placeholders})
        """
        params = {f'param_{i}': customer_ids[i] for i in range(len(customer_ids))}
        existing_records = await database.fetch_all(query, params)
    else:
        input_data_dicts = [input_data_list.dict()]
        input_df = pd.DataFrame(input_data_dicts)

        customer_ids = input_df["customerID"].tolist()
        # print(customer_ids)

        # Query to check if these CustomerIDs already exist
        query = """
            SELECT * FROM past_predictions WHERE customerID = :customer_id
        """
        existing_records = await database.fetch_all(query, {"customer_id": customer_ids[0]})

    if existing_records:
        # Convert the results to a DataFrame
        existing_df = pd.DataFrame(existing_records)
        # print(existing_df)
        l = list(input_df.columns)  # noqa: E741
        l.append("prediction")
        # print(l)
        existing_df = existing_df[l]

        # Return a message with the existing CustomerIDs and associated data
        return {
            "message": "CustomerIDs already exist",
            "existing_data": True,
            "predictions": existing_df.to_dict(orient="records"),
        }

    # Load preprocessing tools and column configurations
    ordinal = load(os.path.join(MODELS_DIR, "Ordinal_Encoder.joblib"))
    scaler = load(os.path.join(MODELS_DIR, "Standard_Scaler.joblib"))
    categorical_columns = load(os.path.join(MODELS_DIR, "categorical_columns.joblib"))
    columns = load(os.path.join(MODELS_DIR, "columns.joblib"))

    # Ensure the dataframe columns are in the correct order
    input_df = input_df[columns]

    # Apply transformations
    input_df[categorical_columns] = ordinal.transform(input_df[categorical_columns])
    input_df = scaler.transform(input_df)

    # Make predictions for all rows at once
    predictions_values = model.predict(input_df).tolist()
    # print(predictions_values)

    current_time = date.today()

    # Prepare the results for database insertion and return
    for idx, prediction_value in enumerate(predictions_values):
        input_data_dicts[idx]["prediction"] = int(prediction_value)
        input_data_dicts[idx]["date"] = current_time
        input_data_dicts[idx]["SourcePrediction"] = source_prediction

        query = predictions.insert().values(**input_data_dicts[idx])
        await database.execute(query)

    return {"predictions": predictions_values}


@app.get("/past_predictions/")
async def get_predictions(
    start_date: str = Query(None),
    end_date: str = Query(None),
    source: str = Query(None),
):
    # Convert string dates to datetime objects
    if start_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    query = predictions.select()

    if start_date and end_date:
        query = query.where(
            and_(predictions.c.date >= start_date, predictions.c.date <= end_date)
        )
    if source and source != "All":
        query = query.where(predictions.c.SourcePrediction == source)

    results = await database.fetch_all(query)

    parsed_results = [dict(result) for result in results]

    return parsed_results


# ==================== AUTHENTICATION ENDPOINTS ====================

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


@app.post("/register", response_model=Token)
async def register(user_data: UserCreate):
    """Register a new user."""
    # Check if username already exists
    query = users.select().where(users.c.username == user_data.username)
    existing_user = await database.fetch_one(query)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    query = users.select().where(users.c.email == user_data.email)
    existing_email = await database.fetch_one(query)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash the password and create user
    hashed_password = get_password_hash(user_data.password)
    
    query = users.insert().values(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        is_active=True,
        created_at=datetime.utcnow()
    )
    
    last_record_id = await database.execute(query)
    
    # Fetch the created user
    query = users.select().where(users.c.id == last_record_id)
    new_user = await database.fetch_one(query)
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user_data.username, "user_id": last_record_id}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": last_record_id,
            "username": user_data.username,
            "email": user_data.email,
            "full_name": user_data.full_name,
            "is_active": True
        }
    }


@app.post("/login", response_model=Token)
async def login(user_data: UserLogin):
    """Login and get access token."""
    # Find user by username
    query = users.select().where(users.c.username == user_data.username)
    user = await database.fetch_one(query)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Verify password
    if not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Check if user is active
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled"
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user["username"], "user_id": user["id"]}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "full_name": user["full_name"],
            "is_active": user["is_active"]
        }
    }


@app.get("/me")
async def get_current_user(request: Request):
    """Get current user from token."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    token = auth_header.split(" ")[1]
    payload = verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    username = payload.get("sub")
    query = users.select().where(users.c.username == username)
    user = await database.fetch_one(query)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "full_name": user["full_name"],
        "is_active": user["is_active"]
    }
