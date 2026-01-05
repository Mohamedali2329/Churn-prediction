import pandas as pd
from sqlalchemy import Table, Column, Integer, String, Float, MetaData
from sqlalchemy.types import Boolean, DateTime
from sqlalchemy import create_engine
from sqlalchemy import DateTime  # noqa: F811
import datetime
import os
from dotenv import load_dotenv
import sys

# Add the environment module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../environment'))
from environment_config import env  # This will set the DATABASE_URL

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# csv_path = '/Users/mohamedaminemrabet/Documents/EPITA/DSP/Final-Project-DSP/data/churn.csv' # 'data/churn.csv'
csv_path = os.path.join(os.path.dirname(__file__), "../../data/churn.csv")
df = pd.read_csv(csv_path)

df = df.apply(pd.to_numeric, errors="ignore")


# Database engine configuration - handle both MySQL and SQLite
if DATABASE_URL and DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=False,
        connect_args={
            "charset": "utf8mb4",
            "autocommit": True
        }
    )
metadata = MetaData()


def map_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return Integer
    elif pd.api.types.is_float_dtype(dtype):
        return Float
    elif pd.api.types.is_bool_dtype(dtype):
        return Boolean
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return DateTime
    else:
        return String(255)  # MySQL requires length for VARCHAR


columns = []
for col_name, col_type in df.dtypes.items():
    sqlalchemy_type = map_dtype(col_type)

    if col_name == "customerID":
        column = Column(col_name, String(50), primary_key=True, nullable=False)
        columns.append(column)

    elif col_name == "TotalCharges":
        column = Column(col_name, Float, nullable=True)
        columns.append(column)

    elif col_name == "Churn":
        # Skip the Churn column as it's not needed in predictions table
        pass

    else:
        column = Column(col_name, sqlalchemy_type)
        columns.append(column)

prediction_column = Column("prediction", Float)
columns.append(prediction_column)

date_column = Column("date", DateTime, default=datetime.datetime.utcnow().date)
columns.append(date_column)

source_prediction = Column("SourcePrediction", String(100))
columns.append(source_prediction)

predictions = Table("past_predictions", metadata, *columns)

# Create the table in the database
print("Predictions table created successfully.")

# Add the new table for statistics
statistics = Table(
    "prediction_statistics",
    metadata,
    Column("run_id", String(100), primary_key=True),
    Column("run_time", DateTime, default=datetime.datetime.utcnow),
    Column("evaluated_expectations", Integer),
    Column("successful_expectations", Integer),
    Column("unsuccessful_expectations", Integer),
    Column("success_percent", Float),
    Column("datasource_name", String(255)),
    Column("checkpoint_name", String(255)),
    Column("expectation_suite_name", String(255)),
    Column("file_name", String(255)),
    Column("nb_rows", Integer),
    Column("nb_valid_rows", Integer),
    Column("nb_invalid_rows", Integer),
    Column("nb_cols", Integer),
    Column("nb_valid_cols", Integer),
    Column("nb_invalid_cols", Integer),
)

# User table for authentication
users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(50), unique=True, nullable=False),
    Column("email", String(100), unique=True, nullable=False),
    Column("hashed_password", String(255), nullable=False),
    Column("full_name", String(100), nullable=True),
    Column("is_active", Boolean, default=True),
    Column("created_at", DateTime, default=datetime.datetime.utcnow),
)

# Create the statistics table
metadata.create_all(engine)

print("Statistics table created successfully.")
print("Users table created successfully.")
