from sqlalchemy import create_engine, MetaData
from databases import Database
import os
from dotenv import load_dotenv
import sys

# Add the environment module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../environment'))
from environment_config import env  # This will set the DATABASE_URL

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Database configuration
database = Database(DATABASE_URL)
metadata = MetaData()

# Sync version for creating tables - handle both MySQL and SQLite
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
