import streamlit as st
import pandas as pd
import requests
import random
import os
import string
from dotenv import load_dotenv

load_dotenv()

# Load FastAPI URLs from environment variables
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
FASTAPI_PREDICT_URL = os.getenv("FASTAPI_PREDICT_URL", f"{FASTAPI_BASE_URL}/predict")
FASTAPI_PAST_PREDICTIONS_URL = os.getenv("FASTAPI_PAST_PREDICTIONS_URL", f"{FASTAPI_BASE_URL}/past_predictions/")
FASTAPI_LOGIN_URL = f"{FASTAPI_BASE_URL}/login"
FASTAPI_REGISTER_URL = f"{FASTAPI_BASE_URL}/register"
FASTAPI_ME_URL = f"{FASTAPI_BASE_URL}/me"

# Page configuration
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if "token" not in st.session_state:
    st.session_state.token = None
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"


def login(username: str, password: str) -> bool:
    """Authenticate user and get token."""
    try:
        response = requests.post(
            FASTAPI_LOGIN_URL,
            json={"username": username, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.token = data["access_token"]
            st.session_state.user = data["user"]
            st.session_state.authenticated = True
            return True
        else:
            try:
                error_detail = response.json().get("detail", "Login failed")
            except:
                error_detail = f"Login failed (Status: {response.status_code})"
            st.error(error_detail)
            return False
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the server. Make sure the API is running.")
        return False
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False


def register(username: str, email: str, password: str, full_name: str) -> bool:
    """Register a new user."""
    try:
        response = requests.post(
            FASTAPI_REGISTER_URL,
            json={
                "username": username,
                "email": email,
                "password": password,
                "full_name": full_name if full_name else None
            }
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.token = data["access_token"]
            st.session_state.user = data["user"]
            st.session_state.authenticated = True
            return True
        else:
            try:
                error_detail = response.json().get("detail", "Registration failed")
            except:
                error_detail = f"Registration failed (Status: {response.status_code})"
            st.error(error_detail)
            return False
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the server. Make sure the API is running.")
        return False
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False


def logout():
    """Logout user."""
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.token = None
    st.rerun()


def show_login_page():
    """Display login form."""
    st.markdown("""
    <style>
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("## Login")
        st.markdown("---")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if username and password:
                    if login(username, password):
                        st.success("Login successful!")
                        st.rerun()
                else:
                    st.warning("Please fill in all fields")
        
        st.markdown("---")
        st.markdown("Don't have an account?")
        if st.button("Create Account", use_container_width=True):
            st.session_state.auth_page = "register"
            st.rerun()


def show_register_page():
    """Display registration form."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("## Create Account")
        st.markdown("---")
        
        with st.form("register_form"):
            username = st.text_input("Username", placeholder="Choose a username")
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Choose a password")
            
            submit = st.form_submit_button("Create Account", use_container_width=True)
            
            if submit:
                if username and email and password:
                    if len(password) < 4:
                        st.error("Password must be at least 4 characters")
                    elif "@" not in email:
                        st.error("Please enter a valid email address")
                    else:
                        if register(username, email, password, None):
                            st.success("Account created successfully!")
                            st.rerun()
                else:
                    st.warning("Please fill in all fields")
        
        st.markdown("---")
        st.markdown("Already have an account?")
        if st.button("Login", use_container_width=True):
            st.session_state.auth_page = "login"
            st.rerun()


def show_main_app():
    """Display main application after authentication."""
    # Sidebar with user info
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.user['username']}!")
        st.markdown(f"{st.session_state.user['email']}")
        if st.session_state.user.get('full_name'):
            st.markdown(f"{st.session_state.user['full_name']}")
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            logout()
    
    st.title("Churn Prediction Web App")
    
    # Options for the user interface
    option = st.selectbox(
        "Select option:", ("Manual Input", "CSV File Upload", "View Past Predictions")
    )
    
    if option == "Manual Input":
        show_manual_input()
    elif option == "CSV File Upload":
        show_csv_upload()
    elif option == "View Past Predictions":
        show_past_predictions()


def generate_unique_id():
    """Generate a unique ID for customerID."""
    letters = "".join(random.choice(string.ascii_uppercase) for _ in range(4))
    numbers = str(random.randint(1000, 9999))
    return f"{numbers}-{letters}"


def generate_input_fields(df):
    """Generate input fields based on DataFrame columns."""
    input_data = {}
    for col in df.columns:
        col_type = df[col].dtype

        if col_type in ["float64", "int64"]:
            input_data[col] = st.selectbox(
                f"Select value for {col}", [0, 1] if col == "SeniorCitizen" else [0.0]
            )
        elif col_type == "object":
            input_data[col] = (
                generate_unique_id()
                if col == "customerID"
                else st.selectbox(f"Select value for {col}", df[col].unique().tolist())
            )
    return input_data


def show_manual_input():
    """Show manual input form."""
    st.header("Input Features Manually")
    
    sample_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../../data/churn.csv"))
    sample_df["TotalCharges"] = pd.to_numeric(
        sample_df["TotalCharges"], errors="coerce"
    )
    sample_df = sample_df.dropna()

    user_input = generate_input_fields(sample_df)

    if st.button("Predict", use_container_width=True):
        with st.spinner("Making prediction..."):
            response = requests.post(
                FASTAPI_PREDICT_URL, 
                json=user_input, 
                headers={
                    "X-Source": "streamlit",
                    "Authorization": f"Bearer {st.session_state.token}"
                }
            )
            if response.status_code == 200:
                prediction = response.json()["predictions"][0]
                user_input["prediction"] = prediction
                
                if prediction == 1:
                    st.error(f"Prediction: Customer will likely CHURN")
                else:
                    st.success(f"Prediction: Customer will likely STAY")
                    
                st.write(pd.DataFrame([user_input]).drop(columns=["Churn"]))
            else:
                st.error("Error: Unable to get prediction")


# Required columns for churn prediction
REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]


def validate_csv_columns(df):
    """Validate that the CSV has the required columns for churn prediction."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in REQUIRED_COLUMNS and col != "Churn"]
    return missing_cols, extra_cols


def show_csv_upload():
    """Show CSV upload form."""
    st.header("Upload CSV File")
    
    # Show expected format
    with st.expander("📋 Expected CSV Format (Click to expand)"):
        st.markdown("""
        **Your CSV must contain these columns for Churn Prediction:**
        
        | Column | Type | Example |
        |--------|------|---------|
        | customerID | Text | 7590-VHVEG |
        | gender | Text | Male/Female |
        | SeniorCitizen | Number | 0 or 1 |
        | Partner | Text | Yes/No |
        | Dependents | Text | Yes/No |
        | tenure | Number | 1-72 |
        | PhoneService | Text | Yes/No |
        | MultipleLines | Text | Yes/No/No phone service |
        | InternetService | Text | DSL/Fiber optic/No |
        | OnlineSecurity | Text | Yes/No/No internet service |
        | OnlineBackup | Text | Yes/No/No internet service |
        | DeviceProtection | Text | Yes/No/No internet service |
        | TechSupport | Text | Yes/No/No internet service |
        | StreamingTV | Text | Yes/No/No internet service |
        | StreamingMovies | Text | Yes/No/No internet service |
        | Contract | Text | Month-to-month/One year/Two year |
        | PaperlessBilling | Text | Yes/No |
        | PaymentMethod | Text | Electronic check/Mailed check/... |
        | MonthlyCharges | Number | 29.85 |
        | TotalCharges | Number | 29.85 |
        
        ⚠️ **Files with different columns (like poverty data, sales data, etc.) will not work!**
        """)

    uploaded_files = st.file_uploader(
        "Choose CSV file(s)", 
        type=["csv"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        all_predictions = []
        
        for uploaded_file in uploaded_files:
            st.subheader(f"📄 File: {uploaded_file.name}")
            
            # Try different encodings
            try:
                csv_data = pd.read_csv(uploaded_file, encoding='utf-8')
            except:
                uploaded_file.seek(0)
                try:
                    csv_data = pd.read_csv(uploaded_file, encoding='latin-1')
                except:
                    uploaded_file.seek(0)
                    try:
                        csv_data = pd.read_csv(uploaded_file, encoding='cp1252')
                    except Exception as e:
                        st.error(f"❌ Cannot read file {uploaded_file.name}: {str(e)}")
                        continue
            
            # Validate columns
            missing_cols, extra_cols = validate_csv_columns(csv_data)
            
            if missing_cols:
                st.error(f"❌ **Invalid file format!** Missing required columns:")
                st.code(", ".join(missing_cols))
                st.warning(f"Your file has these columns: {', '.join(csv_data.columns.tolist())}")
                st.info("💡 This model only works with **Telecom Customer Churn** data. Please upload a valid churn dataset.")
                continue
            
            # Handle TotalCharges if exists
            if "TotalCharges" in csv_data.columns:
                csv_data["TotalCharges"] = pd.to_numeric(
                    csv_data["TotalCharges"], errors="coerce"
                )
                csv_data = csv_data.dropna()

            st.success(f"✅ Valid format - {len(csv_data)} rows")
            st.dataframe(csv_data.head(10))
            all_predictions.append((uploaded_file.name, csv_data))
        
        if all_predictions and st.button("🔮 Predict All Files", use_container_width=True):
            with st.spinner("Processing predictions..."):
                for file_name, csv_data in all_predictions:
                    st.write(f"Processing: {file_name}")
                    
                    try:
                        response = requests.post(
                            FASTAPI_PREDICT_URL,
                            json=csv_data.to_dict(orient="records"),
                            headers={
                                "X-Source": "streamlit",
                                "Authorization": f"Bearer {st.session_state.token}"
                            },
                        )
                        if response.status_code == 200:
                            result = response.json()
                            if "existing_data" in result:
                                st.warning(f"{file_name}: Some CustomerIDs already exist")
                                st.write(pd.DataFrame(result["predictions"]))
                            else:
                                csv_data["Prediction"] = result["predictions"]
                                st.success(f"✅ {file_name}: Predictions completed!")
                                
                                # Show results
                                cols_to_drop = [c for c in ["Churn"] if c in csv_data.columns]
                                display_df = csv_data.drop(columns=cols_to_drop) if cols_to_drop else csv_data
                                st.dataframe(display_df)
                        else:
                            try:
                                error_msg = response.json().get("detail", "Unknown error")
                            except:
                                error_msg = f"Status code: {response.status_code}"
                            st.error(f"❌ {file_name}: Error - {error_msg}")
                    except Exception as e:
                        st.error(f"❌ {file_name}: Error - {str(e)}")


def show_past_predictions():
    """Show past predictions."""
    st.header("Past Predictions")

    # Date filter inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")
    with col3:
        source = st.selectbox(
            "Select prediction source:",
            ("Web Application", "Scheduled Predictions", "All"),
        )

    if st.button("Search", use_container_width=True):
        if start_date and end_date:
            params = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "source": source,
            }
            response = requests.get(
                FASTAPI_PAST_PREDICTIONS_URL, 
                params=params,
                headers={"Authorization": f"Bearer {st.session_state.token}"}
            )

            if response.status_code == 200:
                past_predictions = response.json()
                if past_predictions:
                    past_predictions_df = pd.DataFrame(past_predictions)
                    st.success(f"Found {len(past_predictions)} predictions")
                    st.write(past_predictions_df.drop(columns=["date", "SourcePrediction"], errors='ignore'))
                else:
                    st.info("No past predictions found for the selected date range.")
            else:
                st.error("Failed to fetch past predictions.")


# ==================== MAIN APP LOGIC ====================

st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Show appropriate page based on authentication state
if not st.session_state.authenticated:
    st.title("Churn Prediction App")
    st.markdown("---")
    
    if st.session_state.auth_page == "login":
        show_login_page()
    else:
        show_register_page()
else:
    show_main_app()
