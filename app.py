import streamlit as st
import pandas as pd
import numpy as np
import requests
from openai import OpenAI # Import the OpenAI client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import time
import os # Import os to access environment variables
# from google.colab import userdata # Remove import as userdata is not available in Streamlit process

# Get the API key from Colab Secrets # Remove attempt to get from userdata at the top
# OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Instrumentation Fault Detection Agent", layout="wide")

# -------------------------------
# Sidebar Configuration
# -------------------------------
st.sidebar.title("IFDA Configuration")
# Get OpenAI API Key from sidebar input
openai_api_key_sidebar = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Prioritize sidebar input, fallback to environment variable
OPENAI_API_KEY = None # Initialize OPENAI_API_KEY
if openai_api_key_sidebar:
    OPENAI_API_KEY = openai_api_key_sidebar
else:
    # Attempt to get from environment variable if not provided in sidebar
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
         st.sidebar.warning("OpenAI API Key not found in sidebar or environment variable. Please provide it.")

cmms_endpoint = st.sidebar.text_input("CMMS API Endpoint")
cmms_auth_token = st.sidebar.text_input("CMMS Auth Token", type="password")
data_source = st.sidebar.radio("Select Data Source", ["Upload CSV", "Simulated Live Stream"])

# Initialize OpenAI client if API key is available
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.sidebar.error(f"Failed to initialize OpenAI client: {e}")

# -------------------------------
# Signal Integrity Checker Agent
# -------------------------------
def signal_integrity_checker(df):
    df['fault_flag'] = 'Normal'
    # Ensure 'value' column exists before accessing
    if 'value' in df.columns:
        df.loc[df['value'].isnull(), 'fault_flag'] = 'Dropout'
        df.loc[df['value'].diff().abs() > 50, 'fault_flag'] = 'Spike'
        # Use .loc to avoid SettingWithCopyWarning
        df.loc[df['value'].rolling(window=5).std() < 0.01, 'fault_flag'] = 'Frozen'
    else:
        st.warning("Input DataFrame does not contain a 'value' column.")
    return df

# -------------------------------
# Fault Classifier Agent
# -------------------------------
def fault_classifier(df):
    fault_mapping = {'Normal': 0, 'Dropout': 1, 'Spike': 2, 'Frozen': 3}
    df['fault_code'] = df['fault_flag'].map(fault_mapping)
    # Ensure 'value' column exists before using
    if 'value' in df.columns:
        features = df[['value']].fillna(0)
        labels = df['fault_code']
        # Check if there are enough samples for splitting and training
        if len(features) > 1 and len(np.unique(labels)) > 1:
            # Check if stratification is possible
            class_counts = labels.value_counts()
            stratify_possible = all(count >= 2 for count in class_counts)

            if stratify_possible:
                 X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels) # Use stratify for balanced split
            else:
                 st.warning("Insufficient samples in one or more classes for stratification. Proceeding without stratification.")
                 X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            df['predicted_fault'] = clf.predict(features)
            reverse_map = {v: k for k, v in fault_mapping.items()}
            df['predicted_fault'] = df['predicted_fault'].map(reverse_map)
        else:
             st.warning("Not enough data or classes to train the fault classifier.")
             df['predicted_fault'] = 'Cannot Classify' # Assign a default if cannot train
    else:
        st.warning("Input DataFrame does not contain a 'value' column for classification.")
        df['predicted_fault'] = 'Cannot Classify' # Assign a default if column is missing
    return df

# -------------------------------
# OpenAI Integration
# -------------------------------
def get_recommendation(tag, fault_type):
    # Use the client initialized in the main app
    global client # Access the global client object
    if not client:
        return "OpenAI client not initialized. Please provide API Key."

    prompt = f"What is the recommended action for a sensor with tag {tag} showing a {fault_type} fault?"
    try:
        # Use the new API syntax
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip() # Access content differently
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

# -------------------------------
# CMMS Integration
# -------------------------------
def send_to_cmms(tag, fault, action):
    if not cmms_endpoint or not cmms_auth_token:
        return "CMMS credentials not provided."
    payload = {
        "equipment_id": tag,
        "fault": fault,
        "action": action,
        "priority": "High",
        "requested_by": "IFDA",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    headers = {"Authorization": f"Bearer {cmms_auth_token}", "Content-Type": "application/json"}
    try:
        response = requests.post(cmms_endpoint, headers=headers, data=json.dumps(payload))
        return f"CMMS Response: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error sending to CMMS: {e}"

# -------------------------------
# Main UI
# -------------------------------
st.title("🛠️ Instrumentation Fault Detection Agent (IFDA)")

df = None # Initialize df to None

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Sensor Data CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Raw Data")
            st.dataframe(df)

            # Add basic checks for required columns before processing
            if 'tag' not in df.columns or 'value' not in df.columns:
                 st.error("Uploaded CSV must contain 'tag' and 'value' columns.")
                 df = None # Clear df if required columns are missing
            else:
                df = signal_integrity_checker(df.copy()) # Use .copy() to avoid SettingWithCopyWarning
                df = fault_classifier(df.copy()) # Use .copy()

                st.subheader("Processed Data with Faults")
                # Ensure columns exist before displaying
                display_cols = [col for col in ['tag', 'value', 'fault_flag', 'predicted_fault'] if col in df.columns]
                st.dataframe(df[display_cols])

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            df = None # Clear df on error

elif data_source == "Simulated Live Stream":
    st.info("Simulating live sensor data stream...")
    simulated_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-01-01', periods=100, freq='T'),
        'tag': ['PT-101'] * 100,
        'value': np.random.normal(loc=50, scale=5, size=100)
    })
    simulated_data.loc[20, 'value'] = np.nan  # dropout
    simulated_data.loc[40, 'value'] = 200     # spike
    simulated_data.loc[60:65, 'value'] = 48.0 # frozen

    # Ensure columns exist before processing (though they are created here)
    if 'tag' in simulated_data.columns and 'value' in simulated_data.columns:
        df = signal_integrity_checker(simulated_data.copy()) # Use .copy()
        df = fault_classifier(df.copy()) # Use .copy()

        st.line_chart(df[['value']])
        # Ensure columns exist before displaying
        display_cols = [col for col in ['timestamp', 'tag', 'value', 'fault_flag', 'predicted_fault'] if col in df.columns]
        st.dataframe(df[display_cols])
    else:
        st.error("Simulated data generation failed to create required columns.")
        df = None # Clear df on error

# Only show recommendation and CMMS section if df is not None and contains necessary columns
if df is not None and 'tag' in df.columns and 'predicted_fault' in df.columns:
    # Add check if df is not empty
    if not df.empty:
        selected_row_index = st.selectbox("Select a row to get recommendation", df.index)
        # Ensure selected_row_index is valid for the current df
        if selected_row_index in df.index:
            selected_tag = df.loc[selected_row_index, 'tag']
            selected_fault = df.loc[selected_row_index, 'predicted_fault']
            recommendation = get_recommendation(selected_tag, selected_fault)
            st.markdown(f"**Recommendation:** {recommendation}")

            if st.button("Send to CMMS"):
                cmms_response = send_to_cmms(selected_tag, selected_fault, recommendation)
                st.success(cmms_response)
        else:
             st.warning("Selected row index is invalid for the current data.")
    else:
        st.info("No data available to select a row for recommendation.")
else:
    if data_source != "Upload CSV" or uploaded_file: # Only show warning if file was uploaded or using simulated data
        st.info("Data processing incomplete due to missing columns or errors. Recommendation and CMMS options are not available.")
