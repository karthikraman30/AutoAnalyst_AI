import streamlit as st
import requests
import pandas as pd
import json
import base64

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000"
st.set_page_config(
    page_title="AutoAnalyst AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Pro" Look ---
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "columns" not in st.session_state:
    st.session_state.columns = []

# --- Helper: Send Message to Backend ---
def send_message(prompt):
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Call Backend
    payload = {"message": prompt, "file_id": st.session_state.file_id}
    
    try:
        response = requests.post(f"{BACKEND_URL}/chat", json=payload)
        if response.status_code == 200:
            data = response.json()
            st.session_state.messages.append({
                "role": "assistant",
                "content": data['response_text'],
                "image": data['image_output'],
                "code": data['generated_code']
            })
        else:
            st.error(f"Server Error: {response.text}")
    except Exception as e:
        st.error(f"Connection Failed: {e}")

# ==========================================
#              SIDEBAR DASHBOARD
# ==========================================
with st.sidebar:
    st.title("🤖 AutoAnalyst AI")
    st.markdown("---")
    
    # 1. File Upload Section
    uploaded_file = st.file_uploader("📂 Upload Dataset", type=["csv", "xlsx"])

    if uploaded_file and not st.session_state.file_id:
        with st.spinner("🚀 Ingesting Data..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            try:
                response = requests.post(f"{BACKEND_URL}/upload", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.file_id = data['file_id']
                    st.session_state.columns = data['preview']['columns']
                    
                    # Store preview for dashboard
                    st.session_state.preview = data['preview']
                    st.toast("File Uploaded Successfully!", icon="✅")
                    st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")

    # 2. Data Health Dashboard (Only shows if file is active)
    if st.session_state.file_id:
        st.subheader("📊 Data Snapshot")
        col1, col2 = st.columns(2)
        rows = st.session_state.preview['shape'][0]
        cols = st.session_state.preview['shape'][1]
        col1.metric("Rows", rows)
        col2.metric("Columns", cols)
        
        with st.expander("🔍 View Raw Data"):
            st.dataframe(pd.DataFrame(st.session_state.preview['first_rows']))
            st.write("Column Types:", st.session_state.preview['dtypes'])

        st.markdown("---")

        # 3. AutoML Toolkit (Quick Actions)
        st.subheader("🛠️ AutoML Toolkit")
        
        # Action: Auto Clean
        if st.button("🧹 Auto-Clean Data"):
            send_message("Check the dataset for missing values or duplicates and clean it automatically.")
            st.rerun()
            
        st.markdown("#### 🎯 Model Training")
        # Action: Target Selection
        target_col = st.selectbox("Select Target Column", st.session_state.columns)
        
        # Action: Train Model
        if st.button("🤖 Train Best Model"):
            if target_col:
                send_message(f"Predict the column '{target_col}'. Auto-encode categorical variables first, then find the best model and show feature importance.")
                st.rerun()
        
        st.markdown("---")
        if st.button("🔄 Reset / New File"):
            st.session_state.clear()
            st.rerun()

# ==========================================
#              MAIN CHAT INTERFACE
# ==========================================

st.subheader("💬 Data Scientist Assistant")

if not st.session_state.file_id:
    st.info("👈 Upload a dataset in the sidebar to activate the AI Agent.")
else:
    # Display Chat History
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            
            # Show Image if available
            if "image" in msg and msg["image"]:
                image_data = base64.b64decode(msg["image"])
                st.image(image_data, caption="Generated Insight")
            
            # Show Code inside an expander (Keep UI clean)
            if "code" in msg and msg["code"]:
                with st.expander("See Python Code"):
                    st.code(msg["code"], language="python")

    # Chat Input Area
    if prompt := st.chat_input("Ask a question... (e.g. 'Plot the distribution of Age')"):
        send_message(prompt)
        st.rerun()