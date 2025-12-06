import streamlit as st
import requests
import pandas as pd
import json
from io import BytesIO
import base64

# Configuration
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Data Scientist", layout="wide")

st.title("🤖 AI Data Scientist Assistant")
st.markdown("Upload a dataset and ask questions in plain English!")

# Initialize Session State (to remember file_id and chat history)
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded_file and not st.session_state.file_id:
        with st.spinner("Uploading & Analyzing..."):
            # Prepare file for API
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            try:
                response = requests.post(f"{BACKEND_URL}/upload", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.file_id = data['file_id']
                    st.success("File Processed Successfully!")
                    
                    # Show Preview
                    st.subheader("📊 Dataset Preview")
                    preview = data['preview']
                    st.write(f"**Rows:** {preview['shape'][0]} | **Columns:** {preview['shape'][1]}")
                    
                    # Convert first_rows list to DataFrame for nice display
                    st.dataframe(pd.DataFrame(preview['first_rows']))
                    
                    st.write("**Columns:**")
                    st.json(preview['dtypes'])
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

    # Reset Button
    if st.session_state.file_id:
        if st.button("Upload New File"):
            st.session_state.file_id = None
            st.session_state.messages = []
            st.rerun()

# --- Main Chat Area ---
if st.session_state.file_id:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "image" in message and message["image"]:
                # Decode and display image
                image_data = base64.b64decode(message["image"])
                st.image(image_data)

    # Chat Input
    if prompt := st.chat_input("Ask about your data (e.g., 'Plot histogram of Salary')"):
        # 1. Add User Message to History
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call Backend API
        with st.chat_message("assistant"):
            with st.spinner("Thinking & Coding..."):
                payload = {
                    "message": prompt,
                    "file_id": st.session_state.file_id
                }
                try:
                    response = requests.post(f"{BACKEND_URL}/chat", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        response_text = data['response_text']
                        generated_code = data['generated_code']
                        image_base64 = data['image_output']

                        # Display Response
                        st.markdown(response_text)
                        
                        # Display Image if available
                        if image_base64:
                            image_data = base64.b64decode(image_base64)
                            st.image(image_data)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response_text,
                                "image": image_base64
                            })
                        else:
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response_text,
                                "image": None
                            })

                        # Optional: Show the code it wrote (Expander)
                        with st.expander("See Python Code"):
                            st.code(generated_code, language='python')

                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

else:
    st.info("👈 Please upload a dataset in the sidebar to start chatting!")