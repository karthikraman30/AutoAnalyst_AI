import requests
import base64
import os

BASE_URL = "http://127.0.0.1:8000"

def test_chat():
    # 1. Upload File
    filename = "Mutual_Fund_Data.csv" # Or whatever file you have
    if not os.path.exists(filename):
        filename = input("Enter CSV filename: ")
        
    print(f"--- Uploading {filename} ---")
    with open(filename, "rb") as f:
        resp = requests.post(f"{BASE_URL}/upload", files={"file": f})
        file_id = resp.json()['file_id']
        print("✅ Uploaded. File ID:", file_id)

    # 2. Ask a Question (The Magic Moment)
    while True:
        user_query = input("\nAsk your data a question (or 'exit'): ")
        if user_query.lower() == 'exit':
            break
            
        print("🤖 Thinking...")
        payload = {
            "message": user_query,
            "file_id": file_id
        }
        
        try:
            resp = requests.post(f"{BASE_URL}/chat", json=payload)
            resp.raise_for_status()  # This will raise an HTTPError for bad responses
            data = resp.json()
            
            print("\n--- AI Response ---")
            if 'error' in data:
                print(f"❌ Error: {data.get('detail', 'Unknown error occurred')}")
            else:
                print(f"🐍 Generated Code:\n{data.get('generated_code', 'No code generated')}")
                print(f"\n📝 Output:\n{data.get('response_text', 'No response text')}")
                
                if data.get('image_output'):
                    print("📊 Plot generated! Saving to 'ai_plot.png'...")
                    with open("ai_plot.png", "wb") as f:
                        f.write(base64.b64decode(data['image_output']))
        except requests.exceptions.HTTPError as http_err:
            print(f"❌ HTTP Error: {http_err}")
            print(f"Response content: {resp.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"❌ Request failed: {req_err}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            if 'resp' in locals():
                print(f"Response status: {resp.status_code}")
                print(f"Response content: {resp.text}")

if __name__ == "__main__":
    test_chat()