import requests
import base64
import os

BASE_URL = "http://127.0.0.1:8000"

def test_workflow():
    # --- Step 1: Get User Input ---
    filename = input("Enter the CSV filename (e.g. data.csv): ").strip()
    
    if not os.path.exists(filename):
        print(f"❌ Error: File '{filename}' not found in this folder.")
        return

    # --- Step 2: Upload Data ---
    print(f"\n--- 1. Uploading {filename} ---")
    with open(filename, "rb") as f:
        resp = requests.post(f"{BASE_URL}/upload", files={"file": f})
    
    if resp.status_code == 200:
        print("✅ Upload Successful!")
        # Show columns to help you choose what to plot
        columns = resp.json()['preview']['columns']
        print(f"Available Columns: {columns}")
    else:
        print("❌ Upload Failed:", resp.text)
        return

    # --- Step 3: Dynamic Analysis ---
    target_col = input(f"\nWhich column do you want to analyze/plot? (Copy from above): ").strip()

    print("\n--- 2. Sending Analysis Code ---")
    # We construct the Python code dynamically based on your input
    code_analyze = f"""
print("Shape of DataFrame:", df.shape)
print("Description of '{target_col}':")
print(df['{target_col}'].describe())
    """
    
    resp = requests.post(f"{BASE_URL}/execute", json={"code": code_analyze})
    print("Output:\n", resp.json()['text_output'])

    # --- Step 4: Dynamic Plotting ---
    print("\n--- 3. Sending Plotting Code ---")
    code_plot = f"""
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
# auto-detect if categorical or numerical for better plotting
if df['{target_col}'].dtype == 'object':
    sns.countplot(y=df['{target_col}'])
else:
    sns.histplot(df['{target_col}'], kde=True)

plt.title("Distribution of {target_col}")
    """
    
    resp = requests.post(f"{BASE_URL}/execute", json={"code": code_plot})
    result = resp.json()
    
    if result['image_output']:
        output_filename = "plot_output.png"
        print(f"✅ Plot generated! Saving to '{output_filename}'...")
        img_data = base64.b64decode(result['image_output'])
        with open(output_filename, "wb") as f:
            f.write(img_data)
        print("Check the folder for the image file.")
    else:
        print("❌ No plot returned.")
        if result['error']:
            print("Error from Backend:", result['error'])

if __name__ == "__main__":
    test_workflow()