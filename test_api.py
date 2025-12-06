import requests
import pandas as pd
import io

# Create a sample CSV file for testing
csv_data = """name,age,salary,department
John,25,50000,Engineering
Jane,30,60000,Marketing
Bob,35,70000,Engineering
Alice,28,55000,Design
Charlie,32,65000,Sales"""

# Save test CSV
with open('test_data.csv', 'w') as f:
    f.write(csv_data)

# Test the upload endpoint
url = "http://localhost:8000/upload"

with open('test_data.csv', 'rb') as f:
    files = {'file': ('test_data.csv', f, 'text/csv')}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response:")
print(response.json())

# Clean up
import os
os.remove('test_data.csv')
