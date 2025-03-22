import os
from appwrite.client import Client
from appwrite.services.storage import Storage
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT")
PROJECT_ID = os.getenv("PROJECT_ID")
API_KEY = os.getenv("API_KEY")
BUCKET_ID = os.getenv("BUCKET_ID")

# Initialize Appwrite Client
client = Client()
client.set_endpoint(APPWRITE_ENDPOINT)
client.set_project(PROJECT_ID)
client.set_key(API_KEY)

# Initialize Storage
storage = Storage(client)

# Folder to store dataset
download_folder = "dataset/images"
os.makedirs(download_folder, exist_ok=True)

# Fetch all files from the bucket
files = storage.list_files(BUCKET_ID)

# Download each file
for file in files["files"]:
    file_id = file["$id"]
    file_name = file["name"]
    file_path = os.path.join(download_folder, file_name)

    # Download file
    with open(file_path, "wb") as f:
        response = storage.get_file_download(BUCKET_ID, file_id)
        f.write(response)

    print(f"✅ Downloaded: {file_name}")

print("🎉 Dataset successfully downloaded to SageMaker!")
