from google.oauth2 import service_account
from googleapiclient.discovery import build
import streamlit as st

def check_google_file(file_id):
    try:
        # Load credentials securely from Streamlit Secrets
        creds_dict = st.secrets["google"]
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )

        # Build Google Drive API client
        service = build("drive", "v3", credentials=credentials)
        file = service.files().get(fileId=file_id, fields="name, mimeType, size, trashed").execute()

        # Safety checks
        if file.get("trashed"):
            return {"status": "error", "message": f"🚫 File '{file['name']}' is in trash"}
        if not file["mimeType"].startswith("text/") and not file["mimeType"].endswith("csv"):
            return {"status": "error", "message": f"❌ Unsupported file type: {file['mimeType']}"}
        if int(file.get("size", 0)) > 10_000_000:
            return {"status": "warning", "message": "⚠️ File is large (>10MB). Proceed with caution."}

        return {"status": "ok", "message": f"✅ File '{file['name']}' passed Google safety checks."}

    except Exception as e:
        return {"status": "error", "message": f"Google API Error: {e}"}
