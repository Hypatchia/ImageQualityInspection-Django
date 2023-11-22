from azure.storage.blob import BlobServiceClient
from django.conf import settings
import os
from pathlib import Path


def upload_h5_to_azure(file_path, blob_name):
    connection_string = settings.AZURE_BLOB_STORAGE_CONNECTION_STRING
    container_name = settings.AZURE_BLOB_CONTAINER_NAME
    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    try:
      
        with open(file_path, "rb") as data:
            print("Uploading H5 file to Azure Blob Storage..., wait for it...")
            print('file_path: ', file_path)
            container_client.upload_blob(name=blob_name, data=data)
        return True  # Upload successful
    except Exception as e:
        print(f"Error uploading H5 file to Azure Blob Storage: {str(e)}")
        return False  # Upload failed

# Example usage:
if __name__ == "__main__":
    models_dir = os.path.join(settings.BASE_DIR, 'ml_models')
    
    model_path = os.path.join(models_dir, 'ImageClassifier.h5')
    
    file_path = model_path
    blob_name = "MulticlassCNNClassifierModel.h5"
    if upload_h5_to_azure(file_path, blob_name):
        print("H5 file uploaded successfully to Azure Blob Storage")
    else:
        print("H5 file upload to Azure Blob Storage failed")
