import h5py
import io
import os
from azure.storage.blob import BlobServiceClient
import tensorflow as tf
from django.conf import settings

AZURE_BLOB_STORAGE_CONNECTION_STRING = ""


AZURE_BLOB_CONTAINER_NAME = ""

connection_string = AZURE_BLOB_STORAGE_CONNECTION_STRING
container_name = AZURE_BLOB_CONTAINER_NAME
import h5py
import io
import os
from azure.storage.blob import BlobServiceClient
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

def load_model_from_blob_parallel(workers=16):
    connection_string = AZURE_BLOB_STORAGE_CONNECTION_STRING
    container_name = AZURE_BLOB_CONTAINER_NAME
    blob_name = "ImageClassifier.h5"

    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)
    print('Container is loading...')

    # Get a reference to the blob
    blob_client = container_client.get_blob_client(blob_name)

    # Download the model in chunks using parallel workers
    print('Downloading the model in parallel...')
    model_bytes = io.BytesIO()
    chunk_size = 16 * 1024 * 1024  # Adjust chunk size as needed
    offset = 0

    def download_chunk(offset):
        chunk = blob_client.download_blob(offset=offset, length=chunk_size)
        chunk_data = chunk.readall()
        return offset, chunk_data

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        while True:
            futures.append(executor.submit(download_chunk, offset))
            offset += chunk_size
            if offset >= blob_client.get_blob_properties().size:
                break

        for future in futures:
            offset, chunk_data = future.result()
            model_bytes.write(chunk_data)

    model_bytes.seek(0)  # Reset the buffer position to the beginning

    # Load the model from the bytes buffer using h5py
    with h5py.File(model_bytes, 'r') as f:
        loaded_model = tf.keras.models.load_model(f)

    # Save loaded
    loaded_model.save('loaded_model.h5')
    return loaded_model

# Call the function to load the model in parallel with 8 workers
loaded_model = load_model_from_blob_parallel(workers=16)
