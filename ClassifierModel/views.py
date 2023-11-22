
# Create your views here.

from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
import io
def home(request):
    return HttpResponse("This is home")

from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from PIL import Image
from django.shortcuts import render, redirect
from .forms import UploadImageForm
from .models import UploadedImage 
from django.db.models import Max
from .image_classifier import ImageClassifier

import numpy as np
import tensorflow as tf
import os
import h5py
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from django.http import HttpResponse
from django.conf import settings
from .secrets import secrets
from azure.storage.blob import BlobServiceClient


#AZURE_BLOB_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=classifiersblobstorage;AccountKey=sMp0A9HoYAJv0d5+nm5WR2VhAyEbliKMoz91jSU8bUM7VwWZbsznKDMd0yMhsMhfxIW3uLPPINaL+ASt7CndiA==;EndpointSuffix=core.windows.net"
#AZURE_BLOB_CONTAINER_NAME = "qualityinspection"
#AZURE_BLOB_CONTAINER_NAME_MEDIA ="qualityinspectionmedia"

#AZURE_BLOB_STORAGE_CONNECTION_STRING = os.environ['AZURE_BLOB_STORAGE_CONNECTION_STRING']
#AZURE_BLOB_CONTAINER_NAME_MEDIA =  os.environ['AZURE_BLOB_CONTAINER_NAME_MEDIA']
#AZURE_BLOB_CONTAINER_NAME = os.environ['AZURE_BLOB_CONTAINER_NAME']
#AZURE_BLOB_STORAGE_CONNECTION_STRING = settings.AZURE_BLOB_STORAGE_CONNECTION_STRING
#AZURE_BLOB_CONTAINER_NAME_MEDIA =  settings.AZURE_BLOB_CONTAINER_NAME_MEDIA
#AZURE_BLOB_CONTAINER_NAME = settings.AZURE_BLOB_CONTAINER_NAME



AZURE_BLOB_STORAGE_CONNECTION_STRING =secrets.get('AZURE_BLOB_STORAGE_CONNECTION_STRING')
AZURE_BLOB_CONTAINER_NAME_MEDIA = secrets.get('AZURE_BLOB_CONTAINER_NAME_MEDIA')
AZURE_BLOB_CONTAINER_NAME = secrets.get('AZURE_BLOB_CONTAINER_NAME')

def load_model_from_blob():
    
    connection_string = AZURE_BLOB_STORAGE_CONNECTION_STRING
    container_name = AZURE_BLOB_CONTAINER_NAME
    blob_name = "FinalMulticlass.h5"
    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)
    print('Container is loading...')
    # Get a reference to the blob
    blob_client = container_client.get_blob_client(blob_name)  # Use the blob_name variable here
    # Download the H5 file to a bytes buffer
    blob_data = blob_client.download_blob()
    print('Downloading the model...')
    model_bytes = io.BytesIO()
    blob_data.readinto(model_bytes)
    model_bytes.seek(0)  # Reset the buffer position to the beginning
    # Load the model from the bytes buffer using h5py

    with h5py.File(model_bytes, 'r') as f:
        loaded_model = tf.keras.models.load_model(f)
    #loaded_model = tf.keras.models.load_model(model_bytes)
    # Download the H5 file to a local temporary file
    # Ensure the 'temp' directory exists
    #temp_dir = os.path.join(settings.BASE_DIR, 'temp')
    #os.makedirs(temp_dir, exist_ok=True)
    #local_file_path = os.path.join(temp_dir, blob_name) # Use blob_name here
    #with open(local_file_path, "wb") as my_blob:
    #    blob_data = blob_client.download_blob()
    #    blob_data.readinto(my_blob)

    # Load the model from the downloaded file
    #loaded_model = tf.keras.models.load_model(local_file_path)

    return loaded_model

loaded_model = load_model_from_blob()
input_shape = loaded_model.input_shape[1:3] 
#models_dir = os.path.join(settings.BASE_DIR, 'ClassifierModel','static', 'ml_models')
#model_path = os.path.join(models_dir, 'ImageClassifier.h5')
#loaded_model = tf.keras.models.load_model(model_path)
def preprocess_image(image, target_size):
    # Resize the image to the target size
    image = np.array(image)/255.0
    resized_image = tf.image.resize(image, target_size)
    # Convert the image to dtype tf.float32 
    preprocessed_image = tf.cast(resized_image, tf.float32) 

    return preprocessed_image



# function to convert rgb images to grayscale using tf
def convert_to_grayscale(image):
    grayscale_image = tf.image.rgb_to_grayscale(image)
    return grayscale_image


def upload_image_to_blob(file_content, blob_name):
    
    connection_string = AZURE_BLOB_STORAGE_CONNECTION_STRING
    container_name = AZURE_BLOB_CONTAINER_NAME_MEDIA
    print('Accessing blob storage...')
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    print('Accessing blob storage done')
    container_client = blob_service_client.get_container_client(container_name)
    print('Uploading image to blob storage...')
    try:
        container_client.upload_blob(name=blob_name, data=file_content,connection_timeout=14400,overwrite=True)
        print('Uploading image to blob storage done')
        return True  # Upload successful
    except Exception as e:
        print(f"Error uploading image to Azure Blob Storage: {str(e)}")
        return False  # Upload failed

    # views.py


from django.utils import timezone

import threading

def process_and_upload_in_background(original_image_data, blob_name):
    upload_result = upload_image_to_blob(original_image_data, blob_name)



def upload_image(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
                print('loading Model')
                
                print('loading done,processing img')
                # Get the URL of the uploaded image
                #image = request.FILES['image']
                #image_data = image.read()
                image_instance = form.save(commit=False)
                image = Image.open(image_instance.image)
                #image_array = np.array(image)
                #grayscale = convert_to_grayscale(image)
                grayscale = image
                preprocessed_image = preprocess_image(image, target_size=input_shape)
                #grayscale = convert_to_grayscale(image_data)
                #preprocessed_image = preprocess_image(grayscale, target_size=input_shape)
                preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
                print('predicting...')
                prediction = loaded_model.predict(preprocessed_image)# Replace with actual prediction code
                print('prediction done')
                class_labels = ['Broken', 'Defected', 'Good']
                predicted_label = class_labels[np.argmax(prediction)]


                image_instance.save()
                # Convert the original image to bytes
                original_image_data = image.tobytes()
                # Create a unique blob name (e.g., using a timestamp)
                timestamp = timezone.now().strftime("%Y%m%d%H%M%S")
                #blob_name = f"{timestamp}_{form.cleaned_dat    a['image'].name}"
                # Create a unique blob name based on the prediction label
                blob_name = f"{timestamp}_{predicted_label}"
                #blob_name = f"test_upload"
                # Upload the image to Azure Blob Storage
                #upload_result = upload_image_to_blob(form.cleaned_data['image'].read(), blob_name)
              
               
                
                print('upload success')
            
                # Image was successfully uploaded to Azure Blob Storage
                # Save the image instance with the blob_name
                image_instance.blob_name = blob_name
                image_instance.save()
                # Schedule the image upload as a background task
            
                response_data = {
                'prediction': predicted_label,
                'prediction_probabilityBroken': int(prediction.tolist()[0][0]*10000)/10000.0,
                'prediction_probabilityFlawed': int(prediction.tolist()[0][1]*10000)/10000.0,
                'prediction_probabilityGood': int(prediction.tolist()[0][2]*10000)/10000.0,
                }
                  # Create a thread to run the background processing and uploading
                processing_thread = threading.Thread(
                    target=process_and_upload_in_background,
                    args=(original_image_data, blob_name)
                )
                processing_thread.start()
       
                print('upload_to_blob...')
                
                
                print('upload_to_blob done')
                print('Returning response data...')  
                
                return JsonResponse(response_data)
               

               
    else:
        form = UploadImageForm()
    return render(request, 'ClassifierModel/upload_image.html', {'form': form})

