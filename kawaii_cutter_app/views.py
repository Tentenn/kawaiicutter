import os
import random

from django.conf import settings
from .forms import UploadImageForm
from django.shortcuts import render
from .aniseg.inference import Segmentor

from datetime import datetime
import shutil




def add_log(log_text:str):
    # Get the current date and time
    now = datetime.now()

    # Format it as per your requirement
    formatted_date = now.strftime('%H:%M:%S_%d-%m-%y')

    with open("logs.txt", "a") as log:
        log.write(f"# {formatted_date}# {log_text}\n")



def generate_image(request):
    add_log("Generate image called")
    generated_image_path = None
    file_path = None
    original_image_url = None

    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        add_log("Checking if valid log")
        if form.is_valid():
            add_log("Valid Log Form")
            # Generate a random string of 12 digits for filename
            random_filename = ''.join(random.choices('0123456789', k=12))
            ext = os.path.splitext(request.FILES['image'].name)[1]
            filename = f"{random_filename}{ext}"

            # Save the image to the specified folder
            file_path = os.path.join(settings.MEDIA_ROOT, filename)

            # Ensure the destination directory exists
            dest_dir = os.path.join(settings.STATIC_ROOT, 'generated_images')
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # Save the image to the specified folder
            with open(file_path, 'wb+') as destination:
                for chunk in request.FILES['image'].chunks():
                    destination.write(chunk)

            image_path = os.path.join(settings.MEDIA_ROOT, filename)

            # Pass the path to the segmentor.segment() function
            add_log("Masking...")
            segmentor = Segmentor(settings.MODEL_CHECKPOINT_PATH)
            masked_file_name=filename.rstrip(ext)+f"-masked{ext}"
            segmentor.segment(image_path, file_name=masked_file_name)
            add_log("Masking finished")

            # Note: For URLs, you should use STATIC_URL
            generated_image_path = os.path.join(settings.STATIC_URL, 'generated_images', masked_file_name)
            original_image_url = os.path.join(settings.MEDIA_URL, filename)
        else:
            # Handle the errors
            add_log(str(form.errors))

    return render(request, 'generate_image.html', {'form': UploadImageForm(), 'generated_image': generated_image_path, 'original_image': original_image_url})