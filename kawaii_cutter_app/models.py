from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploaded_images/')
    class Meta:
        app_label = 'kawaii_cutter_app'
