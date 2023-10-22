from django import forms
from .models import UploadedImage

class UploadImageForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['image']

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            if image.size > 24 * 1024 * 1024:  # 24MB
                raise forms.ValidationError("File size exceeds the maximum limit of 24MB.")
        return image
