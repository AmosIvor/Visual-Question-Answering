import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
img_url = 'https://t3.ftcdn.net/jpg/01/02/64/28/360_F_102642850_Mca9lTRDH60DQin39YwCF5Jzd15lcdoo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# question = "how many people are in the picture?"
question = "What color of the sky?"
inputs = processor(raw_image, question, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))