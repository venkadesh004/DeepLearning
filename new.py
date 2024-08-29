from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2
import torch

processor = AutoImageProcessor.from_pretrained("RavenOnur/Sign-Language")
model = AutoModelForImageClassification.from_pretrained("RavenOnur/Sign-Language")

def process_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = processor(img_rgb, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**result).logits
                
    predicted_label = logits.argmax(1).item()
    print(model.config.id2label[predicted_label])

image = cv2.imread("./images/L.jpg")
process_image(image)

image = cv2.imread("./images/W.jpg")
process_image(image)