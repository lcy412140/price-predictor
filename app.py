from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np
import cv2

app = Flask(__name__)

model = YOLO("weights/best.pt")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   if 'file' not in request.files:
      return "No file uploaded", 400
   
   file = request.files['file']
   if file.filename == '':
      return "No selected file", 400

   img_bytes = file.read()
   img = Image.open(io.BytesIO(img_bytes))

   results = model(img)
  
   res_plotted = results[0].plot()
   
   res_image = Image.fromarray(res_plotted[..., ::-1])
   
   img_io = io.BytesIO()
   res_image.save(img_io, 'JPEG')
   img_io.seek(0)
   img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
   
   defect_count = len(results[0]) 
   estimated_price = 114514 - (defect_count * 1000)

   return render_template('predict.html', 
                        result_image=img_base64, 
                        price=estimated_price)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
