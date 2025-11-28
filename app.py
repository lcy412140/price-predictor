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

   # 2. 讀取圖片並進行預測
   img_bytes = file.read()
   img = Image.open(io.BytesIO(img_bytes))
   
   # 執行預測
   results = model(img)
   
   # 3. 處理結果
   # 您原本用 save=True，但在網頁上我們需要把圖傳回去
   # result.plot() 會回傳一個畫好框/mask 的 numpy array (BGR)
   res_plotted = results[0].plot()
   
   # 將 Numpy array 轉回圖片格式以傳輸
   res_image = Image.fromarray(res_plotted[..., ::-1]) # RGB <-> BGR
   
   # 轉為 Base64 字串直接以此顯示在網頁上 (不需存檔)
   img_io = io.BytesIO()
   res_image.save(img_io, 'JPEG')
   img_io.seek(0)
   img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
   
   # 4. 回傳給 predict.html 渲染
   # 假設您想根據缺陷數量算價格，可以在這裡寫邏輯
   defect_count = len(results[0]) 
   estimated_price = 114514 - (defect_count * 1000) # 範例邏輯

   return render_template('predict.html', 
                        result_image=img_base64, 
                        price=estimated_price)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
