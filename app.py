from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image
import io, base64
import numpy as np
from text_to_price_api.service import PricePredictor

app = Flask(__name__)

yolo_model = YOLO("weights/best.pt")
text_model = PricePredictor(model_dir="text_to_price_api/models/")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    desc = request.form.get("description", "")
    if not desc.strip():
        return "No description", 400

    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = yolo_model(img)

    res_plotted = results[0].plot()
    res_image = Image.fromarray(res_plotted[..., ::-1])
    img_io = io.BytesIO()
    res_image.save(img_io, 'JPEG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    defect_count = len(results[0])
    deduction_amount = defect_count * 1000

    base_price_text, cleaned_text = text_model.predict(desc)
    base_price_text *= 145

    final_price = base_price_text - deduction_amount

    if final_price < 0:
        final_price = 0

    return render_template(
        "predict.html",
        result_image=img_base64,
        price=round(final_price, 2), 
        base_price=round(base_price_text, 2), 
        deduction=deduction_amount,           
        defect_count=defect_count,            
        description=desc
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
