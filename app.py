"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
import base64
from PIL import Image, ImageDraw

import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model(img, size=640) # reduce size=320 for faster inference

        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)

        # loop through detected objects and draw bounding boxes on the image
        for det in results.pandas().xyxy[0].to_dict(orient="records"):
            bbox = [
                det["xmin"], det["ymin"], det["xmax"], det["ymax"]
            ]
            draw.rectangle(bbox, outline="red")

        img_buffer = io.BytesIO()
        annotated_img.save(img_buffer, format="JPEG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        response = {
            'image': img_str,
            'objects': results.pandas().xyxy[0].to_dict(orient="records")
        }
        return jsonify(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--model', default='yolov5s', help='model to run, i.e. --model yolov5s')
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', args.model)
    app.run(host="0.0.0.0", port=args.port)
