from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf   # Use tensorflow locally for TFLite interpreter
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

interpreter = tf.lite.Interpreter(model_path="pothole_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_frame(frame):
    # Resize to 128x128 (YOUR MODEL SIZE)
    frame = cv2.resize(frame, (128, 128))

    # Normalize
    frame = frame / 255.0

    # Expand dimensions
    frame = np.expand_dims(frame, axis=0).astype(np.float32)

    # Set tensor
    interpreter.set_tensor(input_details[0]['index'], frame)

    # Run inference
    interpreter.invoke()

    # Get output
    prediction = interpreter.get_tensor(output_details[0]['index'])

    return float(prediction[0][0])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            return render_template("index.html")

        file = request.files["video"]

        if file.filename == "":
            return render_template("index.html")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)

        total_frames = 0
        pothole_frames = 0
        frame_skip = 10

        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            if total_frames % frame_skip == 0:
                processed_frames += 1

                prediction = predict_frame(frame)

                if prediction > 0.5:
                    pothole_frames += 1

        cap.release()

        # Prevent division by zero
        if processed_frames == 0:
            damage_percentage = 0
        else:
            damage_percentage = (pothole_frames / processed_frames) * 100

        damage_percentage = round(damage_percentage, 2)

        # Road condition logic
        if damage_percentage < 20:
            road_condition = "Good Road"
        elif 20 <= damage_percentage <= 50:
            road_condition = "Moderate Damage"
        else:
            road_condition = "Severe Damage"

        return render_template(
            "index.html",
            total_frames=total_frames,
            pothole_frames=pothole_frames,
            damage_percentage=damage_percentage,
            road_condition=road_condition
        )

    return render_template("index.html")



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)