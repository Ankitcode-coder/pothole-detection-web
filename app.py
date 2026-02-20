from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Load trained model
model = tf.keras.models.load_model("pothole_model.h5")


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = 0
    pothole_frames = 0
    frame_count = 0   

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 30 != 0:
            continue

        total_frames += 1

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        prediction = model.predict(frame, verbose=0)[0][0]

        if prediction > 0.5:
            pothole_frames += 1

    cap.release()

    damage_percent = (pothole_frames / total_frames) * 100 if total_frames > 0 else 0

    if damage_percent < 20:
        condition = "Good Road"
        bar_class = "good"
    elif damage_percent < 50:
        condition = "Moderate Damage"
        bar_class = "moderate"
    else:
        condition = "Severe Damage"
        bar_class = "severe"

    return total_frames, pothole_frames, round(damage_percent, 2), condition, bar_class


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video = request.files["video"]

        if video:
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
            video.save(video_path)

            total, potholes, percent, condition, bar_class = process_video(video_path)

            return render_template(
                "index.html",
                total=total,
                potholes=potholes,
                percent=percent,
                condition=condition,
                bar_class=bar_class
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)