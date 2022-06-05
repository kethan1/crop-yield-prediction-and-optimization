from flask import Flask, render_template, request
import numpy as np
import cv2
import joblib

app = Flask(__name__)
pepper_model = joblib.load("models/pepper.joblib")
pepper_labels = ["healthy", "bacterial_spot"]
potato_model = joblib.load("models/potato.joblib")
potato_labels = ["early_blight", "healthy", "late_blight"]
tomato_model = joblib.load("models/tomato.joblib")
tomato_labels = [
    "bacterial_spot",
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold",
    "mosaic_virus",
    "septoria_leaf_spot",
    "spider_mites",
    "target_spot",
    "yellow_leaf_curl_virus",
]
crop_yield_prediction_model = joblib.load("models/crop_yield_predictor.joblib")
crop_yield_predictor_labels = [
    "Maize",
    "Potatoes",
    "Rice, paddy",
    "Sorghum",
    "Soybeans",
    "Wheat",
    "Cassava",
    "Sweet potatoes",
    "Yams",
    "Plantains and others",
]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/yield_prediction", methods=["GET"])
def yield_prediction():
    return render_template("yield_prediction.html", crops=crop_yield_predictor_labels)


@app.route("/disease_detection", methods=["GET"])
def disease_detection():
    return render_template("disease_detection.html")


@app.route("/api/v1/yield_prediction", methods=["POST"])
def yield_prediction_api():
    crop = request.json["crop"]
    avg_rainfall = request.json["avg_rainfall"]
    pesticides = request.json["pesticides"]
    avg_temp = request.json["avg_temp"]
    data = np.array(
        [crop_yield_predictor_labels.index(crop), avg_rainfall, pesticides, avg_temp]
    ).reshape(1, -1)
    return {"yield": crop_yield_prediction_model.predict(data).tolist()}


@app.route("/api/v1/yield_optimizer", methods=["POST"])
def yield_optimizer_api():
    crop = request.json["crop"]
    avg_rainfall = request.json["avg_rainfall"]
    pesticides = request.json["pesticides"]
    avg_temp = request.json["avg_temp"]
    increase_avg_rainfall_by_100_data = np.array(
        [
            crop_yield_predictor_labels.index(crop),
            avg_rainfall + 100,
            pesticides,
            avg_temp,
        ]
    ).reshape(1, -1)
    increase_pesticides_by_50_data = np.array(
        [
            crop_yield_predictor_labels.index(crop),
            avg_rainfall,
            pesticides + 50,
            avg_temp,
        ]
    ).reshape(1, -1)
    increase_pesticides_by_100_data = np.array(
        [
            crop_yield_predictor_labels.index(crop),
            avg_rainfall,
            pesticides + 100,
            avg_temp,
        ]
    ).reshape(1, -1)
    return {
        "increase_avg_rainfall_by_100": crop_yield_prediction_model.predict(
            increase_avg_rainfall_by_100_data
        ).tolist(),
        "increase_pesticides_by_100_data": crop_yield_prediction_model.predict(
            increase_pesticides_by_100_data
        ).tolist(),
        "increase_pesticides_by_50_data": crop_yield_prediction_model.predict(
            increase_pesticides_by_50_data
        ).tolist(),
    }


@app.route("/api/v1/disease_detection", methods=["POST"])
def disease_detection_api():
    crop = request.form.get("crop")
    file = request.files["image"]
    if crop == "pepper":
        model = pepper_model
        labels = pepper_labels
    elif crop == "potato":
        model = potato_model
        labels = potato_labels
    elif crop == "tomato":
        model = tomato_model
        labels = tomato_labels
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    return {"disease_detected": labels[
        min(round(model.predict(image.flatten().reshape(1, -1))[0]), len(labels) - 1)
    ]}


if __name__ == "__main__":
    app.run(debug=True, port=9000)
