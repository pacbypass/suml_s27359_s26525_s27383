import logging
from flask import Flask, jsonify, render_template, request
from model import CatDogBreedClassifier

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif", "bmp"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
classifier = CatDogBreedClassifier()



@app.route("/")
def index():
    """serves the frontend"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """api endpoint which handles the call to predict() in model. returns the predictions as a JSON object."""
    if "image" not in request.files:
        return jsonify({"error": "no image sent"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "no file selected"}), 400
    if not ("." in file.filename and file.filename.rsplit(".", 1)[1].lower()
            in app.config["ALLOWED_EXTENSIONS"]):
        return jsonify({"error": "file type not allowed"}), 400

    try:
        logger.info("Processing uploaded image: %s", file.filename)
        predictions = classifier.predict(file.stream)
        return jsonify(predictions)

    except Exception as exc:
        logger.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
