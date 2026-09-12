import os
import uuid

from flask import Flask, request, jsonify
from flask_cors import CORS

from inference import predict_video
from image_inference import predict_image

app = Flask(__name__)

CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


ALLOWED_VIDEO_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm"
}


ALLOWED_IMAGE_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


def allowed_file(filename, extensions):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in extensions
    )


@app.route("/")
def home():
    return jsonify({
        "success": True,
        "message": "DeepFake Detection API is running"
    })


@app.route("/health")
def health():
    return jsonify({
        "success": True,
        "status": "healthy"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if "video" not in request.files:
        return jsonify({
            "success": False,
            "error": "No video uploaded."
        }), 400

    video = request.files["video"]

    if video.filename == "":
        return jsonify({
            "success": False,
            "error": "No video selected."
        }), 400

    if not allowed_file(
        video.filename,
        ALLOWED_VIDEO_EXTENSIONS
    ):
        return jsonify({
            "success": False,
            "error": "Unsupported video format."
        }), 400

    extension = video.filename.rsplit(
        ".",
        1
    )[1].lower()

    filename = f"{uuid.uuid4()}.{extension}"

    video_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    try:

        video.save(video_path)

        result = predict_video(video_path)

        return jsonify({
            "success": True,
            "result": result["label"],
            "confidence": result["confidence"],
            "fake_probability": result["fake_probability"],
            "real_probability": result["real_probability"],
            "type": "video"
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:

        if os.path.exists(video_path):
            os.remove(video_path)


@app.route("/predict-image", methods=["POST"])
def predict_image_route():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded."
        }), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({
            "success": False,
            "error": "No image selected."
        }), 400

    if not allowed_file(
        image.filename,
        ALLOWED_IMAGE_EXTENSIONS
    ):
        return jsonify({
            "success": False,
            "error": "Unsupported image format."
        }), 400

    extension = image.filename.rsplit(
        ".",
        1
    )[1].lower()

    filename = f"{uuid.uuid4()}.{extension}"

    image_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    try:

        image.save(image_path)

        result = predict_image(image_path)

        return jsonify({
            "success": True,
            "result": result["label"],
            "confidence": result["confidence"],
            "fake_probability": result["fake_probability"],
            "real_probability": result["real_probability"],
            "type": "image"
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:

        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )