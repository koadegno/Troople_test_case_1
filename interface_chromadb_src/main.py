import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64
import io

from utils.screenshot_utils import ChromadbClient


app = Flask(__name__)
CORS(app)

client = ChromadbClient()


@app.route("/", methods=["GET"])
def home():
    print("Server Up")


@app.route("/describe_image", methods=["POST"])
def describe_image():
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files["image"]
    description = ""

    img = Image.open(file.stream)
    description = client.describe_image(img)

    # Add the image to Chromadb
    img_id = "image_" + str(datetime.datetime.now().timestamp())
    client.add_image(img, img_id)

    return jsonify({"description": description})

@app.route("/ask", methods=["POST"])

def question():
    if "question" not in request.json:
        return jsonify({"error": "No question provided"}), 400
    
    question = request.json["question"]
    answer = client.ask(question)
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True,port=5001)
