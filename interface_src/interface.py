from flask import Flask, redirect, render_template, request, url_for
from pathlib import Path
import time
import threading
import base64
from io import BytesIO
import pyautogui
import requests
# from utils.screenshot_utils import ChromadbClient

# Making an application with only a button on it
app = Flask(__name__)

running = False
saving_path = Path("data/")
# chromadb_client = ChromadbClient()


@app.route("/", methods=["GET"])
def home():
    if request.method == "GET":
        return render_template("home.html")


@app.route("/start", methods=["POST"])
def start():
    
    global running
    running = True
    thread = threading.Thread(target=capture_screen)
    thread.start()
    return redirect(url_for("home"))


@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return redirect(url_for("home"))


def capture_screen():
    global running, saving_path, chromadb_client
    while running:
        try:
            print("Taking a photo...")
            buffered = BytesIO()
            screen_image = pyautogui.screenshot()
            screen_image.save(buffered, format="JPEG")
            buffered.seek(0)

            api_url = "http://127.0.0.1:5001/describe_image"
            response = requests.post(api_url, data={"image": ("screenshot.jpg", buffered, "image/jpeg")})

            if response.status_code == 200:
                print("Image processed successfully.")
            else:
                print(f"Error processing image: {response.status_code}")

        except Exception as e:
            print(f"An error occurred while taking a photo. {e}")
        finally:
            time.sleep(10)


if __name__ == "__main__":
    app.run(debug=True)
