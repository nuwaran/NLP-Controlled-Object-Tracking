#All the codes are refined with LLm    --------<<<<<<<<<<<<<<<<<<<<<<
import os
import cv2
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from ultralytics import YOLO
from datetime import timedelta
import numpy as np
import json

# Disable proxy environment variables
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

# Load environment variables
load_dotenv()

# Initialize Gemini client
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Enable Gemini API and set GEMINI_API_KEY in .env.")
    exit()

# Load YOLO model
try:
    model_yolo = YOLO("yolo11n.pt")  # Replace with your model
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Color detection in HSV
def detect_color(img, bbox, target_color):
    x1, y1, x2, y2 = map(int, bbox)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "red": [(np.array([0, 120, 70]), np.array([10, 255, 255])),
                (np.array([170, 120, 70]), np.array([180, 255, 255]))],
        "blue": [(np.array([100, 120, 70]), np.array([130, 255, 255]))],
        "green": [(np.array([40, 40, 70]), np.array([80, 255, 255]))],
        "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
        "white": [(np.array([0, 0, 200]), np.array([180, 30, 255]))],
        "black": [(np.array([0, 0, 0]), np.array([180, 255, 30]))],
        "orange": [(np.array([10, 100, 100]), np.array([20, 255, 255]))],
        "purple": [(np.array([130, 120, 70]), np.array([160, 255, 255]))]
    }

    if target_color not in color_ranges:
        return False

    mask = None
    for lower, upper in color_ranges[target_color]:
        if mask is None:
            mask = cv2.inRange(hsv, lower, upper)
        else:
            mask += cv2.inRange(hsv, lower, upper)

    return np.mean(mask) > 10  # Threshold

def parse_user_intent(prompt):
    try:
        system_prompt = """
        Extract a structured command from the user prompt.

        Respond ONLY with a JSON object in the format:
        {
          "action": "track" or "count",
          "label": "car" or other YOLO label,
          "color": "red", "blue", etc.
        }

        If you cannot determine the action, label, or color, return:
        {"action": "unknown", "label": "", "color": ""}

        Do not explain anything. Only return valid JSON with double quotes.
        """
        response = model.generate_content([system_prompt, prompt])
        raw_text = response.text.strip()

        # Remove markdown code block if present
        if raw_text.startswith("```") and raw_text.endswith("```"):
            raw_text = "\n".join(raw_text.split("\n")[1:-1]).strip()

        print("Gemini cleaned response:", raw_text)
        return json.loads(raw_text)
    except Exception as e:
        print(f"Gemini intent parsing failed: {e}")
        return {"action": "unknown", "label": "", "color": ""}


# Track objects in video
def track_object(video_path, label_to_track, color_to_track):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    detections = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolo.predict(source=frame, conf=0.5, verbose=False)
        names = model_yolo.names

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = names[class_id]
                if class_name.lower() == label_to_track:
                    x1, y1, x2, y2 = box.xyxy[0]
                    if color_to_track and not detect_color(frame, (x1, y1, x2, y2), color_to_track):
                        continue
                    timestamp = timedelta(seconds=frame_idx / fps)
                    detections.append([f"{color_to_track} {class_name}", str(timestamp)])
                    print(f"Detected {color_to_track} {class_name} at {timestamp}")
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{color_to_track} {class_name} {box.conf.item():.2f}",
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Video Object Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if detections:
        df = pd.DataFrame(detections, columns=["Object", "Time Detected"])
        df.to_csv("tracking_log.csv", index=False)
        print("Tracking completed. Log saved to tracking_log.csv.")
    else:
        print("No detections found.")

# Count from tracking log
def count_tracked_objects(color, label):
    try:
        df = pd.read_csv("tracking_log.csv")
        keyword = f"{color} {label}".lower()
        count = df["Object"].str.lower().str.contains(keyword).sum()
        print(f"Total '{keyword}' detections: {count}")
    except FileNotFoundError:
        print("Tracking log not found. Please run tracking first.")

# Main program
if __name__ == "__main__":
    user_prompt = input("What do you want to do?\n")
    parsed = parse_user_intent(user_prompt)

    action = parsed.get("action")
    label = parsed.get("label")
    color = parsed.get("color")

    if action == "unknown" or not label or not color:
        print("Could not understand your prompt. Please try again with an object and color.")
        exit()

    print(f"Action: {action}, Label: {label}, Color: {color}")

    if action == "track":
        video_path = r"C:\Users\M S I\PycharmProjects\llmcv\testv.mp4"
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist")
            exit()
        track_object(video_path, label, color)

    elif action == "count":
        count_tracked_objects(color, label)

    else:
        print("Unsupported action.")
