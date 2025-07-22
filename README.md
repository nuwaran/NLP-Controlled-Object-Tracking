# 🧠 NLP-Controlled Real-Time Object Tracking System

A self-initiated research project (July 2025) that explores the fusion of **Natural Language Processing (LLMs)** with **Real-Time Computer Vision**. This system allows a user to control object tracking through natural language commands — eliminating the need for traditional UI or hardcoded logic.

## 🚀 Key Features

- 🔍 **Prompt-Based Control**: Track objects based on commands like `"Track red cars"` or `"Count people crossing the line."`
- ⚙️ **YOLOv11 Integration**: High-speed object detection for real-time video analysis.
- 🧩 **Modular Architecture**: Easily extendable prompt-to-tracking system.
- 📊 **Logging System**: Automatically logs detections with timestamps in CSV format.
- 🛠️ **Application Use Cases**:
  - Smart traffic monitoring
  - Security and surveillance automation
  - Crowd density estimation and event analysis

## 💡 System Workflow

1. User inputs a natural language command.
2. LLM parses it to identify task (e.g., count, detect), object class (e.g., car, person), and filters (e.g., red, large).
3. YOLOv11 detects objects in real-time.
4. Task-specific actions are executed based on parsed intent (e.g., tracking, logging, counting).

