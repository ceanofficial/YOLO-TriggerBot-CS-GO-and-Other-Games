# 🧠 YOLOv10 TriggerBot for Object Detection

This Python app uses a custom-trained YOLOv10 model to detect specific objects (e.g., a person) on your screen in real time. When the detected object appears near the center of the screen, it automatically clicks the mouse. Ideal for automating interactions in screen-based environments like games, surveillance, or desktop automation.

## 🚀 Features

- Real-time object detection using the **YOLOv10** model (`sneaky.pt`)
- Captures screen with `mss`
- Automatically simulates a mouse click when a detected object is centered
- Works with both GPU and CPU (CUDA support)

## 📦 Requirements

- Python 3.8+
- `torch`
- `numpy`
- `opencv-python`
- `mss`
- `ultralytics` (for YOLOv10)
- `pyautogui`

Install dependencies:

```bash
pip install torch numpy opencv-python mss ultralytics pyautogui
```

> 💡 For best performance, ensure you have a CUDA-compatible GPU and appropriate PyTorch installation.

## 🔧 Setup

1. **Custom Model**: Make sure your YOLOv10 model (`sneaky.pt`) is trained and available in the same directory.
2. **Adjust Coordinates**: The script checks if an object is detected near the screen center (`center_x = 959`, `center_y = 540`), and performs a click if so.
3. **Class ID**: The detection is filtered for a specific class ID (`PERSON_CLASS_ID = 7`). You may need to change this to match your dataset.

## ▶️ How to Run

```bash
python auto_clicker_yolo.py
```

Press `q` to exit the app.

## 📄 How It Works

- Captures the entire screen using `mss`
- Converts the screenshot to a PyTorch tensor
- Passes it to the YOLOv10 model
- For every detection:
  - If it matches the desired class ID and confidence threshold
  - If it’s near the center of the screen
  - It triggers a left mouse click using Windows API (`ctypes`)

## 💻 Example Output

When the target is detected near the center:

```
Clicked at screen center: (959, 540) with confidence 0.873
```

## 🛠 Code

```python
import torch
import numpy as np
import cv2
import pyautogui
from ultralytics import YOLO  
import ctypes
from ctypes import wintypes
from mss import mss

model = YOLO('sneaky.pt')

if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

CONFIDENCE_THRESHOLD = 0.001
PERSON_CLASS_ID = 7  

center_x, center_y = 959, 540

def get_screen():
    with mss() as sct:
        monitor = sct.monitors[1]  
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)  
        return cv2.resize(frame, (1920, 1056))  

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class INPUT(ctypes.Structure):
    class _InputUnion(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("_iu",)
    _fields_ = [("type", wintypes.DWORD), ("_iu", _InputUnion)]

INPUT_MOUSE = 0
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

def click_mouse():
    inp_down = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dwFlags=MOUSEEVENTF_LEFTDOWN))
    inp_up = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dwFlags=MOUSEEVENTF_LEFTUP))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(inp_down), ctypes.sizeof(inp_down))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(inp_up), ctypes.sizeof(inp_up))

def is_cursor_in_bbox(cursor_x, cursor_y, x_min, y_min, x_max, y_max):
    return x_min <= cursor_x <= x_max and y_min <= cursor_y <= y_max

while True:
    frame = get_screen()
    frame_tensor = torch.tensor(frame).permute(2, 0, 1).float() / 255

    if torch.cuda.is_available():
        frame_tensor = frame_tensor.cuda()

    with torch.no_grad():
        results = model(frame_tensor.unsqueeze(0))

    if isinstance(results, list) and len(results) > 0:
        result = results[0]
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                class_id = int(box.cls.item())

                bbox_center_x, bbox_center_y = (x_min + x_max) / 2, (y_min + y_max) / 2

                if class_id == PERSON_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
                    if abs(bbox_center_x - center_x) < 50 and abs(bbox_center_y - center_y) < 50:
                        click_mouse()  
                        print(f"Clicked at screen center: {(center_x, center_y)} with confidence {conf}")

    cv2.imshow('YOLOv10 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

## ⚠️ Disclaimer

This tool is provided for educational purposes only. **Do not use this software to gain unfair advantages in games or violate terms of service.**

## 📫 Contact

Feel free to open an issue or pull request if you have suggestions or improvements!
