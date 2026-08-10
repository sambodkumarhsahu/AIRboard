# AIRboard

AIRboard is a computer-vision based air-writing and drawing application that lets you draw on a live webcam feed using hand gestures.

It uses MediaPipe Hand Landmarker to track hand landmarks in real time and OpenCV to render the drawing, hand skeleton, gestures, colors, shapes, and eraser directly on the webcam view.

## Tech Stack

- Python
- OpenCV
- MediaPipe Hand Landmarker
- NumPy
- Math / Time modules
- Webcam-based real-time computer vision

## Features

- Real-time hand tracking
- Air drawing using finger gestures
- Hand skeleton visualization
- Three drawing colors:
  - Index finger → Red
  - Middle finger → Blue
  - Pinky finger → Black
- Freehand drawing
- Line tool
- Rectangle tool
- Circle tool
- Triangle tool
- Gesture-based eraser
- Drawing smoothing for more stable strokes
- Live gesture and mode indicators
- Webcam mirror view
- Clear canvas with a key press
- Keyboard controls for switching drawing modes

## Gesture Controls

| Gesture | Action |
|---|---|
| Index finger only | Draw in red |
| Middle finger only | Draw in blue |
| Pinky finger only | Draw in black |
| Open palm | Stop drawing |
| Sideways open palm | Erase |
| No hand / other gesture | Stop drawing |

## Keyboard Controls

| Key | Action |
|---|---|
| `0` | Freehand mode |
| `1` | Line mode |
| `2` | Rectangle mode |
| `3` | Circle mode |
| `4` | Triangle mode |
| `C` | Clear drawing |
| `Q` | Quit |

## Project Structure

```text
AIRboard/
├── app.py
├── models/
│   └── hand_landmarker.task
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/sambodkumarhsahu/AIRboard.git
cd AIRboard
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install opencv-python mediapipe numpy
```

## Run

Make sure your webcam is connected, then run:

```bash
python app.py
```

The AIRBOARD window should open and begin tracking your hand.

## Model

The project uses the MediaPipe Hand Landmarker model located at:

```text
models/hand_landmarker.task
```

The model file is included in this repository so the application can locate it using the path defined in `app.py`.

## Demo

[Watch the AIRboard demo on Instagram](https://www.instagram.com/reel/Db0Zv7WO5DI/)

## How It Works

AIRboard captures frames from the webcam and passes them to MediaPipe's Hand Landmarker. The detected hand landmarks are analyzed to determine which fingers are extended.

Based on the detected gesture, AIRboard selects a drawing color, activates drawing or erasing, and uses the corresponding fingertip as the drawing position. OpenCV then renders the strokes and shapes on top of the live camera feed.

Position smoothing is applied to reduce jitter and make the air-drawn strokes more stable.

## Future Improvements

- Save drawings as images
- Undo / redo
- More colors
- Adjustable pen thickness
- More shapes
- Gesture-based UI controls
- Multi-hand support
- Better gesture recognition
- Improved drawing stabilization
- Optional fullscreen presentation mode
- Browser-based version

## Author

**Sambodh Kumar Sahu**

Built as a computer vision project using Python, OpenCV, and MediaPipe.
