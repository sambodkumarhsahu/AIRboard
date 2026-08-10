import cv2
import mediapipe as mp
import numpy as np
import math
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "models/hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.7,
)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

RED = (0, 0, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

current_color = RED
current_color_name = "RED"

PEN_THICKNESS = 4
ERASER_RADIUS = 70
SMOOTHING = 0.25

drawing_layer = None
drawing_mask = None

previous_point = None

smoothed_x = None
smoothed_y = None

shape_mode = "FREEHAND"
shape_start = None
shape_preview = None


def finger_extended(hand, tip, pip, mcp):
    wrist = hand[0]

    tip_point = hand[tip]
    pip_point = hand[pip]
    mcp_point = hand[mcp]

    tip_distance = math.hypot(
        tip_point.x - wrist.x,
        tip_point.y - wrist.y
    )

    pip_distance = math.hypot(
        pip_point.x - wrist.x,
        pip_point.y - wrist.y
    )

    long_enough = tip_distance > pip_distance * 1.15

    a = np.array([
        mcp_point.x,
        mcp_point.y
    ])

    b = np.array([
        pip_point.x,
        pip_point.y
    ])

    c = np.array([
        tip_point.x,
        tip_point.y
    ])

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )

    if denominator == 0:
        return False

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = math.degrees(
        math.acos(cosine)
    )

    straight = angle > 145

    return long_enough and straight


def thumb_extended(hand):
    wrist = hand[0]
    thumb_tip = hand[4]
    thumb_ip = hand[3]

    tip_distance = math.hypot(
        thumb_tip.x - wrist.x,
        thumb_tip.y - wrist.y
    )

    ip_distance = math.hypot(
        thumb_ip.x - wrist.x,
        thumb_ip.y - wrist.y
    )

    return tip_distance > ip_distance * 1.15


def is_horizontal_palm(hand):
    wrist = hand[0]
    middle_mcp = hand[9]

    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y

    angle = abs(
        math.degrees(
            math.atan2(dy, dx)
        )
    )

    angle = min(angle, 180 - angle)

    return angle < 40


def draw_hand(frame, hand):
    height, width, _ = frame.shape

    points = []

    for landmark in hand:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    for start, end in CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            GREEN,
            2,
            cv2.LINE_AA
        )

    for point in points:
        cv2.circle(
            frame,
            point,
            4,
            GREEN,
            -1
        )

    return points


def erase_area(center):
    global drawing_layer
    global drawing_mask

    cv2.circle(
        drawing_layer,
        center,
        ERASER_RADIUS,
        (0, 0, 0),
        -1
    )

    cv2.circle(
        drawing_mask,
        center,
        ERASER_RADIUS,
        0,
        -1
    )


def draw_shape(
    image,
    shape,
    start,
    end,
    color,
    thickness
):
    if start is None or end is None:
        return

    x1, y1 = start
    x2, y2 = end

    if shape == "LINE":

        cv2.line(
            image,
            start,
            end,
            color,
            thickness,
            cv2.LINE_AA
        )

    elif shape == "RECTANGLE":

        cv2.rectangle(
            image,
            start,
            end,
            color,
            thickness,
            cv2.LINE_AA
        )

    elif shape == "CIRCLE":

        dx = x2 - x1
        dy = y2 - y1

        radius = int(
            math.sqrt(
                dx * dx + dy * dy
            )
        )

        if radius > 2:
            cv2.circle(
                image,
                start,
                radius,
                color,
                thickness,
                cv2.LINE_AA
            )

    elif shape == "TRIANGLE":

        dx = x2 - x1
        dy = y2 - y1

        size = max(
            abs(dx),
            abs(dy)
        )

        if size > 2:

            top = (
                x1,
                y1 - size
            )

            bottom_left = (
                x1 - size,
                y1 + size
            )

            bottom_right = (
                x1 + size,
                y1 + size
            )

            triangle = np.array(
                [
                    top,
                    bottom_left,
                    bottom_right
                ],
                dtype=np.int32
            )

            cv2.polylines(
                image,
                [triangle],
                True,
                color,
                thickness,
                cv2.LINE_AA
            )


def commit_shape():
    global shape_start
    global shape_preview

    if (
        shape_mode != "FREEHAND"
        and shape_start is not None
        and shape_preview is not None
    ):

        distance = math.hypot(
            shape_preview[0] - shape_start[0],
            shape_preview[1] - shape_start[1]
        )

        if distance > 5:

            draw_shape(
                drawing_layer,
                shape_mode,
                shape_start,
                shape_preview,
                current_color,
                PEN_THICKNESS
            )

            draw_shape(
                drawing_mask,
                shape_mode,
                shape_start,
                shape_preview,
                255,
                PEN_THICKNESS
            )

    shape_start = None
    shape_preview = None


start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:

    cv2.namedWindow(
        "AIRBOARD",
        cv2.WINDOW_NORMAL
    )

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read webcam")
            break

        frame = cv2.flip(
            frame,
            1
        )

        height, width, _ = frame.shape

        if drawing_layer is None:

            drawing_layer = np.zeros_like(frame)

            drawing_mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        is_drawing = False
        is_erasing = False

        gesture_name = "NO HAND"

        active_tip = None

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            draw_hand(
                frame,
                hand
            )

            index_open = finger_extended(
                hand,
                tip=8,
                pip=6,
                mcp=5
            )

            middle_open = finger_extended(
                hand,
                tip=12,
                pip=10,
                mcp=9
            )

            ring_open = finger_extended(
                hand,
                tip=16,
                pip=14,
                mcp=13
            )

            pinky_open = finger_extended(
                hand,
                tip=20,
                pip=18,
                mcp=17
            )

            thumb_open = thumb_extended(
                hand
            )

            index_only = (
                index_open
                and not middle_open
                and not ring_open
                and not pinky_open
                and not thumb_open
            )

            middle_only = (
                middle_open
                and not index_open
                and not ring_open
                and not pinky_open
                and not thumb_open
            )

            pinky_only = (
                pinky_open
                and not index_open
                and not middle_open
                and not ring_open
                and not thumb_open
            )

            open_palm = (
                index_open
                and middle_open
                and ring_open
                and pinky_open
                and thumb_open
            )

            horizontal_palm = is_horizontal_palm(
                hand
            )

            if index_only:

                is_drawing = True

                current_color = RED
                current_color_name = "RED"

                gesture_name = "INDEX - RED"

                active_tip = 8

            elif middle_only:

                is_drawing = True

                current_color = BLUE
                current_color_name = "BLUE"

                gesture_name = "MIDDLE - BLUE"

                active_tip = 12

            elif pinky_only:

                is_drawing = True

                current_color = BLACK
                current_color_name = "BLACK"

                gesture_name = "PINKY - BLACK"

                active_tip = 20

            elif open_palm and horizontal_palm:

                is_erasing = True

                gesture_name = "SIDEWAYS PALM - ERASE"

                previous_point = None

                commit_shape()

                palm_x = int(
                    (
                        hand[0].x
                        + hand[5].x
                        + hand[9].x
                        + hand[13].x
                        + hand[17].x
                    )
                    / 5
                    * width
                )

                palm_y = int(
                    (
                        hand[0].y
                        + hand[5].y
                        + hand[9].y
                        + hand[13].y
                        + hand[17].y
                    )
                    / 5
                    * height
                )

                erase_point = (
                    palm_x,
                    palm_y
                )

                erase_area(
                    erase_point
                )

                cv2.circle(
                    frame,
                    erase_point,
                    ERASER_RADIUS,
                    WHITE,
                    3
                )

            elif open_palm:

                gesture_name = "OPEN PALM - STOP"

                previous_point = None

                commit_shape()

            else:

                gesture_name = "OTHER GESTURE"

                previous_point = None

                commit_shape()

            if active_tip is not None:

                raw_x = int(
                    hand[active_tip].x * width
                )

                raw_y = int(
                    hand[active_tip].y * height
                )

                if smoothed_x is None:

                    smoothed_x = float(raw_x)
                    smoothed_y = float(raw_y)

                else:

                    smoothed_x = (
                        SMOOTHING * raw_x
                        + (1 - SMOOTHING) * smoothed_x
                    )

                    smoothed_y = (
                        SMOOTHING * raw_y
                        + (1 - SMOOTHING) * smoothed_y
                    )

                smooth_point = (
                    int(smoothed_x),
                    int(smoothed_y)
                )

                if shape_mode == "FREEHAND":

                    if is_drawing:

                        if previous_point is not None:

                            movement = math.hypot(
                                smooth_point[0]
                                - previous_point[0],
                                smooth_point[1]
                                - previous_point[1]
                            )

                            if movement > 2:

                                cv2.line(
                                    drawing_layer,
                                    previous_point,
                                    smooth_point,
                                    current_color,
                                    PEN_THICKNESS,
                                    cv2.LINE_AA
                                )

                                cv2.line(
                                    drawing_mask,
                                    previous_point,
                                    smooth_point,
                                    255,
                                    PEN_THICKNESS,
                                    cv2.LINE_AA
                                )

                        previous_point = smooth_point

                else:

                    if is_drawing:

                        previous_point = None

                        if shape_start is None:

                            shape_start = smooth_point

                        shape_preview = smooth_point

                cv2.circle(
                    frame,
                    smooth_point,
                    9,
                    current_color,
                    -1
                )

                cv2.circle(
                    frame,
                    smooth_point,
                    12,
                    WHITE,
                    2
                )

            else:

                smoothed_x = None
                smoothed_y = None

        else:

            previous_point = None

            smoothed_x = None
            smoothed_y = None

            commit_shape()

        drawing_mask_bool = (
            drawing_mask > 0
        )

        frame[drawing_mask_bool] = (
            drawing_layer[drawing_mask_bool]
        )

        if (
            shape_mode != "FREEHAND"
            and shape_start is not None
            and shape_preview is not None
        ):

            draw_shape(
                frame,
                shape_mode,
                shape_start,
                shape_preview,
                current_color,
                PEN_THICKNESS
            )

        if is_drawing:

            status_color = current_color

        elif is_erasing:

            status_color = WHITE

        else:

            status_color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (10, 10),
            (530, 120),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            gesture_name,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "COLOR: " + current_color_name,
            (20, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            WHITE,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "MODE: " + shape_mode,
            (20, 91),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            WHITE,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "0 FREE | 1 LINE | 2 RECT | 3 CIRCLE | 4 TRIANGLE",
            (20, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            WHITE,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "INDEX = RED",
            (20, height - 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "MIDDLE = BLUE",
            (20, height - 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "PINKY = BLACK",
            (20, height - 81),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "SIDEWAYS PALM = ERASE",
            (20, height - 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "C = CLEAR     Q = QUIT",
            (20, height - 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "AIRBOARD",
            frame
        )

        key = cv2.waitKeyEx(1)

        if key != -1:
            key = key & 0xFF

        if key == ord("q"):

            break

        elif key == ord("0"):

            commit_shape()

            shape_mode = "FREEHAND"

            previous_point = None

        elif key == ord("1"):

            commit_shape()

            shape_mode = "LINE"

            previous_point = None

        elif key == ord("2"):

            commit_shape()

            shape_mode = "RECTANGLE"

            previous_point = None

        elif key == ord("3"):

            commit_shape()

            shape_mode = "CIRCLE"

            previous_point = None

        elif key == ord("4"):

            commit_shape()

            shape_mode = "TRIANGLE"

            previous_point = None

        elif key == ord("c") or key == ord("C"):

            drawing_layer = np.zeros_like(
                frame
            )

            drawing_mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )

            previous_point = None
            shape_start = None
            shape_preview = None


camera.release()
cv2.destroyAllWindows()