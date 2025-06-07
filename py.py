import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import cv2
import numpy as np

import os
import time

import pyconf
from core import *

# Mediapipe Shit
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = vision.RunningMode
mp_drawing = mp.solutions.drawing_utils

# Hand landmark
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult

# Pose landmark
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

conf = pyconf.read_ini("config.ini")

hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=conf["HAND_MODEL_PATH"]),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.4)

pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=conf["POSE_MODEL_PATH"]),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.4)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    try:
        with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
            with PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
                while cap.isOpened():
                    # Read frames and see if it works 
                    success, frame = cap.read()
                    if not success:
                        print(f"Error reading the frame in {cap}")
                        break

                    # Flip the frame
                    frame = cv2.flip(frame, 1)

                    # Convert the frame to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                    # Get copy of the frame to draw on
                    frame_copy = frame.copy()

                    hand_landmarker_result = hand_landmarker.detect_for_video(mp_image, int(time.time() * 1000))
                    pose_landmarker_result = pose_landmarker.detect_for_video(mp_image, int(time.time() * 1000))

                    if hand_landmarker_result.hand_landmarks and pose_landmarker_result.pose_landmarks:
                        for i, hand_landmark_list in enumerate(hand_landmarker_result.hand_landmarks):
                            for q, pose_landmark_list in enumerate(pose_landmarker_result.pose_landmarks):
                                if conf["draw_landmarks"] == "1":
                                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                                    hand_landmarks_proto.landmark.extend([
                                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmark_list
                                    ])
                                    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                                    pose_landmarks_proto.landmark.extend([
                                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmark_list
                                    ])

                                    mp_drawing.draw_landmarks(
                                        frame_copy,
                                        hand_landmarks_proto,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style()
                                    )
                                    
                                    mp_drawing.draw_landmarks(
                                        frame_copy,
                                        pose_landmarks_proto,
                                        mp_pose.POSE_CONNECTIONS,
                                        mp_drawing_styles.get_default_pose_landmarks_style()
                                    )
                                

                    cv2.imshow("Test1",cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

    except Exception as e:
        raise e
#        print(f"Error in 1st try: {e}")
        exit()
    finally:
        cap.release()
        cv2.destroyAllWindows()