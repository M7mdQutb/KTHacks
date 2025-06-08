import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import cv2
import numpy as np

import os
import time
import logging

import pyconf
import core

from tqdm import tqdm 

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

# Create Emply Lang Dict
land_dict = core.Lang_Dict()

pauseVideo = "other/black.mp4"
trainPath = "train/"

Vids = os.listdir(trainPath)

train_vids = [vid for vid in Vids if vid.endswith('.mp4')]
total_vids = len(Vids)
total_vids = []
for i in total_vids:
    if i.endswith('.mp4'):
        total_vids.append(f"{trainPath}{i}")
        total_vids.append(pauseVideo)


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

def get_conditions(allSigns: np.array) -> tuple:
    """
    Extracts conditions from hand and pose landmarks.
    Returns a numpy array of conditions.
    """
    if not allSigns.size > 0:
        raise ValueError("No signs detected. Please ensure the data is available.")
    
    try:
        if len(allSigns) != 1:
            condition = np.empty((0, 6), dtype=bool)
            for i in range(len(allSigns)):
                hand_landmarks_list = allSigns[i][0]
                pose_landmarks_list = allSigns[i][1]
                wrist_landmark = hand_landmarks_list[mp_hands.HandLandmark.WRIST.value]
                wristPos = np.array([wrist_landmark.x, wrist_landmark.y])

                thumb_cmc = hand_landmarks_list[mp_hands.HandLandmark.THUMB_CMC.value]
                thumb_mcp = hand_landmarks_list[mp_hands.HandLandmark.THUMB_MCP.value]
                thumb_ip = hand_landmarks_list[mp_hands.HandLandmark.THUMB_IP.value]
                thumb_tip = hand_landmarks_list[mp_hands.HandLandmark.THUMB_TIP.value]
                thumbPos = np.array([
                    [thumb_cmc.x, thumb_cmc.y],
                    [thumb_mcp.x, thumb_mcp.y],
                    [thumb_ip.x, thumb_ip.y],
                    [thumb_tip.x, thumb_tip.y]
                ])

                index_mcp = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_MCP.value]
                index_pip = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_PIP.value]
                index_dip = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_DIP.value]
                index_tip = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
                indexPos = np.array([
                    [index_mcp.x, index_mcp.y],
                    [index_pip.x, index_pip.y],
                    [index_dip.x, index_dip.y],
                    [index_tip.x, index_tip.y]
                ])

                middle_mcp = hand_landmarks_list[mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value]
                middle_pip = hand_landmarks_list[mp_hands.HandLandmark.MIDDLE_FINGER_PIP.value]
                middle_dip = hand_landmarks_list[mp_hands.HandLandmark.MIDDLE_FINGER_DIP.value]
                middle_tip = hand_landmarks_list[mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value]
                middlePos = np.array([
                    [middle_mcp.x, middle_mcp.y],
                    [middle_pip.x, middle_pip.y],
                    [middle_dip.x, middle_dip.y],
                    [middle_tip.x, middle_tip.y]
                ])

                ring_mcp = hand_landmarks_list[mp_hands.HandLandmark.RING_FINGER_MCP.value]
                ring_pip = hand_landmarks_list[mp_hands.HandLandmark.RING_FINGER_PIP.value]
                ring_dip = hand_landmarks_list[mp_hands.HandLandmark.RING_FINGER_DIP.value]
                ring_tip = hand_landmarks_list[mp_hands.HandLandmark.RING_FINGER_TIP.value]
                ringPos = np.array([
                    [ring_mcp.x, ring_mcp.y],
                    [ring_pip.x, ring_pip.y],
                    [ring_dip.x, ring_dip.y],
                    [ring_tip.x, ring_tip.y]
                ])

                pinky_mcp = hand_landmarks_list[mp_hands.HandLandmark.PINKY_MCP.value]
                pinky_pip = hand_landmarks_list[mp_hands.HandLandmark.PINKY_PIP.value]
                pinky_dip = hand_landmarks_list[mp_hands.HandLandmark.PINKY_DIP.value]
                pinky_tip = hand_landmarks_list[mp_hands.HandLandmark.PINKY_TIP.value]
                pinkyPos = np.array([
                    [pinky_mcp.x, pinky_mcp.y],
                    [pinky_pip.x, pinky_pip.y],
                    [pinky_dip.x, pinky_dip.y],
                    [pinky_tip.x, pinky_tip.y]
                ])
                chest_right = pose_landmarks_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                chest_left = pose_landmarks_list[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                chest_bottom_left = pose_landmarks_list[mp_pose.PoseLandmark.LEFT_HIP.value]
                chest_bottom_right = pose_landmarks_list[mp_pose.PoseLandmark.RIGHT_HIP.value]
                chest_pos = np.array([
                    [chest_right.x, chest_right.y],
                    [chest_left.x, chest_left.y],
                    [chest_bottom_right.x, chest_bottom_right.y],
                    [chest_bottom_left.x, chest_bottom_left.y]
                ])

                midpointOfOppositePoints = chest_pos[0] + chest_pos[2] / 2 
                distanceFromWristToMidPoint = np.linalg.norm(wristPos - midpointOfOppositePoints)
                distanceFromShoulderToMidPoint = np.linalg.norm(chest_pos[0] - midpointOfOppositePoints)

                temp_condition = np.array([
                            thumbPos[-1][1] > thumbPos[0][1],
                            indexPos[-1][1] > indexPos[0][1],
                            middlePos[-1][1] > middlePos[0][1],
                            ringPos[-1][1] > ringPos[0][1],
                            pinkyPos[-1][1] > pinkyPos[0][1],
                            distanceFromWristToMidPoint < distanceFromShoulderToMidPoint
                        ])
                                
                condition = np.append(condition, [temp_condition], axis=0)
                condition = tuple(map(tuple, condition))
            return condition

        # One Position , No Movement
        else:
            wrist_landmark = hand_landmarks_list[mp_hands.HandLandmark.WRIST.value]
            wristPos = np.array([wrist_landmark.x, wrist_landmark.y])

            thumb_cmc = hand_landmarks_list[mp_hands.HandLandmark.THUMB_CMC.value]
            thumb_mcp = hand_landmarks_list[mp_hands.HandLandmark.THUMB_MCP.value]
            thumb_ip = hand_landmarks_list[mp_hands.HandLandmark.THUMB_IP.value]
            thumb_tip = hand_landmarks_list[mp_hands.HandLandmark.THUMB_TIP.value]
            thumbPos = np.array([
                [thumb_cmc.x, thumb_cmc.y],
                [thumb_mcp.x, thumb_mcp.y],
                [thumb_ip.x, thumb_ip.y],
                [thumb_tip.x, thumb_tip.y]
            ])

            index_mcp = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_MCP.value]
            index_pip = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_PIP.value]
            index_dip = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_DIP.value]
            index_tip = hand_landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
            indexPos = np.array([
                [index_mcp.x, index_mcp.y],
                [index_pip.x, index_pip.y],
                [index_dip.x, index_dip.y],
                [index_tip.x, index_tip.y]
            ])

            middle_mcp = hand_landmarks_list[mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value]
            middle_pip = hand_landmarks_list[mp_hands.HandLandmark.MIDDLE_FINGER_PIP.value]
            middle_dip = hand_landmarks_list[mp_hands.HandLandmark.MIDDLE_FINGER_DIP.value]
            middle_tip = hand_landmarks_list[mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value]
            middlePos = np.array([
                [middle_mcp.x, middle_mcp.y],
                [middle_pip.x, middle_pip.y],
                [middle_dip.x, middle_dip.y],
                [middle_tip.x, middle_tip.y]
            ])

            ring_mcp = hand_landmarks_list[mp_hands.HandLandmark.RING_FINGER_MCP.value]
            ring_pip = hand_landmarks_list[mp_hands.HandLandmark.RING_FINGER_PIP.value]
            ring_dip = hand_landmarks_list[mp_hands.HandLandmark.RING_FINGER_DIP.value]
            ring_tip = hand_landmarks_list[mp_hands.HandLandmark.RING_FINGER_TIP.value]
            ringPos = np.array([
                [ring_mcp.x, ring_mcp.y],
                [ring_pip.x, ring_pip.y],
                [ring_dip.x, ring_dip.y],
                [ring_tip.x, ring_tip.y]
            ])

            pinky_mcp = hand_landmarks_list[mp_hands.HandLandmark.PINKY_MCP.value]
            pinky_pip = hand_landmarks_list[mp_hands.HandLandmark.PINKY_PIP.value]
            pinky_dip = hand_landmarks_list[mp_hands.HandLandmark.PINKY_DIP.value]
            pinky_tip = hand_landmarks_list[mp_hands.HandLandmark.PINKY_TIP.value]
            pinkyPos = np.array([
                [pinky_mcp.x, pinky_mcp.y],
                [pinky_pip.x, pinky_pip.y],
                [pinky_dip.x, pinky_dip.y],
                [pinky_tip.x, pinky_tip.y]
            ])

            chest_right = pose_landmarks_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            chest_left = pose_landmarks_list[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            chest_bottom_left = pose_landmarks_list[mp_pose.PoseLandmark.LEFT_HIP.value]
            chest_bottom_right = pose_landmarks_list[mp_pose.PoseLandmark.RIGHT_HIP.value]
            midpointOfOppositePoints = np.array([
                (chest_right.x + chest_bottom_left.x) / 2,
                (chest_right.y + chest_bottom_left.y) / 2
            ])
            condition = np.array([
                            thumbPos[-1][1] > thumbPos[0][1],
                            indexPos[-1][1] > indexPos[0][1],
                            middlePos[-1][1] > middlePos[0][1],
                            ringPos[-1][1] > ringPos[0][1],
                            pinkyPos[-1][1] > pinkyPos[0][1],
                            np.linalg.norm(wristPos - midpointOfOppositePoints) > np.linalg.norm(chest_bottom_right - midpointOfOppositePoints)
                        ])

            return tuple(condition)
        
        print("How did you get here? This should not happen")
    except Exception as e:
        raise e

if __name__ == "__main__":
    cap = cv2.VideoCapture(pauseVideo)
    try:   
        with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
            with PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
                for v, vid in tqdm(enumerate(train_vids),total=len(train_vids)):
                    print(f"{trainPath}{vid}")
                    cap = cv2.VideoCapture(f"{trainPath}{vid}")
                    if not cap.isOpened():
                        logging.error("Could not open video.")
                        exit()
                    
                    allSigns = np.empty((0, 2), dtype=object)
                    innerSigns = np.empty((0, 2), dtype=object)

                    t = time.time()

                    while cap.isOpened():
                        success, frame = cap.read()
                        if not success:
                            logging.error(f"Error reading the frame in {cap}")
                            break

                        frame = cv2.flip(frame, 1)

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                        frame_copy = frame.copy()

                        hand_detection_result = hand_landmarker.detect_for_video(mp_image, int(time.time() * 1000))
                        pose_detection_result = pose_landmarker.detect_for_video(mp_image, int(time.time() * 1000))

                        if hand_detection_result.hand_landmarks and pose_detection_result.pose_landmarks:
                            for i, hand_landmarks_list in enumerate(hand_detection_result.hand_landmarks):
                                for q, pose_landmark_list in enumerate(pose_detection_result.pose_landmarks):
                                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                                    hand_landmarks_proto.landmark.extend([
                                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks_list
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
                                        mp_drawing_styles.get_default_hand_connections_style())
                                    
                                    mp_drawing.draw_landmarks(
                                        frame_copy,
                                        pose_landmarks_proto,
                                        mp_pose.POSE_CONNECTIONS,
                                        mp_drawing_styles.get_default_pose_landmarks_style())

                                    try:
                                        if time.time() - t < 2 and time.time() - t >= 0.5:
                                            pos_row = np.array([[hand_landmarks_list, pose_landmark_list]], dtype=object)
                                            innerSigns = np.append(innerSigns, pos_row, axis=0)
                                            t = time.time()
                                        elif time.time() - t >= 2:
                                            pos_row = np.array([[hand_landmarks_list, pose_landmark_list]], dtype=object)
                                            innerSigns = np.append(innerSigns, pos_row, axis=0)
                                            if allSigns.size == 0:
                                                allSigns = innerSigns.copy()
                                            else:
                                                allSigns = np.append(allSigns, innerSigns, axis=0)
                                            type(innerSigns)
                                            
                                            allSigns = np.append(allSigns, innerSigns, axis=0)
                                            innerSigns = np.empty((0, 2), dtype=object)
                                            t = time.time()
                                    except Exception as e:
                                        raise e
                                    

                        else:
                            if innerSigns.size > 0:
                                if allSigns.size == 0:
                                    allSigns = innerSigns.copy()
                                else:
                                    allSigns = np.append(allSigns, innerSigns, axis=0)
                            innerSigns = np.empty((0, 2), dtype=object)
                    lang_dict = core.add_to_dict(vid.split(".")[0],get_conditions(allSigns),lang_dict=land_dict)
                    lang_dict.saveToFile(conf["LANG_DICT_PATH"])
                    allSigns = np.empty((0, 2), dtype=object)

    except Exception as e:
        raise e
        exit()
    finally:
        cap.release()
        cv2.destroyAllWindows()

