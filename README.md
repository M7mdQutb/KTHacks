## **Sign Translator** is the new way to translate sign language to readable text

> made by Mohammed Kotb for KTHacks 2025 Hackathon

## How To Install
First You have to clone the repository using
```
git clone https://github.com/M7mdQutb/KTHacks.git
```
and install all python libraries with
```
python -m pip install -r requirements.txt
```

you have to create a config file called **config.ini** with:
|Config           |            Values        |
|-----------------|:------------------------:|
| HAND_MODEL_PATH | Path to hand model path  |
| POSE_MODEL_PATH | Path to pose model path  |
| DRAW_LANDMARKS  | Set it to 0 (no) or 1 (yes) |
| USE_DICT        | Set it to 0 (no) or 1 (yes) |
| LANG_DICT_PATH  | Path to The Dictionary (ex. dict.sdict)|
| USE_VIDEO       | Set to (0) or (1) to use video as input |
| VIDEO_PATH      | Path to video for processing |

There will be Models needed to run
| Model   |  Link |
|-------- |:--------:|
| Hand Landmarker | [download](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) |
| Pose Landmarker (lite) | [download](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task)|
| Pose Landmarker (full) | [download](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)|
| Pose Landmarker (heavy) | [download](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task) |
