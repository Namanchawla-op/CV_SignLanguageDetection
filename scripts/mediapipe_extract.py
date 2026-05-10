import cv2
import os, sys, gc
import time
import numpy as np
import mediapipe as mp
from tqdm.auto import tqdm
from glob import glob
import math
import pickle

mp_holistic = mp.solutions.holistic

N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21


def process_body_landmarks(component, n_points):
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x, p.y, p.z] for p in landmarks])
        conf = np.array([p.visibility for p in landmarks])
    return kps, conf


def process_other_landmarks(component, n_points):
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x, p.y, p.z] for p in landmarks])
        conf = np.ones(n_points)
    return kps, conf


def get_holistic_keypoints(
    frames, holistic=mp_holistic.Holistic(static_image_mode=False, model_complexity=2)
):
    keypoints = []
    confs = []

    for frame in frames:
        results = holistic.process(frame)

        body_data, body_conf = process_body_landmarks(
            results.pose_landmarks, N_BODY_LANDMARKS
        )
        face_data, face_conf = process_other_landmarks(
            results.face_landmarks, N_FACE_LANDMARKS
        )
        lh_data, lh_conf = process_other_landmarks(
            results.left_hand_landmarks, N_HAND_LANDMARKS
        )
        rh_data, rh_conf = process_other_landmarks(
            results.right_hand_landmarks, N_HAND_LANDMARKS
        )

        data = np.concatenate([body_data, face_data, lh_data, rh_data])
        conf = np.concatenate([body_conf, face_conf, lh_conf, rh_conf])

        keypoints.append(data)
        confs.append(conf)

    holistic.close()
    del holistic
    gc.collect()

    keypoints = np.stack(keypoints)
    confs = np.stack(confs)
    return keypoints, confs


def gen_keypoints_for_frames(frames, save_path):
    pose_kps, pose_confs = get_holistic_keypoints(frames)
    body_kps = np.concatenate([pose_kps[:, :33, :], pose_kps[:, 501:, :]], axis=1)
    confs = np.concatenate([pose_confs[:, :33], pose_confs[:, 501:]], axis=1)

    d = {"keypoints": body_kps, "confidences": confs}

    with open(save_path + ".pkl", "wb") as f:
        pickle.dump(d, f, protocol=4)


def load_frames_from_video(video_path):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    while vidcap.isOpened():
        success, img = vidcap.read()
        if not success:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    vidcap.release()
    return np.asarray(frames)


def gen_keypoints_for_video(video_path, save_path):
    if not os.path.isfile(video_path):
        print("SKIPPING MISSING FILE:", video_path)
        return
    frames = load_frames_from_video(video_path)
    if len(frames) == 0:
        print("NO FRAMES FOUND IN:", video_path)
        return
    gen_keypoints_for_frames(frames, save_path)


if __name__ == "__main__":
    DIR = "AUTSL/train/"
    SAVE_DIR = "AUTSL/holistic_poses/"

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Scanning for videos in:", DIR)
    video_files = [f for f in os.listdir(DIR) if f.endswith(".mp4")]

    if len(video_files) == 0:
        print("No MP4 files found in AUTSL/train/")
        sys.exit()

    for file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(DIR, file)
        save_path = os.path.join(SAVE_DIR, file.replace(".mp4", ""))
        print("Processing:", video_path)
        gen_keypoints_for_video(video_path, save_path)

    print("DONE! Keypoints saved in:", SAVE_DIR)