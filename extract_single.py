import cv2
import numpy as np
import mediapipe as mp
import pickle
import os

mp_holistic = mp.solutions.holistic

def extract_keypoints(video_path, save_path):
    if not os.path.isfile(video_path):
        print("❌ Video not found:", video_path)
        return

    print("▶ Loading video:", video_path)
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print("❌ No frames found in video.")
        return

    print("▶ Extracting keypoints...")

    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=2)
    keypoints_list = []
    conf_list = []

    for frame in frames:
        results = holistic.process(frame)

        # 33 pose, 21 LH, 21 RH, 468 face
        def get_landmarks(component, n_points):
            if component is None:
                return np.zeros((n_points, 3)), np.zeros(n_points)
            landmarks = component.landmark
            xyz = np.array([[p.x, p.y, p.z] for p in landmarks])
            conf = np.ones(n_points)
            return xyz, conf

        pose_kps, pose_conf = get_landmarks(results.pose_landmarks, 33)
        face_kps, face_conf = get_landmarks(results.face_landmarks, 468)
        lh_kps, lh_conf = get_landmarks(results.left_hand_landmarks, 21)
        rh_kps, rh_conf = get_landmarks(results.right_hand_landmarks, 21)

        # one single frame keypoint vector
        frame_kps = np.concatenate([pose_kps, face_kps, lh_kps, rh_kps])
        frame_conf = np.concatenate([pose_conf, face_conf, lh_conf, rh_conf])

        keypoints_list.append(frame_kps)
        conf_list.append(frame_conf)

    holistic.close()

    keypoints = np.array(keypoints_list)
    confs = np.array(conf_list)

    data = {"keypoints": keypoints, "confidences": confs}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path + ".pkl", "wb") as f:
        pickle.dump(data, f, protocol=4)

    print("✔ Saved output as:", save_path + ".pkl")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python extract_single.py <video_path> <output_path>")
        exit(1)

    video_path = sys.argv[1]
    save_path = sys.argv[2]
    extract_keypoints(video_path, save_path)
