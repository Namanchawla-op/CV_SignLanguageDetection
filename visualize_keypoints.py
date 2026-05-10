import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load PKL
with open("output/unknown_keypoints.pkl", "rb") as f:
    d = pickle.load(f)

kps = d["keypoints"][0]  # first frame

# Only X and Y (ignore Z)
x = kps[:,0]
y = -kps[:,1]  # invert for better view

plt.scatter(x, y, s=10)
plt.title("Extracted Mediapipe Keypoints (Frame 1)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
