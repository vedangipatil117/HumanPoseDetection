import streamlit as st
from PIL import Image
import numpy as np
import cv2

DEMO_IMG='stand.jpg'

body_parts = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

pose_pairs = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"],
    ["LShoulder", "LElbow"], ["LElbow", "LWrist"], ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
    ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

width=368
height=368
inWidth=width
inHeight=height


net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

st.title("Human Pose Estimation OpenCV")

st.text("Make sure you have a clear image with all parts clearly visible")

img_file_buffer=st.file_uploader("Upload an image, Make sure you have clear image", type=["jpg","jpeg","png"])

if img_file_buffer is not None:
    image=np.array(Image.open(img_file_buffer))

else:
    demo_image=DEMO_IMG
    image = np.array(Image.open(DEMO_IMG))

st.subheader("Original Image")
st.image(
    image,caption=f'Original Image',use_column_width=True
)

thres = st.slider("Threshold for detecting the key points",min_value=0,value=20,max_value=100,step=5)

thres=thres/100

@st.cache_data
def poseDetector(frame):
    frameWidth=frame.shape[1]
    frameHeight=frame.shape[0]

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))


    out=net.forward()
    out=out[:, :19, :, :]

    assert len(body_parts) == out.shape[1]

    points = []
    for i in range(len(body_parts)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)

    for pair in pose_pairs:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in body_parts)
        assert(partTo in body_parts)
        idFrom = body_parts[partFrom]
        idTo = body_parts[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    t,_=net.getPerfProfile()
    return frame

output = poseDetector(image)

st.subheader("Positions Estimated")
st.image(
    output, caption=f"Position estimated",use_column_width=True)
st.markdown('''
#
''')

