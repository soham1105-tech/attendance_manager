import numpy as np
import cv2
import os
import shutil
import time
import logging
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import threading
import urllib.request

# ==============================
# 1️⃣ SESSION STATE INITIALIZATION (top-level)
# ==============================
for key, default in {
    'existing_faces_cnt': 0,
    'current_frame_faces_cnt': 0,
    'ss_cnt': 0,
    'current_face_dir': "",
    'face_folder_created_flag': False,
    'out_of_range_flag': False,
    'log_message': "Welcome! Grant webcam permissions to start.",
    'last_frame': None,
    'last_roi_info': {},
    'fps_show': 0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

lock = threading.Lock()
path_photos_from_camera = "data/data_faces_from_camera/"

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==============================
# 2️⃣ HELPER FUNCTIONS
# ==============================
def pre_work_mkdir():
    if not os.path.isdir("data"):
        os.mkdir("data")
    if not os.path.isdir(path_photos_from_camera):
        os.mkdir(path_photos_from_camera)

def check_existing_faces_cnt():
    if os.listdir(path_photos_from_camera):
        person_list = os.listdir(path_photos_from_camera)
        person_num_list = []
        for person in person_list:
            if person.startswith("person_"):
                try:
                    person_order = person.split('_')[1]
                    person_num_list.append(int(person_order))
                except (IndexError, ValueError):
                    continue
        if person_num_list:
            return max(person_num_list)
    return 0

def GUI_clear_data():
    folders_rd = os.listdir(path_photos_from_camera)
    for folder in folders_rd:
        shutil.rmtree(os.path.join(path_photos_from_camera, folder))
    if os.path.isfile("data/features_all.csv"):
        os.remove("data/features_all.csv")

    with lock:
        st.session_state.existing_faces_cnt = 0
        st.session_state.log_message = "Face images and features_all.csv removed!"
        st.session_state.face_folder_created_flag = False
        st.session_state.current_face_dir = ""

def create_face_folder(name_input):
    if not name_input:
        with lock:
            st.session_state.log_message = "Warning: Please input a name."
        return

    with lock:
        st.session_state.existing_faces_cnt += 1
        cnt = st.session_state.existing_faces_cnt
        current_face_dir = f"{path_photos_from_camera}person_{cnt}_{name_input}"
        os.makedirs(current_face_dir, exist_ok=True)
        st.session_state.current_face_dir = current_face_dir
        st.session_state.ss_cnt = 0
        st.session_state.face_folder_created_flag = True
        st.session_state.log_message = f"Folder created: {current_face_dir}"
        logging.info(f"Create folders: {current_face_dir}")

def save_current_face():
    with lock:
        face_folder_created = st.session_state.face_folder_created_flag
        faces_cnt = st.session_state.current_frame_faces_cnt
        out_of_range = st.session_state.out_of_range_flag
        frame = st.session_state.last_frame
        roi = st.session_state.last_roi_info
        current_dir = st.session_state.current_face_dir
        ss_cnt = st.session_state.ss_cnt

    st.session_state.log_message = f"Debug: faces_cnt={faces_cnt}, face_folder_created={face_folder_created}, out_of_range={out_of_range}"

    if not face_folder_created:
        st.session_state.log_message = "Error: Please run step 2 first!"
        return
    if faces_cnt != 1:
        st.session_state.log_message = "Error: Requires exactly one face in the frame!"
        return
    if out_of_range:
        st.session_state.log_message = "Error: Face is out of capture range!"
        return
    if frame is None or not roi:
        st.session_state.log_message = "Error: No frame data available."
        return

    # Rest of the saving code...


# ==============================
# 3️⃣ VIDEO PROCESSOR CLASS
# ==============================
class FaceRegisterProcessor(VideoProcessorBase):
    def __init__(self):
        # --- Setup OpenCV DNN Face Detector ---
        self.modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
        self.configFile = "deploy.prototxt"
        modelURL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        configURL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"

        if not os.path.exists(self.modelFile):
            print(f"Downloading {self.modelFile}...")
            urllib.request.urlretrieve(modelURL, self.modelFile)
        if not os.path.exists(self.configFile):
            print(f"Downloading {self.configFile}...")
            urllib.request.urlretrieve(configURL, self.configFile)

        self.face_net = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)
        self.confidence_threshold = 0.5

        self.frame_start_time = time.time()
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        frame_time = now - self.frame_start_time
        self.fps = 1.0 / frame_time if frame_time > 0 else 0
        self.frame_start_time = now
        return round(self.fps_show, 2)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        (h, w) = img_bgr.shape[:2]

        # --- Face Detection ---
        blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))
                cv2.rectangle(img_bgr, (startX, startY), (endX, endY), (255, 0, 0), 2)

        # FPS
        current_fps = self.update_fps()

        roi_info = {}
        out_of_range = False
        if len(faces) > 0:
            startX, startY, endX, endY = faces[0]
            face_h = endY - startY
            face_w = endX - startX
            hh = face_h // 2
            ww = face_w // 2

            if (endX + ww) > w or (endY + hh) > h or (startX - ww < 0) or (startY - hh < 0):
                out_of_range = True

            roi_info = {"h_start": startY, "w_start": startX,
                        "h": face_h, "w": face_w, "hh": hh, "ww": ww}

        with lock:
            st.session_state.current_frame_faces_cnt = len(faces)
            st.session_state.out_of_range_flag = out_of_range
            st.session_state.fps_show = current_fps
            st.session_state.last_frame = img_bgr
            st.session_state.last_roi_info = roi_info

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# ==============================
# 4️⃣ MAIN APP
# ==============================
def main():
    pre_work_mkdir()
    st.set_page_config(page_title="Face Register", layout="wide")
    st.title("Face Register")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Webcam Feed")
        webrtc_ctx = webrtc_streamer(
            key="face-register",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=FaceRegisterProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        st.text(f"FPS: {st.session_state.fps_show}")
        st.text(f"Faces detected: {st.session_state.current_frame_faces_cnt}")

    with col2:
     st.header("Registration Panel")
    name_input = st.text_input("Step 1: Input name")
    if st.button("Step 2: Create folder"):
        create_face_folder(name_input)
    st.write(f"Step 3: Capture faces (only one face per frame!)")
    st.write(f"Faces detected now: {st.session_state.current_frame_faces_cnt}")
    if st.button("Capture"):
        save_current_face()
    st.write("Step 4: Clear all data")
    if st.button("Clear All"):
        GUI_clear_data()
    st.text_area("Log", value=st.session_state.log_message, height=300)


if __name__ == "__main__":
    main()
