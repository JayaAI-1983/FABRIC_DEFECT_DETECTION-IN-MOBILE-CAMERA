import os
import glob
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="AI Industrial QC Inspector", layout="wide")

# Force Full-Screen Camera Styling for Mobile Viewport
st.markdown(
    """
    <style>
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    div[data-testid="stWebRtcStreamer"] {
        width: 100% !important;
    }
    div[data-testid="stWebRtcStreamer"] video {
        width: 100% !important;
        height: 60vh !important;
        object-fit: cover !important;
        border-radius: 12px;
        border: 2px solid #000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏭 Real-Time Fabric & Bag Defect Inspector")

# GitHub / Cloud Deployment Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_FOLDER = os.path.join(BASE_DIR, "MASTER")

if not os.path.exists(MASTER_FOLDER):
    os.makedirs(MASTER_FOLDER, exist_ok=True)

@st.cache_resource
def load_master_database(folder_path):
    master_dict = {}
    if os.path.exists(folder_path):
        all_files = glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]")) + \
                    glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]")) + \
                    glob.glob(os.path.join(folder_path, "*.[jJ][pP][eE][gG]"))
        
        for fpath in all_files:
            img = cv2.imread(fpath)
            if img is not None:
                h, w = img.shape[:2]
                target_w = 640
                target_h = int(h * (target_w / w))
                resized_m = cv2.resize(img, (target_w, target_h))
                master_dict[os.path.basename(fpath)] = resized_m
    return master_dict

master_dict = load_master_database(MASTER_FOLDER)

st.sidebar.header("⚙️ QC Sensitivity Settings")
defect_sensitivity = st.sidebar.slider("Color Defect Sensitivity", 10, 100, 45)
min_defect_area = st.sidebar.slider("Min Defect Size (Pixels)", 30, 1500, 100)

def match_master_style(test_bgr, master_db):
    if not master_db:
        return None, 0
    sift = cv2.SIFT_create(nfeatures=500)
    gray_test = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    kp_test, des_test = sift.detectAndCompute(gray_test, None)

    if des_test is None or len(kp_test) < 10:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    best_score = 0
    best_master = None

    for name, m_img in master_db.items():
        gray_m = cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY)
        kp_m, des_m = sift.detectAndCompute(gray_m, None)

        if des_m is not None:
            matches = bf.match(des_m, des_test)
            score = len(matches)
            if score > best_score:
                best_score = score
                best_master = m_img

    return best_master, best_score

def inspect_defects(master_bgr, test_bgr, sensitivity, min_area):
    h_m, w_m = master_bgr.shape[:2]
    test_resized = cv2.resize(test_bgr, (w_m, h_m))

    master_lab = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    test_lab = cv2.cvtColor(test_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

    da = master_lab[:, :, 1] - test_lab[:, :, 1]
    db = master_lab[:, :, 2] - test_lab[:, :, 2]
    color_diff = np.sqrt(da**2 + db**2)
    color_diff = np.clip(color_diff, 0, 255).astype(np.uint8)

    blurred_diff = cv2.GaussianBlur(color_diff, (7, 7), 0)
    _, thresh = cv2.threshold(blurred_diff, sensitivity, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_img = test_resized.copy()
    defect_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= (w_m * h_m * 0.3):
            x, y, bw, bh = cv2.boundingRect(contour)
            # Drawing Bounding Box around Defect
            cv2.rectangle(output_img, (x, y), (x + bw, y + bh), (0, 0, 255), 4)
            cv2.putText(output_img, "DEFECT", (x, max(y - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            defect_count += 1

    return output_img, defect_count

class LiveQCProcessor(VideoProcessorBase):
    def __init__(self):
        self.sensitivity = 45
        self.min_area = 100
        self.master_db = master_dict

    def update_params(self, sensitivity, min_area, db):
        self.sensitivity = sensitivity
        self.min_area = min_area
        self.master_db = db

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")

        if not self.master_db:
            # Display Header Banner on Video Frame
            cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(img_bgr, "ERROR: MASTER FOLDER IS EMPTY IN GITHUB", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame.from_ndarray(img_bgr, format="bgr24")

        matched_master, match_score = match_master_style(img_bgr, self.master_db)

        # Background overlay for status text
        cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 60), (0, 0, 0), -1)

        if matched_master is None or match_score < 10:
            cv2.putText(img_bgr, "STATUS: SEARCHING / UNKNOWN STYLE", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            return frame.from_ndarray(img_bgr, format="bgr24")

        processed_img, defects = inspect_defects(matched_master, img_bgr, self.sensitivity, self.min_area)

        cv2.rectangle(processed_img, (0, 0), (processed_img.shape[1], 60), (0, 0, 0), -1)

        if defects == 0:
            cv2.putText(processed_img, "STATUS: PASSED (NO DEFECT)", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(processed_img, f"STATUS: REJECTED ({defects} DEFECTS FOUND)", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame.from_ndarray(processed_img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
            {
                "urls": "turn:openrelay.metered.ca:80",
                "username": "openrelay",
                "credential": "openrelay",
            }
        ]
    }
)

ctx = webrtc_streamer(
    key="industrial-qc-live",
    video_processor_factory=LiveQCProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "facingMode": "environment",
            "width": {"ideal": 1280},
            "height": {"ideal": 720}
        },
        "audio": False
    },
    async_processing=True
)

if ctx.video_processor:
    ctx.video_processor.update_params(defect_sensitivity, min_defect_area, master_dict)
