import os
import glob
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="AI Industrial QC Inspector", layout="centered")

st.title("🏭 Real-Time Fabric & Bag Defect Inspector")
st.caption("SIFT Alignment + Multi-Defect Classifier (Stain, Density, Color Mismatch)")

MASTER_FOLDER = r"D:\PRABHU_PROJECT\DATASET\MASTER"

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

if not master_dict:
    st.error(f"❌ Master images not found in `{MASTER_FOLDER}`. Please add approved `.jpg` files.")
    st.stop()

st.sidebar.header("⚙️ QC Inspection Settings")
defect_sensitivity = st.sidebar.slider("Color Defect Sensitivity", 10, 100, 55)
min_defect_area = st.sidebar.slider("Min Defect Size (Pixels)", 30, 1500, 150)

def match_master_style(test_bgr, master_dict):
    sift = cv2.SIFT_create(nfeatures=500)
    gray_test = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    kp_test, des_test = sift.detectAndCompute(gray_test, None)

    if des_test is None or len(kp_test) < 10:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    best_score = 0
    best_master = None

    for name, m_img in master_dict.items():
        gray_m = cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY)
        kp_m, des_m = sift.detectAndCompute(gray_m, None)

        if des_m is not None:
            matches = bf.match(des_m, des_test)
            score = len(matches)
            if score > best_score:
                best_score = score
                best_master = m_img

    return best_master, best_score

def classify_defect_type(master_roi, test_roi):
    """
    Adaptive Lighting Classifier - Dark spots vs Shadow Filter
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    m_gray = cv2.cvtColor(master_roi, cv2.COLOR_BGR2GRAY)
    t_gray = cv2.cvtColor(test_roi, cv2.COLOR_BGR2GRAY)

    m_norm = clahe.apply(m_gray)
    t_norm = clahe.apply(t_gray)

    # 1. LAB Lightness (L-Channel) Analysis
    m_lab = cv2.cvtColor(master_roi, cv2.COLOR_BGR2LAB)
    t_lab = cv2.cvtColor(test_roi, cv2.COLOR_BGR2LAB)

    m_lightness = m_lab[:, :, 0].astype(float)
    t_lightness = t_lab[:, :, 0].astype(float)

    lightness_diff = np.mean(m_lightness) - np.mean(t_lightness)
    overall_frame_darkness = np.mean(t_gray)

    # Filter out normal low brightness/shadows
    if lightness_diff > 35 and overall_frame_darkness > 40:
        return "STAIN / OIL MARK"

    # 2. Thread Color Mismatch
    color_dist = np.mean(np.abs(m_lab[:, :, 1:].astype(float) - t_lab[:, :, 1:].astype(float)))
    if color_dist > 30:
        return "THREAD COLOR MISMATCH"

    # 3. Stitch / Embroidery Density Check
    m_edges = cv2.Canny(m_norm, 50, 150)
    t_edges = cv2.Canny(t_norm, 50, 150)
    
    m_density = np.sum(m_edges > 0)
    t_density = np.sum(t_edges > 0)

    if t_density < (m_density * 0.45):
        return "DENSITY TOO LOW"
    elif t_density > (m_density * 1.9):
        return "DENSITY TOO HIGH"

    return "PATTERN DEFECT"

def inspect_defects(master_bgr, test_bgr, sensitivity, min_area):
    h_m, w_m = master_bgr.shape[:2]
    test_resized = cv2.resize(test_bgr, (w_m, h_m))

    gray_master = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_resized, cv2.COLOR_BGR2GRAY)

    master_lab = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    test_lab = cv2.cvtColor(test_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

    da = master_lab[:, :, 1] - test_lab[:, :, 1]
    db = master_lab[:, :, 2] - test_lab[:, :, 2]
    color_diff = np.sqrt(da**2 + db**2)
    color_diff = np.clip(color_diff, 0, 255).astype(np.uint8)

    edges = cv2.Canny(gray_master, 60, 180)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge_mask = cv2.dilate(edges, kernel_edge)
    color_diff[edge_mask > 0] = 0

    blurred_diff = cv2.GaussianBlur(color_diff, (7, 7), 0)
    _, thresh = cv2.threshold(blurred_diff, sensitivity, 255, cv2.THRESH_BINARY)

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel_clean)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_img = test_resized.copy()
    defect_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= (w_m * h_m * 0.25):
            x, y, bw, bh = cv2.boundingRect(contour)
            if x <= 10 or y <= 10 or (x + bw) >= (w_m - 10) or (y + bh) >= (h_m - 10):
                continue

            master_roi = master_bgr[y:y+bh, x:x+bw]
            test_roi = test_resized[y:y+bh, x:x+bw]

            defect_type = classify_defect_type(master_roi, test_roi)

            cv2.rectangle(output_img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
            cv2.putText(output_img, f"DEFECT: {defect_type}", (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            defect_count += 1

    return output_img, defect_count

class LiveQCProcessor(VideoProcessorBase):
    def __init__(self):
        self.sensitivity = 55
        self.min_area = 150

    def update_params(self, sensitivity, min_area):
        self.sensitivity = sensitivity
        self.min_area = min_area

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")

        matched_master, match_score = match_master_style(img_bgr, master_dict)

        if matched_master is None or match_score < 18:
            cv2.putText(img_bgr, "STATUS: REJECTED (Unknown / Unregistered Style)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame.from_ndarray(img_bgr, format="bgr24")

        processed_img, defects = inspect_defects(matched_master, img_bgr, self.sensitivity, self.min_area)

        if defects == 0:
            cv2.putText(processed_img, "STATUS: PASSED (NO DEFECT)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(processed_img, f"STATUS: REJECTED ({defects} DEFECTS FOUND)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame.from_ndarray(processed_img, format="bgr24")

# Streamlit WebRTC Live Streamer with Mobile Rear Camera Support
ctx = webrtc_streamer(
    key="industrial-qc-live",
    video_processor_factory=LiveQCProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={
        "video": {
            "facingMode": "environment"  # Mobile-la direct-a Back Camera open aagum
        },
        "audio": False
    },
    async_processing=True
)

if ctx.video_processor:
    ctx.video_processor.update_params(defect_sensitivity, min_defect_area)

# Live Camera Status Display Under WebRTC Widget
st.markdown("---")
if ctx.state.playing:
    st.success("🟢 **Camera Active:** Mobile Back Camera live inspection nadandhu kittu irukku.")
    st.info("ℹ️ Camera-va nirutha mela irukkura **'STOP'** button-a click pannunga.")
else:
    st.warning("🔴 **Camera Inactive:** Mobile-la test panna mela irukkura **'START'** button-a press panni camera permission Allow pannunga.")