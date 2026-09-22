import os
import glob
import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="AI Fabric Defect Inspector", layout="wide")

# CSS Styling to make Camera View and Images 100% Full Width on Mobile
st.markdown(
    """
    <style>
    .main .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        padding-top: 1rem !important;
    }
    div[data-testid="stCameraInput"] {
        width: 100% !important;
    }
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        border-radius: 10px;
    }
    img {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏭 Real-Time Fabric & Bag Defect Inspector")
st.caption("Automated Pattern Inspection System")

# Dynamic Folder Path for GitHub Repo
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

# Sidebar Settings
st.sidebar.header("⚙️ QC Inspection Controls")
sensitivity = st.sidebar.slider("Color Defect Sensitivity", 10, 100, 40)
min_defect_area = st.sidebar.slider("Min Defect Area (Pixels)", 20, 1000, 80)

def match_and_inspect(test_bgr, master_db, sens, min_area):
    if not master_db:
        return None, -1, "No Master Image found in 'MASTER' folder."

    # Pick first master template image
    master_name, master_bgr = list(master_db.items())[0]
    
    h_m, w_m = master_bgr.shape[:2]
    test_resized = cv2.resize(test_bgr, (w_m, h_m))

    # Convert to LAB Color space for defect comparison
    master_lab = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    test_lab = cv2.cvtColor(test_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Color difference calculation
    da = master_lab[:, :, 1] - test_lab[:, :, 1]
    db = master_lab[:, :, 2] - test_lab[:, :, 2]
    color_diff = np.sqrt(da**2 + db**2)
    color_diff = np.clip(color_diff, 0, 255).astype(np.uint8)

    blurred_diff = cv2.GaussianBlur(color_diff, (5, 5), 0)
    _, thresh = cv2.threshold(blurred_diff, sens, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_img = test_resized.copy()
    defect_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= (w_m * h_m * 0.35):
            x, y, bw, bh = cv2.boundingRect(contour)
            cv2.rectangle(output_img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
            cv2.putText(output_img, "DEFECT", (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            defect_count += 1

    return output_img, defect_count, "Success"

# Tab interface for high stability on mobile
tab1, tab2 = st.tabs(["📸 Mobile Snap & Scan (Stable)", "🎥 Live WebRTC Stream"])

with tab1:
    st.subheader("Mobile Quick Inspection")
    camera_file = st.camera_input("Take Fabric Photo", key="mobile_camera")

    if camera_file is not None:
        bytes_data = camera_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        output_img, defects, msg = match_and_inspect(cv_img, master_dict, sensitivity, min_defect_area)

        if defects == -1:
            st.error(f"❌ Error: {msg}")
        elif defects == 0:
            st.success("🟢 STATUS: PASSED (NO DEFECT FOUND)")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_column_width=True)
        else:
            st.error(f"🔴 STATUS: REJECTED ({defects} DEFECT(S) DETECTED)")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_column_width=True)

with tab2:
    st.subheader("Live Stream Inspection")
    
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

    class SimpleQCProcessor(VideoProcessorBase):
        def __init__(self):
            self.master_db = master_dict
            self.sens = sensitivity
            self.min_area = min_defect_area

        def recv(self, frame):
            img_bgr = frame.to_ndarray(format="bgr24")
            
            if not self.master_db:
                cv2.putText(img_bgr, "NO MASTER IMAGE IN GITHUB REPO", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                return frame.from_ndarray(img_bgr, format="bgr24")

            output_img, defects, _ = match_and_inspect(img_bgr, self.master_db, self.sens, self.min_area)
            
            # Header Status Overlay
            cv2.rectangle(output_img, (0, 0), (output_img.shape[1], 50), (0, 0, 0), -1)
            
            if defects == 0:
                cv2.putText(output_img, "STATUS: PASSED (NO DEFECT)", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(output_img, f"STATUS: REJECTED ({defects} DEFECTS)", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            return frame.from_ndarray(output_img, format="bgr24")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="qc-live-stream",
        video_processor_factory=SimpleQCProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True
    )
