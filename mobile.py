import os
import glob
import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="AI Fabric Defect Inspector", layout="wide")

# Mobile Friendly CSS
st.markdown(
    """
    <style>
    .main .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        padding-top: 1rem !important;
    }
    img {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏭 Real-Time Fabric & Bag Defect Inspector")
st.caption("Automated Pattern & Defect Inspection System")

# Master Image Folder Path in GitHub Repo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_FOLDER = os.path.join(BASE_DIR, "MASTER")

if not os.path.exists(MASTER_FOLDER):
    os.makedirs(MASTER_FOLDER, exist_ok=True)

def load_master_images(folder_path):
    master_images = []
    if os.path.exists(folder_path):
        all_files = glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]")) + \
                    glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]")) + \
                    glob.glob(os.path.join(folder_path, "*.[jJ][pP][eE][gG]"))
        
        for fpath in all_files:
            img = cv2.imread(fpath)
            if img is not None:
                master_images.append(img)
    return master_images

master_list = load_master_images(MASTER_FOLDER)

# Sidebar Controls in English
st.sidebar.header("⚙️ QC Inspection Controls")
sensitivity = st.sidebar.slider("Color Defect Sensitivity", 10, 100, 35)
min_defect_area = st.sidebar.slider("Min Defect Area (Pixels)", 20, 1000, 80)
match_threshold = st.sidebar.slider("Style Match Strictness", 10, 100, 25)

def match_and_inspect_strict(test_bgr, master_imgs, sens, min_area, strictness):
    if not master_imgs:
        return None, -1, "No Master Images found in 'MASTER' directory!"

    # SIFT Feature Matching for Style Validation
    sift = cv2.SIFT_create(nfeatures=1000)
    gray_test = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    kp_test, des_test = sift.detectAndCompute(gray_test, None)

    if des_test is None or len(kp_test) < 15:
        return test_bgr, -2, "INVALID IMAGE: Blurred or unreadable fabric pattern."

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    best_master = None
    best_score = 0

    for m_img in master_imgs:
        gray_m = cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY)
        kp_m, des_m = sift.detectAndCompute(gray_m, None)
        if des_m is not None:
            matches = bf.match(des_m, des_test)
            good_matches = [m for m in matches if m.distance < 280]
            if len(good_matches) > best_score:
                best_score = len(good_matches)
                best_master = m_img

    # Reject if Style does not match Master Template
    if best_score < strictness or best_master is None:
        return test_bgr, -3, f"STYLE MISMATCH: Fabric style does not match Master database! (Match Score: {best_score}/{strictness})"

    # Resize test image to matched master image dimensions
    h_m, w_m = best_master.shape[:2]
    test_resized = cv2.resize(test_bgr, (w_m, h_m))

    # LAB Color Space Comparison
    master_lab = cv2.cvtColor(best_master, cv2.COLOR_BGR2LAB).astype(np.float32)
    test_lab = cv2.cvtColor(test_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

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

st.subheader("📸 Mobile Camera / File Inspection")
uploaded_file = st.file_uploader("Take Photo or Upload Fabric Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        output_img, defects, msg = match_and_inspect_strict(
            cv_img, master_list, sensitivity, min_defect_area, match_threshold
        )

        if defects == -1:
            st.error(f"❌ Master Database Error: {msg}")
        elif defects == -2 or defects == -3:
            st.warning(f"⚠️ REJECTED: {msg}")
            st.image(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        elif defects == 0:
            st.success("🟢 STATUS: PASSED (NO DEFECT DETECTED)")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.error(f"🔴 STATUS: REJECTED ({defects} DEFECT(S) DETECTED)")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
    except Exception as e:
        st.error(f"Processing Error: Unable to process image ({str(e)})")
