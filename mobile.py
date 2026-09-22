import os
import glob
import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="AI Fabric Defect Inspector", layout="wide")

# Mobile Friendly CSS Styling
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
st.caption("Automated Pattern & Defect Inspection System")

# GitHub Repository-la MASTER Folder Path setup
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

# Sidebar Settings
st.sidebar.header("⚙️ QC Inspection Controls")
sensitivity = st.sidebar.slider("Color Defect Sensitivity", 10, 100, 35)
min_defect_area = st.sidebar.slider("Min Defect Area (Pixels)", 20, 1000, 80)

def match_and_inspect_best(test_bgr, master_imgs, sens, min_area):
    if not master_imgs:
        return None, -1, "MASTER folder-la GitHub Repo-la images illai!"

    # Find closest matching Master Image using SIFT
    sift = cv2.SIFT_create(nfeatures=500)
    gray_test = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    kp_test, des_test = sift.detectAndCompute(gray_test, None)

    best_master = master_imgs[0]
    best_score = -1

    if des_test is not None:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        for m_img in master_imgs:
            gray_m = cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY)
            kp_m, des_m = sift.detectAndCompute(gray_m, None)
            if des_m is not None:
                matches = bf.match(des_m, des_test)
                if len(matches) > best_score:
                    best_score = len(matches)
                    best_master = m_img

    # Resize test image to matched master image dimensions
    h_m, w_m = best_master.shape[:2]
    test_resized = cv2.resize(test_bgr, (w_m, h_m))

    # Convert to LAB Color space for defect comparison
    master_lab = cv2.cvtColor(best_master, cv2.COLOR_BGR2LAB).astype(np.float32)
    test_lab = cv2.cvtColor(test_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Calculate Delta Color Difference
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
            # Red Bounding Box for Defects
            cv2.rectangle(output_img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
            cv2.putText(output_img, "DEFECT", (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            defect_count += 1

    return output_img, defect_count, "Success"

# Tab View
tab1, tab2 = st.tabs(["📸 Mobile Snap & Scan", "📁 Upload Image Inspection"])

with tab1:
    st.subheader("Mobile Quick Camera Scan")
    st.info("💡 Camera Permission 'Allow' pannunga. Photo click pannadhum Result theryum.")
    
    camera_file = st.camera_input("Take Fabric Photo", key="mobile_camera")

    if camera_file is not None:
        bytes_data = camera_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        output_img, defects, msg = match_and_inspect_best(cv_img, master_list, sensitivity, min_defect_area)

        if defects == -1:
            st.error(f"❌ Error: {msg}")
        elif defects == 0:
            st.success("🟢 STATUS: PASSED (NO DEFECT DETECTED)")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_column_width=True)
        else:
            st.error(f"🔴 STATUS: REJECTED ({defects} DEFECT(S) DETECTED)")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_column_width=True)

with tab2:
    st.subheader("Upload Fabric Photo")
    uploaded_file = st.file_uploader("Choose a fabric image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        output_img, defects, msg = match_and_inspect_best(cv_img, master_list, sensitivity, min_defect_area)

        if defects == -1:
            st.error(f"❌ Error: {msg}")
        elif defects == 0:
            st.success("🟢 STATUS: PASSED (NO DEFECT DETECTED)")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_column_width=True)
        else:
            st.error(f"🔴 STATUS: REJECTED ({defects} DEFECT(S) DETECTED)")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_column_width=True)
