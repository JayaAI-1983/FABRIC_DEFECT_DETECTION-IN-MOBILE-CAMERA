import os
import glob
import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="AI Fabric Defect Inspector", layout="wide")

# Mobile First Layout Styling
st.markdown(
    """
    <style>
    .main .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        padding-top: 0.5rem !important;
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

st.title("🏭 AI Fabric & Bag Defect Inspector")
st.caption("Real-Time Automated Quality Control & Defect Localization System")

# GitHub Repository Master Image Folder Configuration
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
                master_images.append((os.path.basename(fpath), img))
    return master_images

master_list = load_master_images(MASTER_FOLDER)

# Sidebar QC Controls
st.sidebar.header("⚙️ QC Inspection Parameters")
sensitivity = st.sidebar.slider("Color/Texture Difference Threshold", 10, 100, 35)
min_defect_area = st.sidebar.slider("Minimum Defect Area (Pixels)", 20, 1000, 80)
style_strictness = st.sidebar.slider("Style Match Strictness Score", 5, 100, 20)

def get_location_label(x, y, w, h, img_w, img_h):
    """Determines spatial region of defect on fabric surface"""
    center_x = x + (w / 2)
    center_y = y + (h / 2)
    
    col = "Left" if center_x < img_w / 3 else ("Right" if center_x > (2 * img_w / 3) else "Center")
    row = "Top" if center_y < img_h / 3 else ("Bottom" if center_y > (2 * img_h / 3) else "Middle")
    
    return f"{row}-{col}"

def inspect_fabric_pattern(test_bgr, master_db, sens, min_area, strictness):
    if not master_db:
        return None, [], -1, "No Master Template found in 'MASTER' repository directory."

    # SIFT Feature Alignment & Style Verification
    sift = cv2.SIFT_create(nfeatures=1000)
    gray_test = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    kp_test, des_test = sift.detectAndCompute(gray_test, None)

    if des_test is None or len(kp_test) < 10:
        return test_bgr, [], -2, "UNREADABLE PATTERN: Image blur or insufficient fabric lighting."

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    best_master = None
    best_score = 0
    matched_file_name = ""

    for fname, m_img in master_db:
        gray_m = cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY)
        kp_m, des_m = sift.detectAndCompute(gray_m, None)
        if des_m is not None:
            matches = bf.match(des_m, des_test)
            good_matches = [m for m in matches if m.distance < 280]
            if len(good_matches) > best_score:
                best_score = len(good_matches)
                best_master = m_img
                matched_file_name = fname

    # Style Match Verification Guard
    if best_score < strictness or best_master is None:
        return test_bgr, [], -3, f"STYLE MISMATCH: Scanned fabric does not match Master Database (Match Score: {best_score}/{strictness})."

    # Normalize captured image resolution to match reference master
    h_m, w_m = best_master.shape[:2]
    test_resized = cv2.resize(test_bgr, (w_m, h_m))

    # LAB Color Space Analytics for Stain / Density Defect Detection
    master_lab = cv2.cvtColor(best_master, cv2.COLOR_BGR2LAB).astype(np.float32)
    test_lab = cv2.cvtColor(test_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

    da = master_lab[:, :, 1] - test_lab[:, :, 1]
    db = master_lab[:, :, 2] - test_lab[:, :, 2]
    dL = master_lab[:, :, 0] - test_lab[:, :, 0]
    
    color_diff = np.sqrt(da**2 + db**2 + dL**2)
    color_diff = np.clip(color_diff, 0, 255).astype(np.uint8)

    blurred_diff = cv2.GaussianBlur(color_diff, (5, 5), 0)
    _, thresh = cv2.threshold(blurred_diff, sens, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_img = test_resized.copy()
    defect_summary = []
    defect_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= (w_m * h_m * 0.35):
            defect_count += 1
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Spatial position identification
            location = get_location_label(x, y, bw, bh, w_m, h_m)
            
            # Defect type categorization based on luminance difference
            mean_dl = np.mean(dL[y:y+bh, x:x+bw])
            if abs(mean_dl) > 25:
                defect_type = "Stain / Color Patch"
            else:
                defect_type = "Embroidery / Weave Density Anomaly"

            # Draw Annotation on Output Image
            cv2.rectangle(output_img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
            cv2.putText(output_img, f"#{defect_count} {defect_type}", (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            defect_summary.append({
                "id": defect_count,
                "type": defect_type,
                "location": location,
                "area_px": int(area)
            })

    return output_img, defect_summary, defect_count, matched_file_name

# Direct Mobile Camera Component
st.markdown("### 📸 Live Camera Capture")
camera_photo = st.camera_input("Point camera at fabric and click Take Photo", key="direct_camera")

if camera_photo is not None:
    try:
        bytes_data = camera_photo.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        output_img, summary, status_code, extra_info = inspect_fabric_pattern(
            cv_img, master_list, sensitivity, min_defect_area, style_strictness
        )

        if status_code == -1:
            st.error(f"❌ Configuration Error: {extra_info}")
        elif status_code == -2 or status_code == -3:
            st.warning(f"⚠️ REJECTED: {extra_info}")
            st.image(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        elif status_code == 0:
            st.success(f"🟢 STATUS: PASSED — Matched Master Pattern [{extra_info}] (NO DEFECT DETECTED)")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.error(f"🔴 STATUS: REJECTED ({status_code} DEFECT(S) DETECTED)")
            
            # Split View Layout: Image with Red Marks & Defect Summary Panel
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), caption="Annotated Defect Inspection View", use_container_width=True)
                
            with col2:
                st.markdown("#### 📋 Quality Inspection Summary")
                st.write(f"**Matched Pattern:** `{extra_info}`")
                
                for item in summary:
                    st.markdown(f"""
                    **Defect #{item['id']}**
                    - **Type:** {item['type']}
                    - **Location:** {item['location']} Region
                    - **Defect Size:** {item['area_px']} pixels
                    ---
                    """)

    except Exception as e:
        st.error(f"Processing Error: Unable to complete fabric inspection ({str(e)})")
