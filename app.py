from PIL import Image
import numpy as np
import cv2
import streamlit as st
from utils.contours import detect_contours
from utils.threshold import adap_thresh, glob_thresh

st.set_page_config(
    page_title = "Image Thresholding and Conyour Detection",layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
        
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(120deg, #E8F5E9, #F3E5F5);
            animation: gradientFlow 15s ease infinite;
            background-size: 400% 400%;
        }
        @keyframes gradientFlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        [data-testid="stHeader"] {
            background-color: rgba(255, 255, 255, 0);
        }

        .main-title {
            text-align: center;
            font-weight: 900;
            background: #7B1FA2;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.2rem;
            letter-spacing: 1px;
            text-shadow: 1px 1px 10px rgba(150, 50, 255, 0.3);
            margin-bottom: 0.5em;
            transition: transform 0.4s ease;
        }

        .main-title:hover {
            transform: scale(1.03);
        }

        .section-title {
            font-weight: 700;
            color: #6A1B9A;
            font-size: 1.4rem;
            margin-top: 1.2em;
            text-shadow: 0 0 8px rgba(155, 50, 255, 0.15);
        }

        .stFileUploader, .stSelectbox, .stCheckbox, .stRadio, .stSlider {
            background-color: rgba(255,255,255,0.8);
            border-radius: 15px;
            padding: 1em;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            transition: all 0.3s ease-in-out;
        }

        .stFileUploader:hover, .stSelectbox:hover, .stSlider:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 25px rgba(0,0,0,0.12);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            justify-content: center;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #fff;
            padding: 0.6rem 1.2rem;
            border-radius: 12px;
            font-weight: 600;
            color: #6A1B9A;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: #E1BEE7;
            transform: translateY(-3px);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #8E24AA, #CE93D8);
            color: white;
            font-weight: 700;
        }

        .stImage img {
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
        }

        .stImage img:hover {
            transform: scale(1.03);
            box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        }

        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, #8E24AA, #F06292, #AED581);
            border-radius: 10px;
            margin: 2em 0;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 class='main-title'>Image Thresholding and Contour Detection</h1>", unsafe_allow_html=True)
st.write("---")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_container_width=True)

    st.sidebar.header("⚙️ Threshold Settings")
    method = st.sidebar.selectbox("Select Thresholding Method", ["Global", "Adaptive"])

    if method == "Global":
        thresh_value = st.sidebar.slider("Threshold Value", 0, 255, 127)
        th_type = st.sidebar.selectbox(
            "Threshold Type",
            ["THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO", "THRESH_TOZERO_INV"]
        )
        th_type_cv = getattr(cv2, th_type)
        thresh_img = glob_thresh(img, thresh_value, th_type_cv)

    else:
        ada_method = st.sidebar.selectbox("Adaptive Method", ["ADAPTIVE_THRESH_MEAN_C", "ADAPTIVE_THRESH_GAUSSIAN_C"])
        thresh_type = st.sidebar.selectbox("Threshold Type", ["THRESH_BINARY", "THRESH_BINARY_INV"])
        block_size = st.sidebar.slider("Block Size (odd number)", 3, 51, 11, step=2)
        C = st.sidebar.slider("C value", 0, 10, 2)

        ada_method_cv = getattr(cv2, ada_method)
        thresh_type_cv = getattr(cv2, thresh_type)
        thresh_img = adap_thresh(img, ada_method_cv, thresh_type_cv, block_size, C)

    st.subheader("🧩 Thresholded Image")
    st.image(thresh_img, use_container_width=True, caption="Thresholded Image")

    contoured_img, count = detect_contours(img, thresh_img)
    st.subheader(f"🔍 Contour Detection \n Objects Found: {count}")
    st.image(contoured_img, use_container_width=True, caption="Contours Highlighted")

else:
    st.info("👆 Upload an image to begin.")
