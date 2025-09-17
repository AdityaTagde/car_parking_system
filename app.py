import streamlit as st
import cv2
import numpy as np
import tempfile
import pickle
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image
st.set_page_config(layout="wide")
st.title("🚗 Smart Parking Detection")

# Sidebar toggle for annotation mode
mode = st.sidebar.radio("Select Mode", ["Annotate Slots", "Run Detection"])

# ---------------------- SLOT ANNOTATION ---------------------- #
if mode == "Annotate Slots":
    st.markdown("### Draw rectangles to mark parking slots")

    # Load background image
    bg_img = cv2.imread("parks.png")
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

    # Convert to PIL
    bg_img_pil = Image.fromarray(bg_img)
    img_w, img_h = bg_img_pil.size   

    # Streamlit drawable canvas
    canvas = st_canvas(
        fill_color="rgba(0, 255, 0, 0.2)",  # semi-transparent fill
        stroke_color="green",
        stroke_width=1,                      # ✅ thin border
        background_image=bg_img_pil,
        update_streamlit=True,
        height=img_h,
        width=img_w,
        drawing_mode="rect",
        key="canvas",
    )

    rectangles = []
    if canvas.json_data is not None:
        for obj in canvas.json_data.get("objects", []):
            if obj["type"] == "rect":
                left = max(0, int(obj["left"]))
                top = max(0, int(obj["top"]))
                rect_w = int(obj["width"] * obj.get("scaleX", 1))
                rect_h = int(obj["height"] * obj.get("scaleY", 1))
                if rect_w > 5 and rect_h > 5:  # filter out tiny boxes
                    rectangles.append((left, top, left + rect_w, top + rect_h))

    # Show preview of drawn slots
    st.write(f"🖼️ Rectangles drawn: {len(rectangles)}")

    # ✅ Single Save Button with unique key
    if rectangles and st.button("💾 Save Slots", key="save_slots_btn"):
        with open("CarParkPos", "wb") as f:
            pickle.dump(rectangles, f)
        st.success(f"Saved {len(rectangles)} parking slots!")


# ---------------------- SLOT DETECTION ---------------------- #
elif mode == "Run Detection":
    st.markdown("### Upload a parking lot video")

    # Load saved rectangles
    try:
        with open("CarParkPos", "rb") as f:
            rectangles = pickle.load(f)
        st.info(f"Loaded {len(rectangles)} parking slots.")
    except FileNotFoundError:
        st.error("⚠️ No saved slots found. Please annotate first.")
        st.stop()

    # Load model
    model = load_model("car_parking_model.h5")

    # Upload video
    video_file = st.file_uploader('', type=["mp4", "avi", "mov"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        frame_skip = 2
        frame_count = 0
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue  # skip frames for speed

            occupied_count, empty_count = 0, 0
            batch_slots, coords = [], []

            for rect in rectangles:
                x1, y1, x2, y2 = rect
                slot = frame[y1:y2, x1:x2]
                if slot.size == 0:
                    continue
                slot_resized = cv2.resize(slot, (100, 50)) / 255.0
                batch_slots.append(slot_resized)
                coords.append((x1, y1, x2, y2))

            if batch_slots:
                preds = model.predict(np.array(batch_slots), verbose=0).flatten()
                for (x1, y1, x2, y2), pred in zip(coords, preds):
                    if pred > 0.5:
                        color = (0, 0, 255)
                        occupied_count += 1
                    else:
                        color = (0, 255, 0)
                        empty_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw summary box
            cv2.rectangle(frame, (10, 10), (230, 70), (50, 50, 50), -1)
            cv2.putText(frame, f"Empty: {empty_count}", (20, 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
            cv2.putText(frame, f"Occupied: {occupied_count}", (20, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)

            stframe.image(frame, channels="BGR")

        cap.release()
        
        st.success("✅ Video processing finished!")

