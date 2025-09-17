# 🚗 Smart Car Parking Detection using Deep Learning + OpenCV 🎥

An AI-powered parking management system that detects whether a parking slot is **occupied** or **empty** in real-time from CCTV/video footage.

This project combines **Streamlit**, **OpenCV**, and a **CNN model (TensorFlow/Keras)** to provide an interactive tool for:

* Annotating parking slots
* Running real-time detection on uploaded parking lot videos
* Displaying live occupancy counters

---

## 📌 Features

✅ Interactive slot annotation using Streamlit canvas
✅ Real-time parking slot classification (Occupied / Empty)
✅ Occupancy summary counter (live stats)
✅ Save & load parking slot positions (`pickle`)
✅ Processed video display inside Streamlit
✅ Lightweight CNN model for fast inference

---

## 🛠️ Tech Stack

* **Python 3.8+**
* **Streamlit** – UI & interactivity
* **OpenCV** – video/image processing
* **TensorFlow/Keras** – CNN model training & inference
* **NumPy** – data manipulation
* **PIL** – image handling

---
## ⚙️ Installation

1. Clone this repo:

   ```bash
   git clone https://github.com/AdityaTagde/car_parking_system.git
   cd car_parking_system
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## 🚀 Usage

### 1️⃣ Annotate Slots

* Select **"Annotate Slots"** mode.
* Draw rectangles over parking slots on the uploaded image.
* Click **Save Slots** to store coordinates (`CarParkPos`).

### 2️⃣ Run Detection

* Switch to **"Run Detection"** mode.
* Upload a parking lot **video file**.
* The model will classify each slot as:

  * 🟢 **Empty**
  * 🔴 **Occupied**
* Live occupancy stats are displayed on screen.

---

## 📊 Model Details

* Architecture: **Convolutional Neural Network (CNN)**
* Input: Cropped slot image `(100x50)`
* Output: Binary classification → `Empty (0)` / `Occupied (1)`
* Framework: **TensorFlow/Keras**

---

## 📸 Demo Preview

![Parking Detection Screenshot](https://github.com/AdityaTagde/car_parking_system/blob/main/1.png)
![Parking Detection Screenshot](https://github.com/AdityaTagde/car_parking_system/blob/main/2.png)
![Parking Detection Screenshot](https://github.com/AdityaTagde/car_parking_system/blob/main/3.png)
![Parking Detection Screenshot](https://github.com/AdityaTagde/car_parking_system/blob/main/4.png)


---

## 📌 Future Improvements

* Add support for **live CCTV feed** (RTSP).
* Enhance accuracy with **transfer learning (MobileNetV2/ResNet)**.
* Deploy on **Raspberry Pi/Edge devices** for IoT smart parking.
* Create a **dashboard with analytics** (occupancy trends, reports).

---

## 🙌 Acknowledgements

This project was inspired by real-world **Smart City** applications, blending **Computer Vision** + **Deep Learning** to optimize parking space management.

---

## 📢 Hashtags for Social/Portfolio Posts

\#ComputerVision #DeepLearning #OpenCV #TensorFlow #ArtificialIntelligence #MachineLearning #Python #SmartParking #AIProjects #DataScience #SmartCity #IoT

---

