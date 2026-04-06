# 🦴 Pediatric Fracture Detection Web App

A deep learning-based web application that detects **pediatric bone fractures** from wrist X-ray images using a trained **YOLOv8 model**.

---

## 🚀 Live Demo

https://pediatric-wrist-fracture-detection.streamlit.app/

---

## 📌 Features

* 🧠 AI-powered fracture detection using YOLOv8
* 📸 Upload pediatric wrist X-ray images
* 🎯 Detects and highlights fracture regions
* 📊 Displays confidence scores for predictions
* ⚙️ Adjustable confidence threshold
* 📋 Tabular summary of detections
* 🖼️ Side-by-side original and annotated images

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Model:** YOLOv8 (Ultralytics)
* **Libraries:**

  * PyTorch
  * OpenCV (headless)
  * Pillow
  * Pandas
  * Hugging Face Hub

---

## 🧠 Model Details

* Model trained on pediatric wrist X-ray dataset
* Detects fracture regions using object detection
* Hosted on **Hugging Face Hub** for efficient loading

---

## 📂 Project Structure

```
Pediatric_App/
│── app.py
│── inference.py
│── utils.py
│── requirements.txt
│── README.md
│── assets/
│── outputs/
│── sample_images/
```

---

## ⚙️ Installation & Local Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/pediatric-fracture-detection.git
cd pediatric-fracture-detection
```

### 2. Create virtual environment

```
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the app

```
streamlit run app.py
```

---

## 🌍 Deployment

This app is deployed using **Streamlit Cloud**.

Steps:

1. Push code to GitHub
2. Connect repo to Streamlit Cloud
3. Deploy using `app.py` as entry point

---

## ⚠️ Notes

* Model file (`.pt`) is not stored in the repository
* It is dynamically downloaded from Hugging Face during runtime
* First run may take a few seconds for model download

---

## 📸 Sample Output


---

## 💡 Future Improvements

* Add PDF report generation
* Multi-image upload support
* Improve detection accuracy
* Add user authentication

---

## 👨‍💻 Author

**Greeshwanth Ankem**

* Passionate about AI, ML, and Full Stack Development
* Interested in building impactful real-world applications

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!

---
