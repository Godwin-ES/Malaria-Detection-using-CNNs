# 🦠 Malaria Detection with CNN + Streamlit

This project demonstrates a **deep learning-based malaria detection system** using Convolutional Neural Networks (CNNs). The app is deployed with **Streamlit**, allowing users to upload blood cell images for automated diagnosis.

The system performs two key tasks:

1. **Cell Image Validation** – Ensures the uploaded image is a valid blood cell image (flags non-cell images like dogs, cars, etc.).
2. **Malaria Detection** – If the image is valid, the CNN predicts whether the cell is **Infected** or **Uninfected** with malaria.

---

## 🚀 Features

* Upload blood cell images in `.jpg`, `.jpeg`, or `.png` formats.
* Automatically **rejects invalid/non-cell images**.
* Predicts **Infected / Uninfected** with associated confidence scores.
* Displays analysis time for each prediction.
* Built with **TensorFlow** and deployed via **Streamlit** for accessibility.

---

## 📂 Repository Structure

```
├── classify.pkl       # Model to validate whether uploaded image is a blood cell
├── model.pkl          # CNN model trained for malaria detection
├── modeldeploy.py     # Streamlit deployment script
├── requirements.txt   # Python dependencies
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Godwin-ES/Malaria-Detection-using-CNNs
cd Malaria-Detection-using-CNNs
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run modeldeploy.py
```

---

## 🧠 Model Details

* **Cell Classifier (`classify.pkl`)**:
  Binary classifier that flags whether an uploaded image is a valid cell image.

* **Malaria Detector (`model.pkl`)**:
  CNN trained on malaria-infected and uninfected blood cell images.

  * Input size: **128 × 128** pixels
  * Output: Binary classification (**Infected / Uninfected**)

---

## 📸 Example Workflow

1. Upload a cell image (`.jpg/.png/.jpeg`).
2. System checks if the uploaded file is a valid cell image.

   * ❌ Invalid → Error message displayed.
   * ✅ Valid → Passes to malaria detection model.
3. Malaria prediction result with **confidence score** and **analysis time** is displayed.

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`:

```
joblib
numpy
Pillow
streamlit
tensorflow==2.17.0
```

---
