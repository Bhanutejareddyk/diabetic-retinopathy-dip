# Diabetic Retinopathy Detection Using Retinal Images
**Subject:** Digital Image Processing (DIP) — College Mini / Major Project  
**Author / Demo Ready:** College Final Year Project & Technical Defense

---

## 📋 Project Overview

Diabetic Retinopathy (DR) is the leading cause of preventable blindness among working-age adults worldwide. Early clinical detection through retinal screening is critical to prevent irreversible visual impairment.

Unlike black-box Deep Learning systems, this project implements a **transparent, classical Digital Image Processing (DIP) pipeline** combined with an **interpretable Random Forest ML classifier**. Every intermediate image processing transformation (green channel isolation, CLAHE local enhancement, morphological vessel extraction, optic disc elimination, and lesion contour bounding) is rendered in real time, making it ideal for a **College Viva Voce examination**.

---

## 🛠️ Tech Stack

- **Frontend:** HTML5, CSS3, JavaScript (Vanilla ES6), Tailwind CSS, Chart.js, FontAwesome
- **Backend:** Python Flask (`app.py`), RESTful API
- **Digital Image Processing (DIP):** OpenCV (`cv2`), NumPy, Pillow
- **Machine Learning & Staging:** scikit-learn (`RandomForestClassifier`, ETDRS Clinical Rules)
- **Medical Report Generation:** ReportLab (Automatic PDF export with embedded DIP visuals)
- **Database / Persistence:** Local JSON / SQLite history storage
- **Compute Requirements:** 100% CPU friendly (No GPU required, runs seamlessly on laptops)

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Virtual Environment

```bash
# Navigate to project directory
cd diabetic_retinopathy_dip

# (Optional) Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Sample Images & Train Classifier

```bash
# Generates 25 high-fidelity retinal fundus samples across all 5 DR stages
python download_dataset.py

# Trains Random Forest classifier on DIP features (runs in < 15 seconds)
python train.py
```

### 4. Run Flask Web Application

```bash
python app.py
```

Open your browser and navigate to:  
👉 **`http://localhost:5000`**

---

## 🔬 DIP Pipeline Stages (Step-by-Step)

| Step | DIP Technique | Function / OpenCV Call | Viva Justification |
| :--- | :--- | :--- | :--- |
| **1. Resizing & FOV** | Spatial Normalization & Convex Hull Masking | `cv2.resize`, `cv2.convexHull` | Standardizes input to $512 \times 512$ px; masks out dark non-retinal borders. |
| **2. Grayscale Conversion** | Luminance Mapping | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | Converts 3-channel color to intensity for baseline morphological calculations. |
| **3. Green Channel Extraction** | Spectral Band Isolation | `img[:, :, 1]` | **Peak Hemoglobin Absorption (~570nm)** yields maximum contrast for blood vessels & lesions. |
| **4. Contrast Enhancement (CLAHE)** | Contrast Limited Adaptive Histogram Equalization | `cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))` | Prevents noise amplification in uniform regions by clipping local tile histograms. |
| **5. Denoising** | Non-Linear Filtering | `cv2.medianBlur(img, 3)` | Removes salt-and-pepper sensor noise while preserving sharp vessel boundaries. |
| **6. Vessel Segmentation** | Top-Hat & Bottom-Hat Morphology + Adaptive Threshold | `cv2.morphologyEx(..., cv2.MORPH_TOPHAT)`, `cv2.adaptiveThreshold` | Extracts thin vascular branches; connected components filter noisy single-pixel specs. |
| **7. Optic Disc Masking** | MinMax Brightness Localization + Ellipse Masking | `cv2.minMaxLoc`, `cv2.circle` | Optic disc is naturally bright; masking it avoids false positive exudate detections. |
| **8. Lesion Detection** | Morphological Top-Hat + Circularity & Area Thresholding | `cv2.findContours`, `cv2.arcLength` | Identifies **Microaneurysms** (red circles) and **Hard Exudates** (yellow bounding boxes). |
| **9. Feature Extraction** | Quantitative Retinal Biomarkers | Numerical analysis | Computes Vessel Density %, Microaneurysm Count, Exudate Count & Area, Tortuosity. |
| **10. Classification** | Ensemble Random Forest + ETDRS Clinical Staging | `model.predict_proba` | Predicts one of 5 clinical stages with confidence and probability distribution. |

---

## 📊 The 5 Clinical Stages of Diabetic Retinopathy (ETDRS)

1. **Stage 0 — No DR (Normal):** No microaneurysms, hemorrhages, or exudates. Normal vascular branching. Annual checkup recommended.
2. **Stage 1 — Mild NPDR:** Microaneurysms only (small focal capillary outpouchings). Glycemic control; recheck in 6–12 months.
3. **Stage 2 — Moderate NPDR:** Multiple microaneurysms, blot hemorrhages, and hard lipid exudates. Consult ophthalmologist in 2–4 weeks.
4. **Stage 3 — Severe NPDR:** Extensive hemorrhages (>20 in all 4 quadrants), venous beading, or IRMA. Urgent ophthalmologist referral in 1–2 weeks.
5. **Stage 4 — Proliferative DR (PDR):** Neovascularization (abnormal fragile new vessels), vitreous hemorrhage, high risk of retinal detachment. Emergency intervention (Laser / Anti-VEGF).

---

## 🎓 20 Essential Viva Voce Questions & Answers

### Q1: Why do we extract the Green Channel instead of using Grayscale or Red/Blue channels?
> **Answer:** In retinal fundus photography, oxygenated and deoxygenated hemoglobin absorb green light strongly (approx. 540–570 nm). Consequently, blood vessels, microaneurysms, and hemorrhages appear darkest with the sharpest optical contrast in the green channel. The red channel is over-saturated by the choroid, while the blue channel suffers from poor penetration and severe Rayleigh scatter.

### Q2: What is CLAHE and how does it differ from Standard Histogram Equalization?
> **Answer:** Standard Global Histogram Equalization applies a uniform transformation to the whole image, which often over-amplifies noise in smooth retinal tissue. Contrast Limited Adaptive Histogram Equalization (CLAHE) divides the image into small contextual tiles (typically $8 \times 8$). It clips histogram peaks exceeding a threshold (clip limit = 2.5) and redistributes the clipped pixels evenly before bilinear interpolation, boosting local contrast without blowing out noise.

### Q3: What mathematical morphological operations are used for blood vessel segmentation?
> **Answer:** We use **Top-Hat** ($I - \text{Opening}(I)$) and **Bottom-Hat** ($\text{Closing}(I) - I$) transforms using disk and linear structuring elements. A morphological closing with a structuring element larger than vessel thickness estimates the smooth background illumination. Subtracting the enhanced image from this background highlights the vascular valleys.

### Q4: Why must the Optic Disc be masked before detecting Exudates?
> **Answer:** The optic disc (nerve head) is naturally the brightest, yellowish circular anatomical feature in the fundus. Hard exudates are also bright yellow lipid deposits. Without finding and masking the optic disc, intensity thresholding would falsely flag the optic nerve as a gigantic exudate cluster.

### Q5: How do we detect Microaneurysms vs Hard Exudates?
> **Answer:**
> - **Microaneurysms:** Appear as small, round, dark red spots in the green channel. We invert the green channel (so dark spots become bright peaks) and apply a small circular Top-Hat filter ($7 \times 7$), followed by circularity filtering: $\text{Circularity} = 4\pi \frac{\text{Area}}{\text{Perimeter}^2} > 0.35$.
> - **Hard Exudates:** Appear as bright yellowish-white lipid deposits. Detected by positive Top-Hat and adaptive high-intensity thresholding strictly outside the masked optic disc.

### Q6: What is Adaptive Thresholding and why use it over Otsu's Global Thresholding?
> **Answer:** Retinal fundus images have non-uniform illumination due to pupil vignetting and corneal curvature (periphery is darker than center). Global Otsu thresholding picks a single global cutoff that fails either at the center or borders. **Adaptive Thresholding** computes a dynamic local threshold for every pixel window based on Gaussian-weighted neighboring intensities: $T(x,y) = \text{mean}(W_{x,y}) - C$.

### Q7: What is Field of View (FOV) masking and why is it essential?
> **Answer:** Retinal fundus images have a circular aperture surrounded by black background. Without an FOV mask, the black border pixels skew mean intensity calculations, standard deviation, and vessel density percentages. Creating a convex hull mask restricts all metrics strictly to retinal tissue.

### Q8: What is Vessel Density % and how is it calculated?
> **Answer:**  
> $$\text{Vessel Density} = \left( \frac{\text{Total Vessel Pixels}}{\text{Total Pixels inside Retinal FOV}} \right) \times 100\%$$  
> Normal retinas typically range between 9% and 14%. Abnormally low density indicates capillary drop-out / ischemia; abnormally elevated or chaotic density indicates neovascularization.

### Q9: What is Vessel Tortuosity and what does it indicate clinically?
> **Answer:** Vessel tortuosity measures abnormal curvature, twisting, and dilation of retinal veins and arterioles caused by chronic ischemia, hypertension, and endothelial damage. It is quantified using gradient energy along vessel skeletons via Sobel edge magnitude operators.

### Q10: Why did you choose Classical DIP + Random Forest instead of an end-to-end Deep Learning (CNN) black box?
> **Answer:**  
> 1. **Explainability & Clinical Trust:** Doctors and examiners need to understand *why* a classification was made (e.g., exactly 12 microaneurysms detected).  
> 2. **Low Compute Footprint:** Classical DIP + Random Forest trains in under 15 seconds on a basic dual-core CPU with zero GPU requirement.  
> 3. **Viva Demonstration:** An examiner can see each DIP filter step (Green Channel, CLAHE, Top-Hat, Masking, Contours) rather than hidden weights in a neural network.

### Q11: What is the Median Filter and why is it preferred over Gaussian Blur for denoising?
> **Answer:** Gaussian blur is a linear filter that blurs edges while averaging pixels. A **Median Filter** is a non-linear filter that replaces each pixel with the median of neighboring pixels. It completely eliminates impulse / salt-and-pepper noise while strictly preserving sharp vessel boundaries.

### Q12: What are Connected Components with Stats used for in OpenCV?
> **Answer:** In `cv2.connectedComponentsWithStats()`, 8-connectivity groups binary white pixels into distinct labeled components. We inspect `CC_STAT_AREA` to discard small isolated artifacts ($< 18$ pixels) and measure lesion sizes.

### Q13: What are Cotton-Wool Spots and IRMA?
> **Answer:**
> - **Cotton-Wool Spots (Soft Exudates):** Fluffy white patches caused by nerve fiber layer infarctions (axoplasmic stasis).
> - **IRMA (Intraretinal Microvascular Abnormalities):** Dilated, tortuous shunt vessels that bridge arterioles and venules through capillary bed remodeling in Stage 3 NPDR.

### Q14: What is Neovascularization and why is it dangerous?
> **Answer:** In Stage 4 Proliferative DR, retinal hypoxia triggers the release of Vascular Endothelial Growth Factor (VEGF), stimulating the growth of abnormal, fragile new capillaries. These new vessels rupture easily, causing vitreous hemorrhage, fibrovascular traction, retinal detachment, and sudden blindness.

### Q15: How does your Random Forest classifier work on the extracted features?
> **Answer:** The Random Forest builds an ensemble of 100+ decision trees. Each tree splits nodes using a subset of DIP biomarkers (Microaneurysm Count, Exudate Count & Area, Vessel Density, Vessel Tortuosity, Contrast). The final class prediction is obtained through majority probability voting across all trees.

### Q16: What is the role of Structuring Elements in Morphological Operations?
> **Answer:** A structuring element (kernel) is a shape matrix (ellipse, cross, rectangle) with an origin. For retinal vessels, linear and elliptical kernels align with vessel and lesion geometries to preserve vascular paths while rejecting isotropic noise.

### Q17: How is the Optic Disc detected automatically?
> **Answer:** The retinal image is smoothed with a large Gaussian kernel ($25 \times 25$) to suppress high-frequency vessels. The brightest region within the retinal FOV is located using `cv2.minMaxLoc()`. A threshold at the 88th percentile isolates the circular optic nerve candidate, whose center and radius are bounded.

### Q18: What is the clinical triage recommendation for Stage 2 Moderate DR?
> **Answer:** According to clinical guidelines, Stage 2 patients require referral to an ophthalmologist within 2–4 weeks for Optical Coherence Tomography (OCT) to rule out Diabetic Macular Edema (DME), alongside strict glycemic and lipid management.

### Q19: How does the system handle non-retinal uploaded images?
> **Answer:** `is_likely_fundus_image()` validates color distribution (verifying dominant red-green tissue reflectance over blue) and confirms the presence of circular field-of-view aperture. If a user uploads an invalid photo (e.g. landscape or object), an error is returned before processing.

### Q20: How can this system be deployed for telemedicine or remote clinics?
> **Answer:** The Flask application exposes RESTful JSON endpoints (`/api/analyze`, `/api/report/<id>`) that can be deployed on a lightweight VPS, Docker container, Render, or AWS Lambda. Peripheral clinics can upload fundus photos from handheld fundus cameras and receive instant PDF triage reports within 2 seconds.

---

## 📁 Project File Structure

```
diabetic_retinopathy_dip/
├── app.py                  # Flask web backend & API routes
├── dip_pipeline.py         # Complete classical DIP image processing engine
├── model.py                # Random Forest ML classifier & ETDRS clinical staging
├── train.py                # Dataset feature extraction & model training script
├── download_dataset.py     # Generates 25 retinal samples & manages datasets
├── report_generator.py     # ReportLab automated PDF diagnostic report generator
├── requirements.txt        # Python dependency manifest
├── README.md               # Project guide, documentation & Viva Q&A
├── dataset/                # Staged dataset folders (0_No_DR to 4_Proliferative_DR)
├── templates/
│   └── index.html          # Responsive single page application UI
└── static/
    ├── css/style.css       # Custom styles & medical layout theme
    ├── js/main.js          # Interactive frontend controller & AJAX pipeline
    ├── samples/            # 25 pre-generated fundus samples (5 per class)
    └── processed/          # Session cache for intermediate DIP outputs & PDFs
```

---

## 🏆 Project Demonstration Checklist

- [x] Run with simple `python app.py` on `localhost:5000`
- [x] Drag & drop retinal image or 1-click test of all 5 stages
- [x] All 6 intermediate DIP images displayed with explanations
- [x] Click-to-enlarge modal lightbox for pixel inspection
- [x] Dynamic confidence score & probability distribution chart
- [x] Quantitative biomarkers table (vessel density, lesions, tortuosity)
- [x] Downloadable official PDF medical report with embedded DIP steps
- [x] Persistent history table and diagnostic analytics donut chart
- [x] 100% CPU compatible, no GPU or cloud API keys required
