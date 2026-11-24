"""
dip_pipeline.py
===============
Digital Image Processing (DIP) Pipeline for Diabetic Retinopathy Detection.

Viva Highlights:
- Green Channel: Provides highest optical contrast for hemoglobin / blood vessels.
- CLAHE: Contrast Limited Adaptive Histogram Equalization avoids noise amplification.
- Morphological Top-Hat: Enhances fine vessel structures and microaneurysms.
- Optic Disc Masking: Robust morphological closing prevents false positive exudate classifications.
- Lesion Segmentation: Detects microaneurysms (red circular lesions) and exudates (bright lipid deposits).
"""

import os
import cv2
import numpy as np


class DIPPipeline:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def get_retinal_mask(self, img_bgr):
        """
        Creates circular/elliptical Field-of-View (FOV) mask of the retina.
        Eliminates dark background borders so metrics are computed strictly inside retina.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        if contours:
            largest_c = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_c)
            cv2.drawContours(clean_mask, [hull], -1, 255, thickness=-1)
            erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            clean_mask = cv2.erode(clean_mask, erode_k)
        else:
            h, w = gray.shape
            cv2.circle(clean_mask, (w // 2, h // 2), int(min(h, w) * 0.45), 255, -1)

        return clean_mask

    def extract_green_channel(self, img_bgr):
        """
        Extracts Green Channel.
        Viva Question: Why green channel?
        Answer: Hemoglobin absorbs green light (approx 540-570 nm) maximally,
        making retinal blood vessels appear darkest with peak contrast against the fundus background.
        Red is saturated, Blue has poor depth penetration and strong Rayleigh scattering.
        """
        return img_bgr[:, :, 1]

    def apply_clahe(self, green_img, clip_limit=2.5, tile_size=(8, 8)):
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Divides image into 8x8 contextual tiles and applies equalization locally
        with contrast clipping to prevent noise amplification in uniform regions.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(green_img)

    def denoise(self, img_gray):
        """
        Applies Median filter.
        Median filter removes salt-and-pepper noise while preserving vessel edge sharpness.
        """
        return cv2.medianBlur(img_gray, 3)

    def segment_blood_vessels(self, enhanced_img, fov_mask):
        """
        Blood vessel segmentation using Morphological Bottom-Hat / Top-Hat
        and Adaptive Thresholding.
        """
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        background = cv2.morphologyEx(enhanced_img, cv2.MORPH_CLOSE, kernel_bg)

        vessel_contrast = cv2.subtract(background, enhanced_img)

        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        tophat = cv2.morphologyEx(vessel_contrast, cv2.MORPH_TOPHAT, kernel_line)
        vessel_signal = cv2.addWeighted(vessel_contrast, 0.7, tophat, 0.3, 0)

        vessels_thresh = cv2.adaptiveThreshold(
            vessel_signal,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            17,
            -3
        )

        vessels_clean = cv2.bitwise_and(vessels_thresh, vessels_thresh, mask=fov_mask)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessels_clean, connectivity=8)
        filtered_vessels = np.zeros_like(vessels_clean)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 18:
                filtered_vessels[labels == i] = 255

        return filtered_vessels

    def detect_optic_disc(self, green_img, fov_mask):
        """
        Detects Optic Disc (brightest circular/elliptical region).
        Uses morphological closing to bridge traversing blood vessels and creates
        a complete coverage mask.
        """
        kernel_fuse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        closed = cv2.morphologyEx(green_img, cv2.MORPH_CLOSE, kernel_fuse)
        blurred = cv2.GaussianBlur(closed, (21, 21), 0)

        min_v, max_v, min_l, max_l = cv2.minMaxLoc(blurred, mask=fov_mask)
        thresh = int(max_v * 0.80)
        _, bright = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
        bright = cv2.bitwise_and(bright, fov_mask)

        num_c, labels, stats, centroids = cv2.connectedComponentsWithStats(bright)
        disc_mask = np.zeros_like(green_img)
        disc_center = max_l
        disc_radius = 48

        if num_c > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            x = stats[largest_label, cv2.CC_STAT_LEFT]
            y = stats[largest_label, cv2.CC_STAT_TOP]
            w = stats[largest_label, cv2.CC_STAT_WIDTH]
            h = stats[largest_label, cv2.CC_STAT_HEIGHT]
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            disc_center = (cx, cy)
            disc_radius = int(max(w, h) / 2 + 15)
            cv2.circle(disc_mask, disc_center, disc_radius, 255, -1)
        else:
            cv2.circle(disc_mask, disc_center, disc_radius, 255, -1)

        return disc_mask, disc_center, disc_radius

    def detect_lesions(self, green_img, clahe_img, vessel_mask, disc_mask, fov_mask):
        """
        Detects Diabetic Retinopathy lesions:
        1. Microaneurysms / Hemorrhages (MA): Dark spots in green channel.
        2. Hard Exudates (EX): Bright yellowish/white lipid deposits.
        """
        h, w = green_img.shape

        non_disc_fov = cv2.bitwise_and(fov_mask, cv2.bitwise_not(disc_mask))
        tissue_roi = cv2.bitwise_and(non_disc_fov, cv2.bitwise_not(vessel_mask))

        # --- A. Exudates Detection (Bright Lesions) ---
        # Exudates appear distinctly bright in green channel outside optic disc
        cand_ex = cv2.bitwise_and((green_img > 185).astype(np.uint8) * 255, tissue_roi)
        num_ex, _, stats_ex, centroids_ex = cv2.connectedComponentsWithStats(cand_ex)
        valid_exudate_blobs = []
        exudate_area = 0

        for i in range(1, num_ex):
            area = stats_ex[i, cv2.CC_STAT_AREA]
            if 2 <= area <= 600:
                x = stats_ex[i, cv2.CC_STAT_LEFT]
                y = stats_ex[i, cv2.CC_STAT_TOP]
                bw = stats_ex[i, cv2.CC_STAT_WIDTH]
                bh = stats_ex[i, cv2.CC_STAT_HEIGHT]
                valid_exudate_blobs.append((x, y, bw, bh, area))
                exudate_area += area

        # --- B. Microaneurysms Detection (Dark Red circular lesions) ---
        inv_green = cv2.bitwise_not(green_img)
        kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat_ma = cv2.morphologyEx(inv_green, cv2.MORPH_TOPHAT, kernel_ma)

        _, ma_binary = cv2.threshold(tophat_ma, 18, 255, cv2.THRESH_BINARY)
        cand_ma = cv2.bitwise_and(ma_binary, tissue_roi)
        num_ma, _, stats_ma, centroids_ma = cv2.connectedComponentsWithStats(cand_ma)

        valid_ma_blobs = []
        for i in range(1, num_ma):
            area = stats_ma[i, cv2.CC_STAT_AREA]
            if 2 <= area <= 75:
                cx = int(centroids_ma[i][0])
                cy = int(centroids_ma[i][1])
                r = int(max(2, np.sqrt(area / np.pi) + 1.5))
                valid_ma_blobs.append((cx, cy, r))

        # Create Lesion Visualization
        lesion_vis = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

        # Draw Exudates in Yellow / Gold boxes
        for (x, y, bw, bh, area) in valid_exudate_blobs:
            cv2.rectangle(lesion_vis, (x - 2, y - 2), (x + bw + 2, y + bh + 2), (0, 220, 255), 1)
            cv2.rectangle(lesion_vis, (x, y), (x + bw, y + bh), (0, 255, 255), -1)

        # Draw Microaneurysms in Vivid Red Circles
        for (cx, cy, cr) in valid_ma_blobs:
            cv2.circle(lesion_vis, (cx, cy), max(3, cr), (0, 0, 255), 2)
            cv2.circle(lesion_vis, (cx, cy), 1, (0, 0, 255), -1)

        # Outline optic disc in subtle cyan for examiner clarity
        disc_c, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if disc_c:
            cv2.drawContours(lesion_vis, disc_c, -1, (255, 180, 0), 1)

        return {
            "lesion_vis": lesion_vis,
            "microaneurysm_count": len(valid_ma_blobs),
            "exudate_count": len(valid_exudate_blobs),
            "exudate_area": int(exudate_area),
            "ma_blobs": valid_ma_blobs,
            "ex_blobs": valid_exudate_blobs
        }

    def extract_features(self, green_img, vessel_mask, lesion_info, fov_mask):
        """
        Computes quantitative DIP features for medical diagnosis and ML classification.
        """
        fov_pixels = int(np.count_nonzero(fov_mask))
        if fov_pixels == 0:
            fov_pixels = 1

        vessel_pixels = int(np.count_nonzero(vessel_mask))
        vessel_density = round((vessel_pixels / fov_pixels) * 100.0, 2)

        ma_count = lesion_info["microaneurysm_count"]
        ex_count = lesion_info["exudate_count"]
        ex_area_pct = round((lesion_info["exudate_area"] / fov_pixels) * 100.0, 3)

        retinal_pixels = green_img[fov_mask > 0]
        mean_intensity = round(float(np.mean(retinal_pixels)), 2) if len(retinal_pixels) > 0 else 0.0
        contrast_std = round(float(np.std(retinal_pixels)), 2) if len(retinal_pixels) > 0 else 0.0

        sobelx = cv2.Sobel(vessel_mask, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(vessel_mask, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        vessel_tortuosity = round(float(np.mean(grad_mag[vessel_mask > 0])) if vessel_pixels > 0 else 0.0, 2)

        return {
            "vessel_density": vessel_density,
            "microaneurysm_count": ma_count,
            "exudate_count": ex_count,
            "exudate_area_pct": ex_area_pct,
            "mean_intensity": mean_intensity,
            "contrast_std": contrast_std,
            "vessel_tortuosity": vessel_tortuosity,
            "total_lesions": ma_count + ex_count
        }

    def create_result_overlay(self, original_bgr, vessel_mask, lesion_info):
        """
        Creates composite ophthalmology overlay:
        - Segmented vessels in neon cyan
        - Microaneurysms in vivid red
        - Exudates in bright yellow
        """
        overlay = original_bgr.copy()

        vessel_indices = vessel_mask > 0
        overlay[vessel_indices] = cv2.addWeighted(
            overlay[vessel_indices], 0.35,
            np.full_like(overlay[vessel_indices], (240, 210, 30)), 0.65, 0
        )

        for (x, y, bw, bh, _) in lesion_info["ex_blobs"]:
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 235, 255), 2)

        for (cx, cy, cr) in lesion_info["ma_blobs"]:
            cv2.circle(overlay, (cx, cy), max(3, cr), (0, 0, 255), 2)

        return overlay

    def process(self, img_bgr, output_dir=None):
        """
        Executes full step-by-step DIP pipeline, saving all intermediate images.
        """
        original_resized = cv2.resize(img_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        fov_mask = self.get_retinal_mask(original_resized)
        gray_img = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)
        green_img = self.extract_green_channel(original_resized)
        clahe_img = self.apply_clahe(green_img)
        denoised_img = self.denoise(clahe_img)
        vessel_mask = self.segment_blood_vessels(denoised_img, fov_mask)
        disc_mask, disc_center, disc_radius = self.detect_optic_disc(green_img, fov_mask)

        disc_masked_view = clahe_img.copy()
        cv2.circle(disc_masked_view, disc_center, disc_radius, 0, -1)

        lesion_info = self.detect_lesions(green_img, clahe_img, vessel_mask, disc_mask, fov_mask)
        lesion_vis = lesion_info["lesion_vis"]

        features = self.extract_features(green_img, vessel_mask, lesion_info, fov_mask)
        result_overlay = self.create_result_overlay(original_resized, vessel_mask, lesion_info)

        saved_paths = {}
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            image_map = {
                "original": original_resized,
                "gray": gray_img,
                "green": green_img,
                "clahe": clahe_img,
                "denoised": denoised_img,
                "vessels": vessel_mask,
                "optic_disc": disc_masked_view,
                "lesions": lesion_vis,
                "overlay": result_overlay
            }
            for key, img in image_map.items():
                file_path = os.path.join(output_dir, f"{key}.jpg")
                cv2.imwrite(file_path, img)
                saved_paths[key] = f"{key}.jpg"

        return {
            "features": features,
            "saved_images": saved_paths,
            "optic_disc": {
                "center": disc_center,
                "radius": disc_radius
            }
        }
