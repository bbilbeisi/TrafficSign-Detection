#!/usr/bin/env python3
"""
Traffic Sign Detection & Classification Pipeline
CLI script to detect and classify traffic signs from images or folders.
"""

import argparse
import os
import sys
import cv2
import numpy as np
from tensorflow import keras

# Label mapping
LABELS = {
    0: "Stop",
    1: "Do not Enter",
    2: "Traffic jam is close",
    3: "Yield"
}

def load_traffic_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return keras.models.load_model(model_path)

def preprocess_image(img, experiment=1):
    """
    Apply illumination enhancement, spatial blur, and red-color thresholding.
    experiment: 1 (Median + CLAHE), 2 (Gaussian + Linear), 3 (Bilateral + CLAHE)
    """
    # 1. Contrast Enhancement
    if experiment in (1, 3):
        # CLAHE in YUV space
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        img = cv2.addWeighted(img, 0.5, enhanced, 0.5, 0)
    else:
        # Linear Contrast Scaling
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=0)

    # 2. Gamma Shadow Correction
    inv_gamma = 1.0 / 1.3
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    img = cv2.LUT(img, table)

    # 3. Saturation Boost
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.6, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    # 4. Spatial Filtering
    if experiment == 1:
        filtered = cv2.medianBlur(img, 5)
    elif experiment == 2:
        filtered = cv2.GaussianBlur(img, (5, 5), 0)
    else:
        filtered = cv2.bilateralFilter(img, 9, 75, 75)

    # 5. Dual-band HSV Red Segmentation
    hsv_f = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    if experiment == 2:
        mask1 = cv2.inRange(hsv_f, np.array([0, 120, 70]), np.array([5, 255, 255]))
        mask2 = cv2.inRange(hsv_f, np.array([170, 120, 70]), np.array([179, 255, 255]))
    elif experiment == 3:
        mask1 = cv2.inRange(hsv_f, np.array([0, 120, 70]), np.array([3, 255, 255]))
        mask2 = cv2.inRange(hsv_f, np.array([173, 120, 70]), np.array([179, 255, 255]))
    else:
        mask1 = cv2.inRange(hsv_f, np.array([0, 120, 70]), np.array([4, 255, 255]))
        mask2 = cv2.inRange(hsv_f, np.array([167, 120, 70]), np.array([179, 255, 255]))

    red_mask = cv2.bitwise_or(mask1, mask2)

    # 6. Binary Thresholding + Morphology
    _, thresh = cv2.threshold(red_mask, 155, 255, cv2.THRESH_BINARY)
    kernel = np.ones((11, 11), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return opened

def classify_crop(model, crop, confidence_threshold=0.85):
    """Normalize crop and predict traffic sign."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    norm = ((resized - 89.774) / 70.852).reshape(1, 28, 28, 1)

    probs = model.predict(norm, verbose=0)[0]
    best_idx = np.argmax(probs)
    confidence = float(probs[best_idx])

    is_valid = confidence >= confidence_threshold
    label = LABELS[best_idx] if is_valid else "Unknown"
    return label, confidence, is_valid

def process_file(image_path, model, experiment=1, confidence=0.85, output_dir=None, display=False):
    """Run detection and classification on a single image file."""
    original = cv2.imread(image_path)
    if original is None:
        print(f"[Warning] Failed to read image: {image_path}")
        return None

    frame = cv2.resize(original, (500, 400))
    annotated = frame.copy()

    # Preprocessing
    processed = preprocess_image(frame, experiment=experiment)

    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detection_result = {
        "file": os.path.basename(image_path),
        "detected": False,
        "label": None,
        "confidence": 0.0
    }

    if contours:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        crop = frame[y:y+h, x:x+w]

        if crop.size > 0:
            label, conf, is_valid = classify_crop(model, crop, confidence_threshold=confidence)
            detection_result["label"] = label
            detection_result["confidence"] = conf

            if is_valid:
                detection_result["detected"] = True
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = f"{label} ({conf*100:.0f}%)"
                cv2.putText(annotated, text, (x, max(20, y-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"[+] {detection_result['file']} -> {label} ({conf*100:.1f}%)")
            else:
                print(f"[-] {detection_result['file']} -> Ignored (Best guess: {label} at {conf*100:.1f}%)")
        else:
            print(f"[-] {detection_result['file']} -> Invalid crop")
    else:
        print(f"[-] {detection_result['file']} -> No red contours detected")

    # Draw an instruction banner on the window
    cv2.rectangle(annotated, (0, 365), (500, 400), (30, 30, 30), -1)
    cv2.putText(annotated, "[SPACE/ENTER] Next   |   [Q/ESC] Quit", (40, 388),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Save output if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, annotated)

    # Display window (enabled by default)
    user_quit = False
    if display:
        window_title = f"Traffic Sign Detection - {os.path.basename(image_path)}"
        cv2.imshow(window_title, annotated)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_title)
        
        # 'q' or 'Q' or ESC (27) to cleanly exit
        if key in (ord('q'), ord('Q'), 27):
            user_quit = True

    return detection_result, user_quit

def main():
    parser = argparse.ArgumentParser(description="Traffic Sign Detection and Classification")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, default="images", help="Path to a folder of images (default: images)")
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3], default=1,
                        help="Experiment preset: 1 (Median+CLAHE), 2 (Gaussian+Linear), 3 (Bilateral+CLAHE)")
    parser.add_argument("--confidence", type=float, default=0.85,
                        help="Confidence cutoff threshold (0.0 to 1.0, default 0.85)")
    parser.add_argument("--save", action="store_true", help="Save annotated results to an 'outputs/' directory")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without showing GUI windows")
    parser.add_argument("--model", type=str, default="DetectionModel.h5", help="Path to .h5 model file")

    args = parser.parse_args()
    display = not args.no_display

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model) if not os.path.isabs(args.model) else args.model

    print(f"Loading model: {model_path}")
    model = load_traffic_model(model_path)
    output_dir = os.path.join(script_dir, "outputs") if args.save else None

    if display:
        print("\n Controls in Image Window:")
        print("  - Press [SPACE] or [ENTER] to view the next image")
        print("  - Press [Q] or [ESC] to quit anytime\n")

    if args.image:
        process_file(args.image, model, experiment=args.experiment,
                     confidence=args.confidence, output_dir=output_dir, display=display)
    else:
        folder_path = os.path.join(script_dir, args.folder) if not os.path.isabs(args.folder) else args.folder
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' does not exist.")
            sys.exit(1)

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}
        images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]

        print(f"Processing {len(images)} images in '{folder_path}' (Press Q to quit)...")
        for img_name in sorted(images):
            img_path = os.path.join(folder_path, img_name)
            res, should_quit = process_file(img_path, model, experiment=args.experiment,
                                            confidence=args.confidence, output_dir=output_dir, display=display)
            if should_quit:
                print("\n[!] Exited by user.")
                break

    cv2.destroyAllWindows()
    if args.save:
        print(f"\nAll annotated images saved to: {output_dir}")

if __name__ == "__main__":
    main()
