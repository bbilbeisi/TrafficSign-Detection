import cv2
import numpy as np
import os
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(SCRIPT_DIR, 'DetectionModel.h5')
model = keras.models.load_model(model_path) # 'DetectionModel.h5': pre-trained CNN model

image_path = os.path.join(SCRIPT_DIR, "images")
ImageNamePath = [f for f in os.listdir(image_path) if not f.startswith('.')] # list of images

def readImage(imageFilePath):
    img = cv2.imread(imageFilePath, 1) # read images. '1': 3-channel BGR
    img = cv2.resize(img, (500, 400)) # resize image
    return img

# CONTRAST ENHANCEMENT (LINEAR Contrast Enhancement)
def histogramEqualization(img):
    alpha = 1.2  # Contrast control
    beta = 0     # Brightness control
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def increaseSaturation(img, saturation=1.5):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	s = np.clip(s * saturation, 0, 255).astype(np.uint8)
	hsv = cv2.merge([h,s,v])
	saturated_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	return saturated_img

def brighten_dark_areas(img, gamma=1.5):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    brightened_img = cv2.LUT(img, table)
    return brightened_img

# FILTERING (Spatial/Frequency Domain): Gaussian Blur Filtering
def filteringImages(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def returnRedness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define range for red color in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([5, 255, 255]) # '5' increased orange inclusion
    lower_red2 = np.array([170, 120, 70]) # '170' lower HSV color range, includes darker shades of red
    upper_red2 = np.array([179, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    return mask

# BINARY OBJECT DETECTION: Thresholding and contour detection
def threshold(img,T=150):
	_,img=cv2.threshold(img,T,255,cv2.THRESH_BINARY)
	return img 

def morphology(img,kernelSize=7):
	kernel = np.ones((kernelSize,kernelSize),np.uint8)
	opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	return opening

def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def findContour(img):
	contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours

def findBiggestContour(contours):
	c=[cv2.contourArea(i) for i in contours]
	return contours[c.index(max(c))]

def extractSign(img, contour):
    x, y, w, h = cv2.boundingRect(contour)
    sign = img[y:y+h, x:x+w]
    return sign, (x, y, w, h)

def drawDetection(img, bbox, label, conf):
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    text = f"{label} ({conf*100:.0f}%)"
    cv2.putText(img, text, (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img

def preprocessingImageToClassifier(image=None,imageSize=28,mu=89.77428691773054,std=70.85156431910688):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (imageSize, imageSize))
    image = (image - mu) / std
    image = image.reshape(1, imageSize, imageSize, 1)
    return image

def predict4(sign, confidence_threshold=0.85):
    img = preprocessingImageToClassifier(sign, imageSize=28)
    probabilities = model.predict(img, verbose=0)[0]
    best_class = np.argmax(probabilities)
    confidence = probabilities[best_class]
    
    is_valid = confidence >= confidence_threshold
    sign_name = label_identifier[best_class] if is_valid else "Unknown"
    return sign_name, confidence, is_valid

label_identifier={0:"Stop",
                  1:"Do not Enter",
                  2:"Traffic jam is close",
                  3:"Yield"}

# -----------------------------------------------------

for i in ImageNamePath:
    fullImagePath = os.path.join(image_path, i)
    testCase = readImage(fullImagePath)
    img = np.copy(testCase)
    img = histogramEqualization(img)
    img = brighten_dark_areas(img, 1.3)
    img = increaseSaturation(img, saturation=1.6)

    img = filteringImages(img)
    img = returnRedness(img)
    img = threshold(img, T=155)
    img = morphology(img, 11)
    contours = findContour(img)
    
    if contours:
        big = findBiggestContour(contours)
        sign, bbox = extractSign(testCase, big)
        
        if sign.size > 0:
            sign_name, conf, is_valid = predict4(sign, confidence_threshold=0.85)
            if is_valid:
                testCase = drawDetection(testCase, bbox, sign_name, conf)
                print(f"File: {i} | Detected: {sign_name} (Confidence: {conf*100:.1f}%)")
            else:
                print(f"File: {i} | Ignored: Below confidence threshold (Best match: {sign_name} at {conf*100:.1f}%)")
        else:
            print(f"File: {i} | Ignored: Empty crop")
    else:
        print(f"File: {i} | No red contours found")

    show(testCase)

# --------------------------------------

# RESULTS: (Biggest Contour)
# Correct: 53 | Partially Correct: 1| No Contours found: 0 | Incorrect: 1

# RESULTS: (Classification)
# Correct: 51 | Incorrect: 4

# --------------------------------------
