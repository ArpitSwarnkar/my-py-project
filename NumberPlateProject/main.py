# VEHICLE NUMBER PLATE DETECTION

import cv2
import numpy as np
import easyocr
import re
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: Load the Image
# -----------------------------
image_path = 'car.jpg'   # 🔸 Change filename if needed
img = cv2.imread(image_path)

if img is None:
    print("❌ Error: Could not load image. Make sure the file exists.")
    exit()

# Resize large images for faster processing
h, w, _ = img.shape
if w > 1200:
    img = cv2.resize(img, (1200, int(h * 1200 / w)))

# -----------------------------
# STEP 2: Preprocess (Gray + Blur + Edge)
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)   # Reduce noise
edged = cv2.Canny(bfilter, 30, 200)               # Detect edges

# -----------------------------
# STEP 3: Find possible number plate contour
# -----------------------------
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = keypoints[0] if len(keypoints) == 2 else keypoints[1]
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:   # Plate is likely rectangular
        location = approx
        break

if location is None:
    print("⚠️ Could not detect number plate region.")
    exit()

# -----------------------------
# STEP 4: Crop the plate region
# -----------------------------
mask = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask, [location], 0, 255, -1)
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))

# Add small padding
padding = 10
topx = max(0, topx - padding)
topy = max(0, topy - padding)
bottomx = min(gray.shape[0], bottomx + padding)
bottomy = min(gray.shape[1], bottomy + padding)

cropped_image = gray[topx:bottomx, topy:bottomy]

cv2.imwrite("detected_plate.jpg", cropped_image)

# -----------------------------
# Adaptive threshold for clarity
cropped_image = cv2.adaptiveThreshold(
    cropped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 31, 2
)

# Invert if plate background is dark
if np.mean(cropped_image) < 127:
    cropped_image = cv2.bitwise_not(cropped_image)

# -----------------------------
# STEP 6: OCR Detection
# -----------------------------
print("🔍 Reading text from number plate...")
reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext(cropped_image, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

if not result:
    print("⚠️ OCR could not detect readable text. Try a clearer image.")
    exit()

# Combine all detected strings
text_detected = " ".join([res[1] for res in result])
clean_text = re.sub(r'[^A-Z0-9]', '', text_detected.upper())

print(f"✅ Detected Number Plate Text: {clean_text}")

# -----------------------------
# STEP 7: Identify Indian State (RTO Code)
# -----------------------------
state_codes = {
    "AP":"Andhra Pradesh","AR":"Arunachal Pradesh","AS":"Assam","BR":"Bihar",
    "CG":"Chhattisgarh","CH":"Chandigarh","DL":"Delhi","GA":"Goa","GJ":"Gujarat",
    "HR":"Haryana","HP":"Himachal Pradesh","JH":"Jharkhand","JK":"Jammu and Kashmir",
    "KA":"Karnataka","KL":"Kerala","LD":"Lakshadweep","MP":"Madhya Pradesh",
    "MH":"Maharashtra","MN":"Manipur","ML":"Meghalaya","MZ":"Mizoram",
    "NL":"Nagaland","OD":"Odisha","PB":"Punjab","PY":"Puducherry","RJ":"Rajasthan",
    "SK":"Sikkim","TN":"Tamil Nadu","TS":"Telangana","TR":"Tripura","UP":"Uttar Pradesh",
    "UK":"Uttarakhand","WB":"West Bengal"
}

state_name = state_codes.get(clean_text[:2], "Unknown")
print(f"🌍 State: {state_name}")

# -----------------------------
# STEP 8: Draw Result on Image
# -----------------------------
for (bbox, text, prob) in result:
    (tl, tr, br, bl) = bbox
    tl = tuple(map(int, tl))
    br = tuple(map(int, br))
    cv2.rectangle(img, tl, br, (0,255,0), 2)

cv2.putText(img, clean_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"Plate: {clean_text} | State: {state_name}")
plt.axis('off')
plt.show()

