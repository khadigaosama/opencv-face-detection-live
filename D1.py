import cv2
import os
import urllib.request

# Directory for models
model_dir = "/Users/osamaahmed/Desktop/CV/dnn_models"
os.makedirs(model_dir, exist_ok=True)

# Model and config file paths
modelFile = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
configFile = os.path.join(model_dir, "deploy.prototxt")

# URLs for download
modelURL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
configURL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"

# Expected sizes (approx)
expected_model_size = 5000000  # ~5MB
expected_config_size = 2000    # ~2KB

def download_if_missing(file_path, url, expected_size):
    if not os.path.exists(file_path) or os.path.getsize(file_path) < expected_size:
        print(f"Downloading {os.path.basename(file_path)}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded {file_path}")
    else:
        print(f"{os.path.basename(file_path)} already exists and is valid.")

# Ensure both files are present and valid
download_if_missing(modelFile, modelURL, expected_model_size)
download_if_missing(configFile, configURL, expected_config_size)

# Load the model
print("Loading DNN model...")
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Prepare input for the network
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{confidence*100:.1f}%"
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show output
    cv2.imshow('Webcam Live - Face Detection (DNN)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
