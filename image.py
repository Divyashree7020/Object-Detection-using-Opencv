import cv2

# Load class names for COCO dataset (80 object classes)
class_names = []
with open('coco.names', 'rt') as f:
    class_names = f.read().strip().split('\n')

# Paths to the configuration and model files
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

# Load the pre-trained model and configuration
net = cv2.dnn_DetectionModel(frozen_model, config_file)
net.setInputSize(320, 320)  # Set input size for the model (320x320)
net.setInputScale(1.0 / 127.5)  # Normalize the input image
net.setInputMean((127.5, 127.5, 127.5))  # Mean subtraction for normalization
net.setInputSwapRB(True)  # Swap the channels for BGR to RGB conversion

# Load the image
img = cv2.imread('man-tuxedo-car-style-wallpaper-preview.jpg')  # Provide the path to your image file

# Perform object detection on the image
class_ids, confidences, boxes = net.detect(img, confThreshold=0.5)

# If there are any detections, draw bounding boxes and labels
if len(class_ids) > 0:
    for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
        label = f'{class_names[class_id - 1]}: {confidence:.2f}'  # Object class and confidence score
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)  # Draw bounding box
        cv2.putText(img, label, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Show the image with detected objects
cv2.imshow("Object Detection", img)

# Wait for any key to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
