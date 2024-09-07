import cv2
import torch

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pre-trained YOLOv5 model

# Load your custom-trained model (if you have one trained specifically for tomato plants)
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/your/custom_model.pt')

# Open the video file
cap = cv2.VideoCapture('tomato2.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Reached the end of the video.")
        break

    # Perform object detection on the current frame
    results = model(frame)

    # Extract bounding boxes and labels
    detected_objects = results.pandas().xyxy[0]  # Results as pandas DataFrame

    tomato_found = False  # Flag to indicate if a tomato plant was found

    # Filter for tomato plant class
    for _, obj in detected_objects.iterrows():
        label = obj['name']  # Get the label for the detected object
        confidence = obj['confidence']
        xmin, ymin, xmax, ymax = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])

        # Check if detected object is a tomato plant (this depends on your custom model or dataset used for training)
        if label == 'tomato plant':
            tomato_found = True  # Set flag if a tomato plant is detected

            # Draw bounding box around the detected tomato plant
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the "Detected" or "Not Detected" text on the video
    if tomato_found:
        cv2.putText(frame, 'Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Not Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame with detections and notification text
    cv2.imshow('Tomato Plant Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display window
cap.release()
cv2.destroyAllWindows()
