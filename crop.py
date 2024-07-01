from ultralytics import YOLO
import os
import cv2
import numpy as np
import wordninja
import time

def plot_boxes(img_path, output_path):
    # Load YOLO model
    model_path = os.path.join('.','train4', 'train4', 'weights', 'last.pt')
    model = YOLO(model_path)

    # Predict bounding boxes on the original image
    results = model.predict(img_path)

    # Read the image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a directory to save cropped images
    os.makedirs(output_path, exist_ok=True)

    # Initialize a dictionary to store coordinates of cropped regions for each line
    line_boxes = {}

    # Define a threshold for determining line breaks based on y-coordinate differences
    y_threshold = 100  # Adjust as needed

    # Initialize a list to store class names in sorted order
    sorted_class_names = {}

    # Iterate through the bounding boxes and separate them into lines
    for i, result in enumerate(results):
        boxes = results[i].boxes.xyxy.cpu().numpy()
        classes = results[i].boxes.cls.cpu().numpy()  # Extract class labels from yolo

        # Combine boxes and class labels into a single list
        combined = list(zip(boxes, classes))
        
        # Sort combined list based on the y-coordinate of the bounding boxes
        combined = sorted(combined, key=lambda x: x[0][1])

        # Separate sorted boxes and classes
        boxes, classes = zip(*combined)

        num_lines = 1
        prev_y = None
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            class_id = int(classes[j])  # Get class ID
            class_name = model.names[class_id]  # Get class name from model

            if prev_y is not None and abs(y1 - prev_y) > y_threshold:
                num_lines += 1
            
            prev_y = y1

            # For new line sorting
            if num_lines not in line_boxes:
                line_boxes[num_lines] = []
                sorted_class_names[num_lines] = []

            # Check for overlap with previously cropped regions
            overlap = False
            for cropped_box in line_boxes[num_lines]:
                cropped_x1, cropped_y1, cropped_x2, cropped_y2 = cropped_box

                # Check if the boxes overlap
                if not (x2 < cropped_x1 or x1 > cropped_x2 or y2 < cropped_y1 or y1 > cropped_y2):
                    overlap = True
                    break
            
            if not overlap:
                line_boxes[num_lines].append((x1, y1, x2, y2))
                sorted_class_names[num_lines].append(class_name)  # Save the class name in sorted order
                pathh = "./yolotest/"
                save_path11 = os.path.join(pathh, "actual_coloredOO.jpg")
                appc = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 25), 2)
                cv2.putText(image, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 25), 2)
                cv2.imwrite(save_path11, appc)

    # Initialize a list to store whether each boundary box is close to the next one
    proximity_list = []

    # Sort bounding boxes within each line based on their x-coordinates
    for line_num, boxes in line_boxes.items():
        # Combine boxes and class names into a single list
        combined = list(zip(boxes, sorted_class_names[line_num]))
        
        # Sort combined list based on the x-coordinate of the bounding boxes
        combined = sorted(combined, key=lambda x: x[0][0])

        # Separate sorted boxes and class names
        boxes, classes = zip(*combined)
        
        # Update the sorted class names in the dictionary
        sorted_class_names[line_num] = list(classes)

        # Calculate the mean distance between boundary boxes
        mean_distance = np.mean([boxes[i+1][0] - boxes[i][0] for i in range(len(boxes) - 1)])

        # Define a threshold for determining proximity based on the mean distance
        threshold = mean_distance * 0.25  # Adjust as needed

        for j in range(len(boxes)):
            if j < len(boxes) - 1:
                x1_curr, _, x2_curr, _ = boxes[j]
                x1_next, _, _, _ = boxes[j + 1]
                x_diff = abs(x1_next - x2_curr)

                if x_diff <= threshold:
                    proximity_list.append(1)  # Append 1 if boxes are far
                else:
                    proximity_list.append(0)  # Append 0 if there's a close

        proximity_list.append(1)  # Append 0 when a new line begins

        # Save cropped images for each line based on proximity
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            save_path = os.path.join(output_path, f"line_{line_num}segment{i + 1}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(image[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_RGB2BGR))

    # Flatten the sorted class names dictionary to a list
    sorted_class_names_flat = [name for line in sorted_class_names.values() for name in line]

    return proximity_list, sorted_class_names_flat

def add_spaces_to_string(input_string):
    words = wordninja.split(input_string)
    result_string = " ".join(words)
    return result_string