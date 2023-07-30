import cv2
import numpy as np

def get_eye_bounding_boxes(image):
    # Convert the image to grayscale.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale image to extract the eyes.
    threshold_image = cv2.threshold(grayscale_image, 70, 255, cv2.THRESH_BINARY_INV)[1]

    # Find the contours in the threshold image.
    contours = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Find the two largest contours, which correspond to the eyes.
    eye_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Calculate the bounding boxes for the eyes.
    eye_bounding_boxes = [cv2.boundingRect(eye_contour) for eye_contour in eye_contours]

    return eye_bounding_boxes

def get_mouth_bounding_box(image):
    # Convert the image to grayscale.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale image to extract the mouth.
    threshold_image = cv2.threshold(grayscale_image, 100, 255, cv2.THRESH_BINARY_INV)[1]

    # Find the contours in the threshold image.
    contours = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Find the largest contour, which corresponds to the mouth.
    mouth_contour = max(contours, key=cv2.contourArea)

    # Calculate the bounding box for the mouth.
    mouth_bounding_box = cv2.boundingRect(mouth_contour)

    return mouth_bounding_box

def main():
    # Load the facial image.
    image = cv2.imread('00018.png')

    # Get the bounding boxes for the eyes and mouth.
    eye_bounding_boxes = get_eye_bounding_boxes(image)
    mouth_bounding_box = get_mouth_bounding_box(image)

    # Print the coordinates of the bounding boxes.
    print(eye_bounding_boxes)
    print(mouth_bounding_box)

if __name__ == '__main__':
    main()
