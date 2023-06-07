import cv2
import numpy as np

# Load the pre-trained face cascade from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

# Additional code starts here

# Define a function to count the number of faces
def count_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Return the number of faces detected
    return len(faces)

# Load an image from file
image = cv2.imread('example_image.jpg')

# Count the number of faces in the image
num_faces = count_faces(image)

# Print the result
print(f"Number of faces detected: {num_faces}")

# Apply a Gaussian blur to the image
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

# Display the blurred image
cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the blurred image to file
cv2.imwrite('blurred_image.jpg', blurred_image)

# Flip the image horizontally
flipped_image = cv2.flip(image, 1)

# Display the flipped image
cv2.imshow('Flipped Image', flipped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Resize the image
resized_image = cv2.resize(image, (500, 500))

# Display the resized image
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection to the grayscale image
edges = cv2.Canny(gray_image, 100, 200)

# Display the edge-detected image
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply a color map to the grayscale image
color_mapped_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

# Display the color-mapped image
cv2.imshow('Color Mapped Image', color_mapped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Extract the hue channel
hue_channel = hsv_image[:,:,0]

# Display the hue channel as a grayscale image
cv2.imshow('Hue Channel', hue_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract the saturation channel
saturation_channel = hsv_image[:,:,1]

# Display the saturation channel as a grayscale image
cv2.imshow('Saturation Channel', saturation_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract the value channel
value_channel = hsv_image[:,:,2]

# Display the value channel as a grayscale image
cv2.imshow('Value Channel', value_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a blank image of the same size as the original image
blank_image = np.zeros_like(image)

# Draw a rectangle on the blank image
cv2.rectangle(blank_image, (100, 100), (400, 400), (0, 255, 0), 3)

# Display the image with the rectangle
cv2.imshow('Rectangle', blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a blank image of the same size as the original image
blank_image = np.zeros_like(image)

# Draw a circle on the blank image
cv2.circle(blank_image, (250, 250), 150, (0, 0, 255), 3)

# Display the image with the circle
cv2.imshow('Circle', blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a blank image of the same size as the original image
blank_image = np.zeros_like(image)

# Draw a line on the blank image
cv2.line(blank_image, (100, 100), (400, 400), (255, 0, 0), 3)

# Display the image with the line
cv2.imshow('Line', blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to the grayscale image
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Display the binary image
cv2.imshow('Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply adaptive thresholding to the grayscale image
adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Display the adaptive threshold image
cv2.imshow('Adaptive Threshold', adaptive_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)

# Apply Otsu's thresholding to the blurred image
_, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the threshold image
cv2.imshow('Threshold Image', threshold_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply morphological operations to the grayscale image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated_image = cv2.dilate(gray_image, kernel, iterations=1)
eroded_image = cv2.erode(gray_image, kernel, iterations=1)

# Display the dilated and eroded images
cv2.imshow('Dilated Image', dilated_image)
cv2.waitKey(0)
cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
