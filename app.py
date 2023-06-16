from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Initialize the OpenCV face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Function to perform face detection on the webcam frames
def detect_faces():
    # Access the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = video_capture.read()

        # Convert the frame to grayscale for face detection
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw bounding boxes around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Yield the frame as a byte array
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
