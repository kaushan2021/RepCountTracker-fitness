from flask import Flask
import cv2
import mediapipe as mp
import numpy as np
# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Squat counter variables
    squat_counter = 0
    squat_stage = None

    # Open the saved video file
    cap = cv2.VideoCapture('input.mp4')

    # Get the default frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Perform pose detection
            results = pose.process(image)

            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Calculate coordinates for hip, knee, and ankle joints
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame_width,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame_height]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame_width,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame_height]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame_width,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame_height]

                # Calculate angles
                knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame_width,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_height]
                hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)

                # Determine the squat stage and visualize
                color = (0, 255, 0) if knee_angle > 160 and hip_angle > 160 else (0, 0, 255)
                if knee_angle < 90 and squat_stage != "down":
                    squat_stage = "down"
                elif knee_angle > 160 and squat_stage == "down":
                    squat_stage = "up"
                    squat_counter += 1

                cv2.putText(image, f'Squats: {squat_counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Knee Angle: {int(knee_angle)}', (int(left_knee[0]), int(left_knee[1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(image, f'Hip Angle: {int(hip_angle)}', (int(left_hip[0]), int(left_hip[1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw pose landmarks with specified colors
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2))

            # Write the frame to the output video
            out.write(image)

    # Release the resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Post-processing complete. Output saved as 'output.mp4'.")
    return 'Hello World'


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle




# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()
