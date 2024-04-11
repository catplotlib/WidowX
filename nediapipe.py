import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize the MediaPipe face mesh model
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the video capture device for real-time input
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    success, image = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Convert the image to RGB and process it with the face mesh model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # Extract unique indices from the tuples in FACEMESH_LIPS
    unique_indices = set()
    for pair in mp_face_mesh.FACEMESH_LIPS:
        unique_indices.update(pair)

    # Convert the set to a sorted list to maintain consistency
    unique_indices = sorted(list(unique_indices))

    # Now, unique_indices contains individual integers of mouth landmark indices

    # Then, inside the loop where you process the landmarks:
    if results.multi_face_landmarks:
        for face_index, face_landmarks in enumerate(results.multi_face_landmarks):
            print(f"Face {face_index + 1} Mouth Landmarks 3D Coordinates:")

            for idx in unique_indices:
                landmark = face_landmarks.landmark[idx]
                x, y, z = landmark.x, landmark.y, landmark.z
                print(f"Landmark {idx}: ({x}, {y}, {z})")
    
            # Visualize the landmarks on the image
            mp_drawing.draw_landmarks(
                image, face_landmarks, mp_face_mesh.FACEMESH_LIPS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
            )

    # Display the original image and the one with detected mouth
    cv2.imshow("Real-time Video", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
