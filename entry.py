import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


# Initialize ArcFace Model (InsightFace)
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))


# ----- Step 1: Capture reference face -----
print("Capturing reference face. Press 's' to save a face from webcam.")
cap = cv2.VideoCapture(0)
ref_embedding = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("Capture Reference Face (Press 's')", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and faces:
        ref_embedding = faces[0].embedding
        print("[INFO] Reference face embedding saved.")
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()


# ----- Step 2: Live Recognition -----
print("Starting live recognition...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        box = face.bbox.astype(int)
        embedding = face.embedding
        similarity = cosine_similarity(ref_embedding, embedding)

        label = "Unknown"
        if similarity > 0.6:
            label = f"Matched ({similarity:.2f})"
        else:
            label = f"No Match ({similarity:.2f})"

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Live Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
