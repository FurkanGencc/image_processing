import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

mpDraw = mp.solutions.drawing_utils

# göz kapalı süre takibi
eye_closed_start = None
ALERT_TIME = 2
status = "GOZ ACIK"


# EAR fonksiyonu 
def eye_aspect_ratio(landmarks, eye_points):
    # dikey
    A = landmarks[eye_points[1]]
    B = landmarks[eye_points[5]]
    C = landmarks[eye_points[2]]
    D = landmarks[eye_points[4]]

    vertical1 = abs(A.y - B.y)
    vertical2 = abs(C.y - D.y)

    vertical = (vertical1 + vertical2) / 2

    # yatay
    E = landmarks[eye_points[0]]
    F = landmarks[eye_points[3]]

    horizontal = abs(E.x - F.x)

    ear = vertical / horizontal
    return ear


# göz landmark indexleri
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:

            # EAR hesapla
            leftEAR = eye_aspect_ratio(faceLms.landmark, LEFT_EYE)
            rightEAR = eye_aspect_ratio(faceLms.landmark, RIGHT_EYE)

            ear = (leftEAR + rightEAR) / 2

            # eşik (kritik değer)
            if ear < 0.20:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                else:
                    duration = time.time() - eye_closed_start

                    if duration > ALERT_TIME:
                        status = "UYARI: UYKU HALI!"
                    else:
                        status = "GOZ KAPALI"
            else:
                eye_closed_start = None
                status = "GOZ ACIK"

            # yüz çizimi
            mpDraw.draw_landmarks(
                img,
                faceLms,
                mpFaceMesh.FACEMESH_TESSELATION
            )

    # ekrana yazı
    cv2.putText(img, status, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 3)

    cv2.imshow("EAR Eye Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty("EAR Eye Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()