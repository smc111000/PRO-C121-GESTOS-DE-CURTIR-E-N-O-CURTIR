import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    finger_fold_status = []

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            for finger_tip in finger_tips:
                # Obtenha as posições x e y das pontas dos dedos
                finger_tip_x = int(lm_list[finger_tip].x * w)
                finger_tip_y = int(lm_list[finger_tip].y * h)

                # Desenhe círculos azuis nas pontas dos dedos
                cv2.circle(img, (finger_tip_x, finger_tip_y), 10, (255, 0, 0), cv2.FILLED)

                # Verifique se o dedo está dobrado
                if finger_tip_x < lm_list[thumb_tip].x * w:
                    # Desenhe círculos verdes nas pontas dos dedos
                    cv2.circle(img, (finger_tip_x, finger_tip_y), 10, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            # Verifique se todos os dedos estão dobrados
            if all(finger_fold_status):
                # Verifique se o polegar está levantado para cima ou para baixo
                if lm_list[thumb_tip].y * h < lm_list[thumb_tip - 2].y * h:
                    # Imprima "CURTI" e mostre o texto "CURTI" em verde na imagem
                    print("CURTI")
                    cv2.putText(img, "CURTI", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    # Imprima "NÃO CURTI" e mostre o texto "NÃO CURTI" em vermelho na imagem
                    print("NÃO CURTI")
                    cv2.putText(img, "NÃO CURTI", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("detector de maos", img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

