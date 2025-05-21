import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def initialize_camera():
    """Inicializa a captura de vídeo com configurações otimizadas"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def main():
    # Configurações iniciais
    cap = initialize_camera()
    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )
    
    # Classes de A a Z
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z']
    
    try:
        model = load_model('model_libras.h5')
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            continue
            
        # Espelhar o frame para visualização mais intuitiva
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Processamento da mão
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calcular bounding box dinâmico
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                padding = int(max(w, h) * 0.1)  # Padding proporcional
                x_min = max(0, int(min(x_coords)) - padding)
                y_min = max(0, int(min(y_coords)) - padding)
                x_max = min(w, int(max(x_coords)) + padding)
                y_max = min(h, int(max(y_coords)) + padding)
                
                try:
                    # Pré-processamento da imagem
                    hand_img = frame[y_min:y_max, x_min:x_max]
                    resized_img = cv2.resize(hand_img, (224, 224))
                    
                    # Normalização para o modelo
                    normalized_img = (resized_img.astype(np.float32) / 127.5) - 1
                    input_data = np.expand_dims(normalized_img, axis=0)
                    
                    # Predição
                    predictions = model.predict(input_data, verbose=0)
                    class_idx = np.argmax(predictions[0])
                    confidence = np.max(predictions[0])
                    
                    if confidence > 0.7:  # Limiar de confiança
                        # Desenhar resultados
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame, 
                                   f"{classes[class_idx]} ({confidence:.2f})", 
                                   (x_min, y_min - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"Erro no processamento: {e}")
                    continue
        
        # Exibição otimizada para o Codespaces
        cv2.imshow('Tradutor LIBRAS - Pressione Q para sair', frame)
        
        # Finalizar com 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()