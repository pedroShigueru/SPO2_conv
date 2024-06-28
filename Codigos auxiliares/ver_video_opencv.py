import cv2

video_path = r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\raw\Left\100001-1487003054311-0-1487003073393.mp4"

cap = cv2.VideoCapture(video_path)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(frames)

if not cap.isOpened():
    print("Erro ao abrir o arquivo de vídeo")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while cap.isOpened():
        ret, frame = cap.read() 
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Video', frame_rgb)
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
