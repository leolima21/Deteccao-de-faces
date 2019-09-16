import cv2

# Leitura do arquivo haar para criar um classificador
classificadorFace = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Captura do video
video = cv2.VideoCapture(0)

# Exibicao das imgs capturadas em forma de video
while True:
	conectado, frame = video.read()
	# print(conectado)
	# print(frame)

	# Converssao da imagem para escala de cinza
	frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Deteccao das faces na imagem 
	# minSize = menor obj a ser reconhecido. 30 x 30 eh o padrao
	facesDetectadas = classificadorFace.detectMultiScale(frameCinza, minSize=(50, 50))
	
	# Desenho do retangulo para cada face detectada
	for (x, y, largura, altura) in facesDetectadas:	
		# Desenho do retangulo. No final cor e largura da borda
		cv2.rectangle(frame, (x,y), (x + largura, y + largura), (0, 0 , 255), 2)

	cv2.imshow('Video', frame)

	# Interromper a exibicao
	if cv2.waitKey(1) == ord('q'):
		break

# Limpeza de memoria
video.release()
cv2.destroyAllWindows()