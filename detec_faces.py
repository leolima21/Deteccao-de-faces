import cv2

# Leitura do arquivo haar para criar um classificador
classificador = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Leitura da imagem que sera usada
imagem = cv2.imread('pessoas/pessoas4.jpg')
# Conversao para tons de cinza
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Deteccao de faces na imagem carregada 
# Usando o parametro scalefactor(>1). 1.1 eh o padrao = escala da detecao
# minNeighbors = Quantos vizinhos cada retangulo candidato deve ter para mante-lo(valor alto = menos deteccoes e maior qualidade)
# minSize = menor obj a ser reconhecido. 30 x 30 eh o padrao
facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=9, minSize=(30,30))

# Print do numero de faces detectadas pelo algoritmo
print(len(facesDetectadas))

# Print da regiao da faces detectadas
print(facesDetectadas)

# Desenhar o quadrado nas faces detectadas. Comparar com o print anterior
for (x, y, largura, altura) in facesDetectadas:
	print(x, y, largura, altura)
	# Desenho do retangulo. No final cor e largura da borda
	cv2.rectangle(imagem, (x,y), (x + largura, y + largura), (0, 0 , 255), 2)

# Mostrar a imagem na janela para o usuario
cv2.imshow('Faces encontradas', imagem)

cv2.waitKey()