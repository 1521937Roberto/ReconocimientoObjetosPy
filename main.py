import torch
import cv2
import matplotlib.pyplot as plt

# Cargar el modelo preentrenado de YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 

# Función para detectar objetos en una lista de imágenes
def detectar_objetos_imagenes(rutas_imagenes):
    for ruta_imagen in rutas_imagenes:
        img = cv2.imread(ruta_imagen)
        if img is None:
            print(f"Error: No se pudo cargar la imagen en {ruta_imagen}. Verifica la ruta.")
            continue  # Pasar a la siguiente imagen si no se carga correctamente
        
        resultados = model(img)
        resultados_img = resultados.render()[0]
        img_rgb = cv2.cvtColor(resultados_img, cv2.COLOR_BGR2RGB)

        # Mostrar la imagen con detecciones
        plt.imshow(img_rgb)
        plt.title(f"Detecciones en: {ruta_imagen}")
        plt.axis('off')
        plt.show()

# Función para detectar objetos en tiempo real usando la cámara
def detectar_objetos_video():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo acceder a la cámara.")
            break

        # Realizar detección de objetos
        resultados = model(frame)
        resultados_frame = resultados.render()[0]
        
        cv2.imshow('Detección de Objetos - Cámara', resultados_frame)
        
        # Presiona 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Lista de rutas de imágenes que deseas procesar
rutas_imagenes = [
    'D:/PythonReconocimientoObjetos/Images/andrey-matveev-nIvfHUTSKFI-unsplash.jpg',
    'D:/PythonReconocimientoObjetos/Images/descarga.jpg'
]

# Detectar objetos en cada imagen de la lista
detectar_objetos_imagenes(rutas_imagenes)

# Detectar objetos en tiempo real usando la cámara
# Se ejecutará después de la detección en imágenes
detectar_objetos_video()
