from flask import Flask, render_template, request
import face_recognition
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    img1_mostrar, img2_mostrar = None, None

    if request.method == "POST":
        archivo1 = request.files.get("image1")
        archivo2 = request.files.get("image2")

        if archivo1 and archivo2:
            # Cargar imágenes
            img1 = face_recognition.load_image_file(archivo1.stream)
            img2 = face_recognition.load_image_file(archivo2.stream)

            # Extraer codificaciones de rostros
            enc1 = face_recognition.face_encodings(img1)
            enc2 = face_recognition.face_encodings(img2)

            if enc1 and enc2:
                # Comparar la primera cara encontrada en cada imagen
                resultados = face_recognition.compare_faces([enc1[0]], enc2[0])
                distancia = face_recognition.face_distance([enc1[0]], enc2[0])[0]
                similitud = round((1 - distancia) * 100, 2)
                resultado = f"¿Misma persona?: {'Sí ✅' if resultados[0] else 'No ❌'} — Similitud: {similitud}%"
            else:
                resultado = "No se detectaron rostros en una de las imágenes."

            img1_mostrar = imagen_a_base64(archivo1)
            img2_mostrar = imagen_a_base64(archivo2)

    return render_template("index.html", result=resultado, img1=img1_mostrar, img2=img2_mostrar)

def imagen_a_base64(archivo):
    imagen = Image.open(archivo.stream).convert("RGB")
    buffer = io.BytesIO()
    imagen.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
