import os
import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

# CONFIGURACION
model = tf.keras.models.load_model("modelos/modelo_mnist.keras")

# PROCESAMIENTO DE IMAGENES
def procesar_imagen(imagen):
    if imagen is None:
        return None, None, "No se ha seleccionado ninguna imagen."

    gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY) if len(imagen.shape) == 3 else imagen.copy()
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = []

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        if cv2.contourArea(contorno) >= 20 and h >= 8 and w >= 3:
            cajas.append((x, y, w, h))

    if not cajas:
        return None, None, "No se detectaron digitos."

    # ORDENAR DIGITOS POR FILAS
    altura_media = np.median([h for x, y, w, h in cajas])
    filas = []

    for caja in sorted(cajas, key=lambda c: c[1] + c[3] / 2):
        x, y, w, h = caja
        centro_y = y + h / 2
        encontrada = False

        for fila in filas:
            centro_fila = np.mean([c[1] + c[3] / 2 for c in fila])
            if abs(centro_y - centro_fila) < max(10, altura_media * 0.6):
                fila.append(caja)
                encontrada = True
                break

        if not encontrada:
            filas.append([caja])

    cajas = [c for fila in filas for c in sorted(fila, key=lambda c: c[0])]
    datos, visuales = [], []

    # ADAPTAR DIGITOS A MNIST (28x28)
    for x, y, w, h in cajas:
        recorte = binaria[y:y+h, x:x+w]
        escala = 20 / max(w, h)
        nuevo_w, nuevo_h = max(1, int(w * escala)), max(1, int(h * escala))

        reducido = cv2.resize(recorte, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)
        final = np.zeros((28, 28), dtype=np.uint8)

        inicio_x, inicio_y = (28 - nuevo_w) // 2, (28 - nuevo_h) // 2
        final[inicio_y:inicio_y+nuevo_h, inicio_x:inicio_x+nuevo_w] = reducido

        datos.append(final.astype("float32") / 255.0)
        visuales.append(final)

    datos = np.array(datos).reshape(-1, 28, 28, 1)
    return datos, visuales, cajas

# RECONOCIMIENTO
def reconocer(imagen):
    datos, visuales, cajas = procesar_imagen(imagen)

    if datos is None:
        raise gr.Error(cajas)

    predicciones = model.predict(datos, verbose=0)
    numeros = np.argmax(predicciones, axis=1)
    confianzas = np.max(predicciones, axis=1) * 100
    secuencia = "".join(map(str, numeros))

    imagen_resultado = imagen.copy()
    if len(imagen_resultado.shape) == 2:
        imagen_resultado = cv2.cvtColor(imagen_resultado, cv2.COLOR_GRAY2RGB)

    # MOSTRAR DIGITOS DETECTADOS
    for i, (x, y, w, h) in enumerate(cajas):
        cv2.rectangle(imagen_resultado, (x, y), (x+w, y+h), (30, 180, 80), 2)
        cv2.putText(imagen_resultado, str(numeros[i]), (x, max(y-5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 180, 80), 2)

    tabla = [[i+1, int(numero), f"{confianzas[i]:.2f}%"]
             for i, numero in enumerate(numeros)]

    resumen = (
        f"Numeros reconocidos: {secuencia}\n"
        f"Digitos detectados: {len(numeros)}\n"
        f"Confianza promedio: {np.mean(confianzas):.2f}%"
    )

    return imagen_resultado, secuencia, tabla, resumen

# TEMA CELESTE PASTEL Y BLANCO
css = """
.gradio-container {
    background: #F0F9FF !important;
    font-family: 'Segoe UI', sans-serif;
}

/* ENCABEZADO */
#encabezado {
    background: linear-gradient(135deg, #BAE6FD, #7DD3FC);
    padding: 28px;
    border-radius: 16px;
    color: #164E63;
    text-align: center;
    margin-bottom: 22px;
}

#encabezado h1 {
    color: #164E63;
    font-size: 32px;
    letter-spacing: 3px;
}

#encabezado p {
    color: #155E75;
    font-size: 13px;
    letter-spacing: 1px;
}

/* PANELES */
.panel {
    background: #FFFFFF;
    border: 1px solid #BAE6FD;
    border-radius: 14px;
    padding: 18px;
}

/* RESULTADO */
#resultado {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: #0284C7;
}

/* BOTON */
#boton {
    background: #38BDF8 !important;
    color: #FFFFFF !important;
    border: none;
    border-radius: 9px;
    font-weight: bold;
}

#boton:hover {
    background: #0EA5E9 !important;
}

/* PIE DE PAGINA */
#pie {
    text-align: center;
    color: #64748B;
    font-size: 12px;
    padding: 18px;
}
"""

# INTERFAZ GRADIO
with gr.Blocks(
    title="TecnoForms | Reconocimiento CNN",
    css=css,
    theme=gr.themes.Soft(
        primary_hue="sky",
        secondary_hue="blue",
        neutral_hue="slate"
    )
) as app:

    gr.HTML("""
    <div id="encabezado">
        <h1>TECNOFORMS</h1>
        <p>PLATAFORMA DE RECONOCIMIENTO INTELIGENTE</p>
        <span style="color:#15803D;font-size:12px;">● SISTEMA ACTIVO</span>
    </div>
    """)

    gr.Markdown("## Análisis de documentos")
    gr.Markdown("Carga una imagen para identificar automáticamente los números mediante nuestra red neuronal.")

    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes="panel"):
                gr.Markdown("### 📁 Documento de entrada")
                entrada = gr.Image(
                    label="Seleccionar imagen",
                    type="numpy",
                    height=300
                )
                boton = gr.Button(
                    "🔍 Iniciar reconocimiento",
                    variant="primary",
                    elem_id="boton"
                )

        with gr.Column():
            with gr.Group(elem_classes="panel"):
                gr.Markdown("### 🖼️ Imagen procesada")
                imagen_resultado = gr.Image(
                    label="Detección de dígitos",
                    height=300
                )

    with gr.Group(elem_classes="panel"):
        gr.Markdown("### Resultados del reconocimiento")

        numero_final = gr.Textbox(
            label="Secuencia identificada",
            interactive=False,
            elem_id="resultado"
        )

        detalle = gr.Dataframe(
            headers=["Posición", "Dígito", "Confianza"],
            datatype=["number", "number", "str"],
            label="Detalle de predicciones",
            interactive=False
        )

        resumen = gr.Textbox(
            label="Resumen del análisis",
            interactive=False,
            lines=3
        )

    boton.click(
        fn=reconocer,
        inputs=entrada,
        outputs=[
            imagen_resultado,
            numero_final,
            detalle,
            resumen
        ]
    )

    gr.HTML("""
    <div id="pie">
        TECNOFORMS © 2026 | Sistema de reconocimiento mediante redes neuronales convolucionales
    </div>
    """)

# EJECUTAR APLICACION
if __name__ == "__main__":
    app.launch()


    #ENTRAR