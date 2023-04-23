# import gradio as gr
# def greet(name):
#     return "Hello" + name + "!!"
# iface = gr.Interface(fn=greet, inputs="text", outputs="text")
# iface.launch()

import gradio as gr
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")

def predict(image):
    predictions = pipeline(image)
    return {p["label"]: p["score"] for p in predictions}

gr.Interface(
    predict,
    inputs=gr.inputs.Image(label="Upload hot dog candidate", type="filepath"),
    outputs=gr.outputs.Label(num_top_classes=2),
    title="Hot Dog? Or Not?",
).launch()。
