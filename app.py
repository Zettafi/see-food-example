from fastai.vision.all import *
import gradio as gr

learn = load_learner("hotdogModel.pkl")


def classify_image(image: Image):
    prediction, index, probability = learn.predict(image)
    return "is_hotdog.png" if prediction == "hotdog" else "not_hotdog.png"


examples = ["examples/hotdog.jpg", "examples/burger.jpg", "examples/sandwich.jpg", "examples/dog.jpg",
            "examples/hotdog_dog.jpg", "examples/fancy_hotdog.jpg"]

iface = gr.Interface(fn=classify_image, inputs="image", outputs="image", examples=examples, allow_flagging="never")
iface.launch(inline=False)
