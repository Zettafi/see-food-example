---
title: SeeFood (Not Hotdog)
emoji: 🌭
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 3.35.2
app_file: app.py
pinned: false
license: mpl-2.0
models: [resnet-18]
tags: [resnet-18, hotdogs]
---

# Seefood (Not Hotdog)

An example creating the SeeFood app from the Silicon Valley TV show.

The first steps to run are to create a Python virtual environment and installing the needed requirements:

```
python -m venv venv
pip install -r requirements.txt
```

## Training the model

In order to train the model from scratch you can first run:

```
python create_model.py
```

## Running the App

In order to start the app you can run:
```
python app.py
```

And it will launch the application at http://127.0.0.1:7860/ or the 
first port that is accessible. So please check the output from the command.
