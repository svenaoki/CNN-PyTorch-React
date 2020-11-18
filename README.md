# Predicting Cats and Dogs using convolutional neural networks

## What this repo is about?
```
├── backend/
│   ├── app.py
│   ├── cnn.py
│   ├── cnn_resnet.py
│   ├── utils.py
│   ├── state_dict_model.pt
│   └── checkpoint_dict_model.pt
└── frontend/
    ├── package.json
    ├── package-lock.json
    ├── src/
    │   ├── App.css
    │   ├── App.js
    │   ├── index.js
    │   └── components/
    │       ├── Upload.js
    │       └── Upload.css
    └── public/
        ├── content.md
        ├── meta.json
        ├── image.png
        ├── image.jpg
        ├── image.jpeg
        └── document.pdf
```

In the backend folder are the python scripts containing two CNNs, one from scratch one from transfer learning, and app.py where the flask application creates and API endpoint to the model.
In the frontend folder is the react js application which allows us to upload picture and run the CNN as per the pictures below.

Click on "Choose file" and upload your cat or dog image

<img src = "/docs/FrontEndtool.png">

Click on "Run CNN" to request probabilities on whether it is a dog or a cat 
(This prediction was created when the model wasn't trained, but neverthelesss I found it amusing and left it)

<img src = "/docs/FrontEndtool2.png">



