# Predicting Cats and Dogs using convolutional neural networks

## What this repo is about?
```
├── backend/
│   ├── app.py
│   ├── cnn.py
│   ├── cnn_resnet.py
│   ├── utils.py
│   ├── state_dict_model.pt
│   ├── checkpoint_dict_model.pt
│   ├── __init__.py
│   └── README.md
└── frontend/
    ├── package.json
    ├── package-lock.json
    ├── src/
    │   ├── components/
    │   │   ├── Upload.js
    │   │   ├── Upload.css
    │   ├── App.css
    │   ├── App.js
    │   └── index.js
    └── public/
        ├── content.md
        ├── meta.json
        ├── image.png
        ├── image.jpg
        ├── image.jpeg
        └── document.pdf
        
In the backend folder are the python scripts containing two CNNs, one from scratch one from transfer learning, and app.py where the flask application creates and API endpoint to the model.

In the frontend folder is the react js application which allows us to upload picture and run the CNN as per the pictures below.

Click on "Choose file" and upload your cats or dog image
<img src = "/docs/FrontEndtool.png">

Click on "Run CNN" to make a prediciton (will change to display probabilties, too)
<img src = "/docs/FrontEndtool2.png">
