# Predicting Cats and Dogs using convolutional neural networks

## What this repo is about?
This repo originated from the objective to create a CNN using PyTorch and hosting it on Flask.
But I would hate the idea of creating a frontend in Flask as well plus some akward HTML. Therefore I link the model with a ReactJS application.
People who would like to use the repo should therefore know a little bit of Javascript.

Since this is just a fun project, I do not intend to spend much time on it - maybe a couple of evening so roughly 5-6 hours.
Also I did not include a requirement.txt (it is basically pytorch that you need to install and flask, flask-cors) and I won't go into detail how to run a JS application such as React, as there is loads of documentation how to do that in the www.


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
        ├── index.html
        ├── ...
        └── manifest.json
```

In the backend folder are the python scripts containing two CNNs. One built from scratch, one built from transfer learning using ResNet. The weights that are also included are calculated on google colab. The Flask application is found in /backend/app.py where the application creates an API endpoint (port 5000) for predictions.

In the frontend folder is the ReactJS application which allows us to upload a picture and run the CNN.

Click on "Choose file" and upload your cat or dog image

<img src = "/docs/FrontEndtool.png">

Click on "Run CNN" to request probabilities on whether it is a dog or a cat 
(This prediction was created when the model wasn't trained, but neverthelesss I found it amusing and left it)

<img src = "/docs/FrontEndtool2.png">

## Takeaways
PyTorch has a steeper learning curve than for example Keras but it becomes more intuitiv after a while. Additonally, with PyTorch Lightning being launched users will get a similar experience as compared to Keras. If you therefore compare PyTorch with Tensorflow, PyTorch is winning in my opinion. It is just more pythonic.
Furthermore, debugging is a delight especially when you create your own neural network from scratch.
If there is one minus point, it is that PyTorch is rigorous about the data type (float, int, long...) but so is Tensorflow.



