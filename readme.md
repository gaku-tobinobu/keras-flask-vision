# Keras-Flask-Vision

This is a simple image classification python REST API built with Flask and Keras/Tensorflow.
 is on the model deployment using REST API than model training. There are many other resources where it talks about machine learning model building process. Works in quality detection doesn’t actually perform well but created as a very quick prototype. If you are interested in image quality detection, please refer to  NIMA https://arxiv.org/pdf/1709.05424.pdf and several git repositories, for instance https://github.com/idealo/image-quality-assessment

## Getting Started the enviroment
In order to get started, clone this library and cd into the dir.
Run ./setup.sh
This will create a virtual environment called .env and install all the dependencies.

### Starting the API
You can do either of the following:
Run below on command line (for dev and testing)
python run_server.py

This will open flask api listening to 8080

Alternatively, once gunicorn is installed, run the following, and api will be running on 8000. this is the preferred way of deploying it for production.

gunicorn --bind 0.0.0.0:8000 myapp:app

Or if you want to let it run forever, do
nohup gunicorn --bind 0.0.0.0:8000 myapp:app </dev/null >/dev/null 2>&1


### Prerequisites

I tested this locally on my Mac and on ec2 environment.


### How to call this API?
```
You can either use curl or request library in python to call this API.
For example, once you set up the repo and running this in ec2, do the following:

curl -X POST -F image=@your_imagefilename.jpg 'http://ec2-url.compute.amazonaws.com:8000/predict'
