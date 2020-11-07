# OpenCVFaceDetection

I I am using a Raspberry Pi V3 updated to the last version of Raspbian (Stretch), so the best way to have OpenCV installed, is to follow the excellent tutorial developed by Adrian Rosebrock: Raspbian Stretch: Install OpenCV 3 + Python on your Raspberry Pi(https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/).

Once you finished Adrian’s tutorial, you should have an OpenCV virtual environment ready to run our experiments on your Pi.
Let’s go to our virtual environment and confirm that OpenCV 3 is correctly installed.
Adrian recommends run the command “source” each time you open up a new terminal to ensure your system variables have been set up correctly.

source ~/.profile

Next, let’s enter on our virtual environment:

workon cv

If you see the text (cv) preceding your prompt, then you are in the cv virtualenvironment:

(cv) pi@raspberry:~$

Adrian calls the attention that the cv Python virtual environment is entirely independent and sequestered from the default Python version included in the download of Raspbian Stretch. So, any Python packages in the global site-packages directory will not be available to the cv virtual environment. Similarly, any Python packages installed in site-packages of cv will not be available to the global install of Python.

Now, enter in your Python interpreter:

python

and confirm that you are running the 3.5 (or above) version.
Inside the interpreter (the “>>>” will appear), import the OpenCV library:

import cv2

If no error messages appear, the OpenCV is correctly installed ON YOUR PYTHON VIRTUAL ENVIRONMENT.
You can also check the OpenCV version installed:
cv2.__version__

The 3.3.0 should appear (or a superior version that can be released in future).


Testing Raspberi Pi Camera:-

sudo apt update
sudo apt full-upgrade
sudo raspi-config -> Enable camera support using the raspi-config program you will have used when you first set up your Raspberry Pi.
raspistill -v -o test.jpg The display should show a five-second preview from the camera and then take a picture, saved to the file test.jpg, whilst displaying various informational messages.

Test OpenCV and Camera:-

    import numpy as np
    import cv2
    cap = cv2.VideoCapture(0) <- use cap = cv2.VideoCapture(1) if you are using an USB camera(Like me)
    cap.set(3,640) 
    cap.set(4,480)
    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, -1) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', frame)
        cv2.imshow('gray', gray)

        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()

To finish the program, you must press the key [ESC] on your keyboard. Click your mouse on the video window, before pressing [ESC].





Now 

Face Detection is divided into two parts:-

a) Train the Model

b) Predict a face using trained data

Train a Model:-

Input User name and Take Multiple snapshot of a user using 
    
     cap = cv2.VideoCapture(1)
     ret, frame = cap.read()
Frame would have the image, there on, we save every snampshot to create a model:

    model = cv2.face.LBPHFaceRecognizer_create()
    
Onec model is created, we predict the user in an infinite loop(only interupted by 13/escape) We take another screenshot of the user's camera 
 
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
and then prediction is very simple:-

    results = model.predict(face)

Confidence of prediction can be found by:-

    confidence = int(100 * (1 - (results[1]) / 400))

We consider a threshold of 89% as a positive detection of a face.



    
