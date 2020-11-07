import cv2
import numpy as np
from pathlib import Path
import glob

def face_extractor(img):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x:x + w]
    return cropped_face
def face_detector(img, size=0.5):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
      return img, []
    for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
      roi = img[y:y + h, x:x + w]
      roi = cv2.resize(roi, (300,300))
    return img, roi
def train_model(name):
    rootd = Path(__file__).parent
    rootd = rootd.joinpath(rootd, "faces")
    data_path = rootd.joinpath(rootd, "user")
    pattern = '*' + name + '.jpg'
    data_path = data_path.joinpath(data_path, pattern)
    onlyfiles = []
    for filename in glob.glob(str(data_path.absolute())):
        onlyfiles.append(filename)
    Training_Data, Labels = [], []
    for i, files in enumerate(onlyfiles):
        rootd = Path(__file__).parent
        rootd = rootd.joinpath(rootd, "faces")
        rootd = rootd.joinpath(rootd, "user")
        data_path = rootd.joinpath(rootd,onlyfiles[i])
        image_path = data_path
        images = cv2.imread(image_path.__str__(), cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    Labels = np.asarray(Labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    return model
def predict_user(model, name):
    # Open Webcam
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            results = model.predict(face)
            if results[1] < 500:
                confidence = int(100 * (1 - (results[1]) / 400))
            if confidence > 89:
                display_string = str(confidence) + '% Confident it is User ' + name
                cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 120, 150), 2)
                cv2.putText(image, "This is User " + name, (250, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Recognition', image)
            else:
                cv2.putText(image, "Cannot Detect User " + name, (250, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Recognition', image)
        except:
            cv2.putText(image, "No Face Found", (220, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Recognition', image)
            pass
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            print("User Interupted")
            break
    cap.release()
    cv2.destroyAllWindows()
def face_extractor(img):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x:x + w]
    return cropped_face
def train_user(name):
    cap = cv2.VideoCapture(1)
    count = 0
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (300, 300))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = './faces/user/' + str(count) + name + '.jpg'
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count)+"/200", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Face not found-> Please look at the camera!!!")
            pass

        if cv2.waitKey(1) == 13 or count == 200:  # 13 is the Enter Key
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    while True:
        name = str(input("Enter your First Name Please!: "))
        if name:
            break
    prompt = input("Do You want to Train/Retrain the model Please?[y/n or Y/N]:")
    if prompt == 'y' or prompt == 'Y':
        train_user(name)
    predict_user(train_model(name),name)
