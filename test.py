import cv2
import numpy as np
from keras.models import model_from_json
from tensorflow.python.ops.signal.shape_ops import frame

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

img= cv2.imread("C:\\Users\\KIIT\\PycharmProjects\\mental-health-tracker\\test1.jpg")
frame = cv2.resize(img, (1280, 720))
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
num_faces = np.array(num_faces, dtype='uint8')
num_faces = np.expand_dims(np.expand_dims(cv2.resize(num_faces, (48, 48)), -1), 0)
emotion_prediction = emotion_model.predict(num_faces)
maxindex = int(np.argmax(emotion_prediction))

print(emotion_dict[maxindex])
cv2.putText(frame, emotion_dict[maxindex], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
cv2.imshow('Emotion Detection', frame)

cv2.waitKey(0)

cv2.destroyAllWindows()
