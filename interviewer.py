import cv2
import time
import pyttsx3
import threading
import numpy as np
import speech_recognition as sr
import google.generativeai as genai
from keras.models import model_from_json
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__)

recognizer = sr.Recognizer()

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
engine = pyttsx3.init()
engine.setProperty('rate', 130)
engine.setProperty('volume', 0.8)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

question = ""  

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def interviewer():
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not access the camera")
        return

    threading.Thread(target=ask_question, daemon=True).start()

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Failed to read frame from webcam")
            break
        
        frame = cv2.flip(frame, 1)  
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48)) 
            img = extract_features(face) 
            pred = model.predict(img)  
            prediction_label = labels[pred.argmax()] 
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, prediction_label, (x-10, y-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    webcam.release()

def ask_question():
    global question
    genai.configure(api_key='AIzaSyDG6lsSWErYYs5K0uwkIIYUbZmt5XfSQcc')
    gmodel = genai.GenerativeModel('gemini-pro')
    time.sleep(2)
    response = gmodel.generate_content(f"Generate a behavioral interview question for a software engineer role.") 
    question = response.text.strip()
    engine.say(question)
    engine.runAndWait()

answer_given = ""
def answergiven():
    global answer_given
    with sr.Microphone() as source:
        print("Listening for answer...")
        audio = recognizer.listen(source)  # Capture audio input
        try:
            answer_given = recognizer.recognize_google(audio, language='en-us')
            return answer_given
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return "Error: Could not understand the audio"
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return "Error: Could not request results from Google"


@app.route('/answergiven')
def answer():
    return jsonify({"Answer Given": answer_given })

@app.route('/get_question')
def get_question():
    global question
    if question:  
        return jsonify({"question": question})
    else:
        return jsonify({"question": "No question available yet."})

@app.route('/')
def index():
    return render_template('interviewer.html')

@app.route('/video_feed')
def video_feed():
    return Response(interviewer(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
