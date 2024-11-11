import os
import time
import threading
import numpy as np
import torch
import pyttsx3
import cv2
import speech_recognition as sr
import google.generativeai as genai
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import model_from_json
from flask import Flask, Response, render_template, request, flash, redirect, url_for

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flash messages

# Initialize global variables
questions = []
user_answers = []
question_index = 0

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load emotion detection model and weights
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(model_json)
emotion_model.load_weights("emotiondetector.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Speech recognizer
recognizer = sr.Recognizer()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def uservideo():
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
        
        frame = cv2.flip(frame, 1)  # Flip the frame to avoid mirror effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract face region
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))  # Resize to 48x48
            img = extract_features(face)  # Extract features
            pred = emotion_model.predict(img)  # Predict emotion
            prediction_label = labels[pred.argmax()]  # Get the label
            
            # Draw rectangle around the face and display prediction label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, prediction_label, (x-10, y-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
    webcam.release()


def ask_question():
    global question_index
    global questions
    global user_answers

    if question_index < len(questions):
        question = questions[question_index]
        pyttsx3.speak(question)  # Ask the question using TTS

        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)

            try:
                user_answer = recognizer.recognize_google(audio, language='en-US')
                user_answers.append(user_answer)  # Store the user's answer
                question_index += 1  # Move to the next question
            except sr.UnknownValueError:
                print("Sorry, I did not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

@app.route('/check_interview', methods=['POST'])
def check_interview():
    global question_index
    global questions
    global user_answers

    if request.method == 'POST':
        company = request.form.get('company')
        role = request.form.get('job-role')
        name = request.form.get('candidate-name')

        if not company or not role or not name:
            flash("Please fill all the details.....")
            return redirect(url_for('home'))
        
        genai.configure(api_key='AIzaSyDG6lsSWErYYs5K0uwkIIYUbZmt5XfSQcc')  # Replace with a secure method
        gmodel = genai.GenerativeModel('gemini-pro')

        no_of_questions = 3
        questions = []

        for _ in range(no_of_questions):
            response = gmodel.generate_content(f"Generate an interview question for a {role} role at {company}.")
            cleaned_question = response.text.strip()
            questions.append(cleaned_question)

        question_index = 0
        threading.Thread(target=ask_question, daemon=True).start()

        return render_template('interviewer.html', questions=questions, user_answers=user_answers)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/interviewer')
def interviewer():
    return render_template('interviewer.html', questions=questions, user_answers=user_answers)

@app.route('/video_feed')
def video_feed():
    return Response(uservideo(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
