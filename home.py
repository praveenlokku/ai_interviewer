import os
import cv2
import torch
import pyttsx3
import sklearn.metrics
import threading
from fer import FER
import speech_recognition as sr
import google.generativeai as genai
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,request,jsonify,redirect,flash,url_for,render_template

app = Flask(__name__)


@app.route('/check_interview', methods=['POST', 'GET'])
def check_interview():
    if request.method == 'POST':
        company = request.form.get('company')
        job_role = request.form.get('job-role')
        candidate_name = request.form.get('candidate-name')
        difficulty = request.form.get('difficulty')

        if not company or not job_role or not candidate_name or not difficulty:
            flash("Please fill out all fields.")  
            return redirect(url_for('check_interview'))
        else:
            def interviewer():
                 return render_template('interviewer.html')
            interviewer()
            genai.configure(api_key='AIzaSyDG6lsSWErYYs5K0uwkIIYUbZmt5XfSQcc')  # Replace with a secure method
            detector = FER(mtcnn=True)
            gmodel = genai.GenerativeModel('gemini-pro')
            engine = pyttsx3.init()
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            model = BertModel.from_pretrained('bert-base-uncased')
            def get_embedding(text):
                inputs = tokenizer(text, return_tensors='pt')

                outputs = model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).detach().numpy()
            
            def speak_and_listen(question, expected_answer):
                engine.say(question)

                engine.runAndWait()
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:

                    print("Listening: ")
                    try:
                        audio = recognizer.listen(source)
                        user_answer = recognizer.recognize_google(audio, language="en-us")
                        question_embedding = get_embedding(question)
                        expected_answer_embedding = get_embedding(expected_answer)

                        user_answer_embedding = get_embedding(user_answer)
                        similarity = cosine_similarity(user_answer_embedding, expected_answer_embedding)
                        print(f"Relevance Score: {similarity[0][0]}")

                    except sr.UnknownValueError:
                        print("Sorry, I could not understand the audio.")
                    except sr.RequestError:
                        print("Could not request results from Google Speech Recognition service.")
            number_of_questions = 3
            questions_and_answers = []

            for _ in range(number_of_questions):
                response = gmodel.generate_content(f"Generate an interview question which can be asked for a {job_role} position at {company}.")

                cleaned_question = response.text.replace("Focused Interview Question:", "").strip()  # Clean the question
                expected_answer = gmodel.generate_content(f"Generate an answer for this question: {cleaned_question}")
                cleaned_expected_answer = expected_answer.text.strip()  # Clean the expected answer

                questions_and_answers.append((cleaned_question, cleaned_expected_answer))



            cap = cv2.VideoCapture(0)
            question_index = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                result = detector.detect_emotions(frame)
                for face in result:
                    (x, y, w, h) = face["box"]

                    emotions = face["emotions"]
                    anxiety_score = emotions.get("stress", 0)
                    fear_score = emotions.get("fear", 0)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f'Fear: {fear_score:.2f}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    cv2.putText(frame, f'Anxiety: {anxiety_score:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    if fear_score > 0.5:
                        print("You seem to be feeling fearful. Try to stay calm and confident!")


                cv2.imshow('Emotion Detection', frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
                
                if not threading.active_count() > 1 and question_index < len(questions_and_answers):
                    question, expected_answer = questions_and_answers[question_index]
                    threading.Thread(target=speak_and_listen, args=(question, expected_answer)).start()
                    question_index += 1

            cap.release()
            cv2.destroyAllWindows()

# Print the generated questions and answers
            for idx, (q, a) in enumerate(questions_and_answers):
                print(f"{q}")
                print(f"AI's Answer: {a}")
 

         

@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
# import pymysql
# import speech_recognition as sr
# from fer import FER
# from transformers import BertTokenizer, BertModel
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# import pyttsx3
# import threading
# import google.generativeai as genai
# import cv2

# app = Flask(__name__)
# app.secret_key = 'your_secret_key_here'  # Set your secret key for flash messages

# @app.route('/check_interview', methods=['POST', 'GET'])
# def check_interview():
#     if request.method == 'POST':
#         company = request.form.get('company')
#         job_role = request.form.get('job-role')
#         candidate_name = request.form.get('candidate-name')
#         difficulty = request.form.get('difficulty')

#         if not company or not job_role or not candidate_name or not difficulty:
#             flash("Please fill out all fields.")
#             return redirect(url_for('home'))  # Change this to redirect to the home page instead

#         # Set up your detection and model code here
#         detector = FER(mtcnn=True)
#         genai.configure(api_key='AIzaSyDG6lsSWErYYs5K0uwkIIYUbZmt5XfSQcc')  # Replace with a secure method
#         gmodel = genai.GenerativeModel('gemini-pro')

#         engine = pyttsx3.init()
#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertModel.from_pretrained('bert-base-uncased')

#         def get_embedding(text):
#             inputs = tokenizer(text, return_tensors='pt')
#             outputs = model(**inputs)
#             return outputs.last_hidden_state.mean(dim=1).detach().numpy()

#         def speak_and_listen(question, expected_answer):
#             engine.say(question)
#             engine.runAndWait()

#             recognizer = sr.Recognizer()
#             with sr.Microphone() as source:
#                 print("Listening: ")
#                 try:
#                     audio = recognizer.listen(source)
#                     user_answer = recognizer.recognize_google(audio, language="en-us")

#                     question_embedding = get_embedding(question)
#                     expected_answer_embedding = get_embedding(expected_answer)
#                     user_answer_embedding = get_embedding(user_answer)

#                     similarity = cosine_similarity(user_answer_embedding, expected_answer_embedding)
#                     print(f"Relevance Score: {similarity[0][0]}")
#                 except sr.UnknownValueError:
#                     print("Sorry, I could not understand the audio.")
#                 except sr.RequestError:
#                     print("Could not request results from Google Speech Recognition service.")

#         number_of_questions = 3
#         questions_and_answers = []

#         for _ in range(number_of_questions):
#             response = gmodel.generate_content(f"Generate an interview question which can be asked for a {job_role} position at {company}.")
#             cleaned_question = response.text.replace("Focused Interview Question:", "").strip()  # Clean the question

#             expected_answer = gmodel.generate_content(f"Generate an answer for this question: {cleaned_question}")
#             cleaned_expected_answer = expected_answer.text.strip()  # Clean the expected answer

#             questions_and_answers.append((cleaned_question, cleaned_expected_answer))

#         # Start video capture and emotion detection
#         cap = cv2.VideoCapture(0)
#         question_index = 0

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame = cv2.flip(frame, 1)
#             result = detector.detect_emotions(frame)

#             for face in result:
#                 (x, y, w, h) = face["box"]
#                 emotions = face["emotions"]

#                 anxiety_score = emotions.get("stress", 0)
#                 fear_score = emotions.get("fear", 0)

#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(frame, f'Fear: {fear_score:.2f}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
#                 cv2.putText(frame, f'Anxiety: {anxiety_score:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

#                 if fear_score > 0.5:
#                     print("You seem to be feeling fearful. Try to stay calm and confident!")

#             cv2.imshow('Emotion Detection', frame)

#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
            
#             if not threading.active_count() > 1 and question_index < len(questions_and_answers):
#                 question, expected_answer = questions_and_answers[question_index]
#                 threading.Thread(target=speak_and_listen, args=(question, expected_answer)).start()
#                 question_index += 1

#         cap.release()
#         cv2.destroyAllWindows()

#         # Print the generated questions and answers (for debugging)
#         for idx, (q, a) in enumerate(questions_and_answers):
#             print(f"{q}")
#             print(f"AI's Answer: {a}")

#         # Redirect to the interviewer page after the interview session is completed
#         return redirect(url_for('interviewer'))  # Redirect to the interviewer page

# @app.route('/interviewer')
# def interviewer():
#     return render_template('interviewer.html')  # Render the interviewer.html template

# @app.route('/home')
# def home():
#     return render_template('home.html')  # Render the home page

# if __name__ == '__main__':
#     app.run(debug=True)
