from flask import Flask, request, jsonify, flash, render_template, redirect, url_for, session
import pymysql
import pyttsx3

app = Flask(__name__)
app.secret_key = 'your_secret_key'

def speak():
    engine = pyttsx3.init()

    engine.setProperty('rate', 125)
    engine.setProperty('volume', 0.9)
    
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)

def sqlconnection():
    try:
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="Bu@#9063808032",
            database="ap",
        )
        return connection
    except pymysql.MySQLError as e:
        print(f"Database connection error: {e}")
        return None

@app.route('/login', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        email = request.form.get('username')
        password = request.form.get('password')

        if not email or not password:
            flash("Please fill out all fields.")
            return redirect(url_for('verify'))

        conn = sqlconnection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    query = "SELECT password FROM login WHERE email = %s"
                    cursor.execute(query, (email,))
                    result = cursor.fetchone()
            finally:
                conn.close()

            if result is None or result[0] != password:
                flash("Invalid credentials. Please try again.")
                return redirect(url_for('verify'))

            flash("Login successful!")
            return redirect(url_for('home'))
        else:
            flash("Database connection failed.")
            return redirect(url_for('verify'))

    return render_template('index.html')

@app.route('/forgetpassword', methods=['POST', 'GET'])
def changepass():
    if request.method == 'POST':
        email = request.form.get('email')
        changedpassword = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not email or not changedpassword or not confirm_password:
            flash("Please fill all the details")
            return redirect(url_for('changepass'))

        if changedpassword != confirm_password:
            flash("Passwords do not match")
            return redirect(url_for('changepass'))

        conn = sqlconnection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    query = "UPDATE login SET password = %s WHERE email = %s"
                    cursor.execute(query, (changedpassword, email))
                    conn.commit()
            finally:
                conn.close()

            flash("Successfully changed the password!")
            return redirect(url_for('index'))
        else:
            flash("Database connection failed.")
            return redirect(url_for('changepass'))

    return render_template('forgotPassword.html')  # Make sure this file exists

@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        name = request.form.get('fullname')
        email = request.form.get('email')
        password = request.form.get('password')

        if not name or not email or not password:
            flash("Please fill all the details")
            return redirect(url_for('signup'))

        conn = sqlconnection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    query = "INSERT INTO login (email, password) VALUES (%s, %s)"
                    cursor.execute(query, (email, password))
                    conn.commit()
            finally:
                conn.close()

            flash("Signup Successful!")
            return redirect(url_for('index'))
        else:
            flash("Database connection failed.")
            return redirect(url_for('signup'))

    return render_template('signup.html') 

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

        conn = sqlconnection()
        try:
            with conn.cursor() as cursor:
                query = "SELECT * FROM companies WHERE company_names = %s AND job_role = %s"
                cursor.execute(query, (company, job_role))
                result = cursor.fetchone()
        finally:
            conn.close()
        if result:
            return render_template('interviewer.html')
        else:
            flash(f'Sorry, the company "{company}" or the job role "{job_role}" was not found.')
            return redirect(url_for('home')) 




@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
