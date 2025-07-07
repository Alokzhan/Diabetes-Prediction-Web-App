import os
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or 'dev-secret-key'

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Ensure static directory exists
os.makedirs('static', exist_ok=True)

# Load model
model = joblib.load("model.pkl")


# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    data = db.Column(db.String(200))
    result = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# Create tables
with app.app_context():
    db.create_all()


# Routes
@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        data = [float(x) for x in request.form.values()]
        prediction = model.predict([np.array(data)])
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        pred = Prediction(
            user_id=session['user_id'],
            data=str(data),
            result=result
        )
        db.session.add(pred)
        db.session.commit()

        return render_template("index.html", prediction=result)
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']

        if User.query.filter_by(username=uname).first():
            return render_template("register.html", error="Username already exists")

        hashed_pwd = generate_password_hash(pwd)
        user = User(username=uname, password=hashed_pwd)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template("register.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        user = User.query.filter_by(username=uname).first()

        if user and check_password_hash(user.password, pwd):
            session['user_id'] = user.id
            return redirect(url_for('home'))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    preds = Prediction.query.filter_by(user_id=session['user_id']).all()
    diabetic = sum(1 for p in preds if p.result == "Diabetic")
    nondiabetic = len(preds) - diabetic

    # Plot pie chart
    plt.figure(figsize=(4, 4))
    plt.pie(
        [diabetic, nondiabetic],
        labels=['Diabetic', 'Not Diabetic'],
        autopct='%1.1f%%',
        colors=['#FF5733', '#2ECC71']
    )
    plt.title("Prediction Summary")
    chart_path = os.path.join('static', 'chart.png')
    plt.savefig(chart_path)
    plt.close()

    return render_template(
        "dashboard.html",
        diabetic=diabetic,
        nondiabetic=nondiabetic
    )


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))


# Do NOT use app.run() if using Gunicorn (Render will call gunicorn)
# Just keep this reference for dev:
if __name__ == "__main__":
    app.run(debug=True)
