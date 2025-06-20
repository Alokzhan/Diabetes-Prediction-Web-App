from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "your-secret-key"

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

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

# Routes
@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        data = [float(x) for x in request.form.values()]
        prediction = model.predict([np.array(data)])
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        pred = Prediction(user_id=session['user_id'], data=str(data), result=result)
        db.session.add(pred)
        db.session.commit()

        return render_template("index.html", prediction=result)
    except:
        return render_template("index.html", prediction="Invalid input! Please enter all values correctly.")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
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
    return render_template("login.html")

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    preds = Prediction.query.filter_by(user_id=session['user_id']).all()
    diabetic = sum(1 for p in preds if p.result == "Diabetic")
    nondiabetic = len(preds) - diabetic

    # Plot pie chart
    labels = ['Diabetic', 'Not Diabetic']
    values = [diabetic, nondiabetic]
    colors = ['#FF5733', '#2ECC71']
    plt.figure(figsize=(4, 4))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title("Prediction Summary")
    plt.savefig("static/chart.png")
    plt.close()

    return render_template("dashboard.html", diabetic=diabetic, nondiabetic=nondiabetic)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

# Only run db.create_all() inside app context
if __name__ == "__main__":
    if not os.path.exists('database.db'):
        with app.app_context():
            db.create_all()
    app.run(debug=True)
