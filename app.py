from flask import Flask, render_template, render_template_string, request, redirect, url_for, flash
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

app = Flask(__name__)
app.secret_key = 'salary_secret_key'  

model = joblib.load('model.pkl')

GENDER_OPTIONS = ['Male', 'Female']
DEPARTMENT_OPTIONS = ['HR', 'Engineering', 'Sales', 'Marketing', 'Finance']
JOB_TITLE_OPTIONS = ['Manager', 'Engineer', 'Analyst', 'HR Specialist', 'Accountant']
PROMOTION_OPTIONS = ['No', 'Yes']

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict_form.html',
            gender_options=GENDER_OPTIONS,
            department_options=DEPARTMENT_OPTIONS,
            job_title_options=JOB_TITLE_OPTIONS,
            promotion_options=PROMOTION_OPTIONS)
    else:
        gender = request.form['gender']
        age = int(request.form['age'])
        department = request.form['department']
        job_title = request.form['job_title']
        years_at_company = int(request.form['years_at_company'])
        avg_monthly_hours = int(request.form['avg_monthly_hours'])
        promotion = request.form['promotion']
        if years_at_company > (age - 18):
            flash('Years at company cannot exceed total possible working years since age 18.')
            return redirect(url_for('predict'))
        if years_at_company < 5 and promotion == 'Yes':
            flash('An employee cannot be promoted in the last 5 years if they have not worked for at least 5 years.')
            return redirect(url_for('predict'))
        gender_val = 1 if gender == 'Male' else 0
        department_val = DEPARTMENT_OPTIONS.index(department)
        job_title_val = JOB_TITLE_OPTIONS.index(job_title)
        promotion_val = 1 if promotion == 'Yes' else 0
        X = np.array([[gender_val, age, department_val, job_title_val, years_at_company, avg_monthly_hours, promotion_val]])
        pred = model.predict(X)[0]
        salary = f"{pred:,.2f}"
        return render_template('predict_result.html', salary=salary)

if __name__ == '__main__':
    app.run(debug=True) 