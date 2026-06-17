import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.pipeline.predict_pipeline import CustomData

application = Flask(__name__)

app= application

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/predict", methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Get the form data
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        predict_df = data.get_data_as_dataframe()

        # Make predictions using the predict_pipeline
        from src.pipeline.predict_pipeline import PredictPipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(predict_df)

        return render_template('home.html', results=results[0])

if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0')
