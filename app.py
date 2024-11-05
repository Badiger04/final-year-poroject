import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)
model = joblib.load('best_model.pkl')
model

scaler = joblib.load('scaler.bin')
scaler


@app.route('/')
def Front():
    return render_template('front.html')


@app.route('/index.html', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Days = int(request.form['Days'])
        Age = int(request.form['Age'])

        gender = request.form['sex']
        sex = -1
        if gender == 'male':
            sex = 1
        elif gender == 'female':
            sex = 0

        ascites = request.form['Ascites']
        Ascites = -1
        if ascites == 'Yes':
            Ascites = 1
        elif ascites == 'No':
            Ascites = 0

        hepatomegaly = request.form['Hepatomegaly']
        Hepatomegaly = -1
        if hepatomegaly == 'No':
            Hepatomegaly = 0
        elif hepatomegaly == 'Yes':
            Hepatomegaly = 1

        spiders = request.form['Spiders']
        Spiders = -1.0
        if spiders == 'No':
            Spiders = 0.0
        elif spiders == 'Yes':
            Spiders = 1.0

        edema = request.form['Edema']
        Edema = -1
        if edema == 'No':
            Edema = 0
        elif edema == 'Yes':
            Edema = 1

        Total_Bilirubin = float(request.form['Total_Bilirubin'])
        Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
        Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
        Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
        Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
        Total_Protiens = float(request.form['Total_Protiens'])
        Albumin = float(request.form['Albumin'])
        Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])

        values = np.array([[Days, Age, sex, Ascites, Hepatomegaly, Spiders, Edema, Total_Bilirubin, Direct_Bilirubin,
                            Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens,
                            Albumin, Albumin_and_Globulin_Ratio]])

        prediction = model.predict(values)
        output = prediction
        print(output)
        if output == 1:
            return render_template('result.html',
                                   prediction_text="The person with the given details has a normal liver.")
        elif output == 2:
            return render_template('result.html',
                                   prediction_text="The person with the given details have 1st stage(Compensated) Liver Cirrhosis.")
        elif output == 3:
            return render_template('result.html',
                                   prediction_text="The person with the given details have 2nd Stage(Intermediate) Liver Cirrhosis.")
        elif output == 4:
            return render_template('result.html',
                                   prediction_text="The person with the given details have 3rd Stage(Decompensated) Liver Cirrhosis.")

        # return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
