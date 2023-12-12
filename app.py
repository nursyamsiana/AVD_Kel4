import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model2.pkl", "rb"))

# Kamus untuk mapping nilai prediksi ke label
prediction_labels = {0: "Tidak", 1: "Mungkin", 2: "Ya"}

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(X) for X in request.form.values()]
    features = [np.array(float_features)]
    
    # Melakukan prediksi
    prediction = model.predict(features)
    
    # Mengonversi nilai prediksi ke label
    predicted_label = prediction_labels.get(prediction[0], "Label tidak ditemukan")
    
    # Mengonversi hasil prediksi menjadi string dengan label
    prediction_text = "{}".format(predicted_label)
    
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)




# import numpy as np
# import pandas as pd
# from flask import Flask, render_template, request
# import pickle

# app = Flask(__name__)
# model = pickle.load(open("model.pkl", "rb"))

# @app.route("/")
# def home():
#     return render_template('index.html')

# @app.route("/predict", methods=["POST"])
# def predict():
#     float_features = [float(X) for X in request.form.values()]
#     features = [np.array(float_features)]
#     prediction = model.predict(features)
#     return render_template("index.html", prediction_text = "{}".format(prediction))

# if __name__ == "__main__":
#     app.run(debug=True)
