'''Importing necessary packages'''
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras
import pickle
from flask import Flask,request,jsonify,render_template

'''Loading the model which was saved after training'''
loaded_model = tf.saved_model.load('trail_1')

def predict(data):
    '''Function to make a classification using the trained model'''
    y_pred = loaded_model(data)
    return y_pred[0]


app = Flask(__name__)

@app.route('/')
def main():
    '''Function to redirect to template page'''
    return render_template("app.html")

@app.route("/result", methods=["POST","GET"])
def start():
    '''Function takes in requests and processes it to model and sends the response back'''
    data = [ request.form['Text'] ]
    print(data)
    spam_score = predict(data)
    spam_score = spam_score.numpy()

    if spam_score[0] > 0.7:
        result = 'Someone is trying to spam you'
    elif 0.5< spam_score[0] < 0.7:
        result = "Looks like spam. I'm just a machine,you have to decide"
    elif 0.3 < spam_score[0] < 0.5:
        result = "Looks like a text. I'm just a machine you have to decide"
    else:
        result = "Someone just texted. Might be a girl, Reply soon"
    return render_template("app.html", result=result)

if __name__ == '__main__':
    app.run(debug=False)
