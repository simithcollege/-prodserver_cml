import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open(f'model/model/model.pkcl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        # Extract the input
        fixedAcidity = flask.request.form['fixed-acidity']
        volatileAcidity = flask.request.form['volatile-acidity']
        citricAcid = flask.request.form['citric-acid']

        residualSugar = flask.request.form['residual-sugar']
        chlorides = flask.request.form['chlorides']
        freeSulfur = flask.request.form['free-sulfur']

        totalSulfur = flask.request.form['total-sulfur']
        density = flask.request.form['density']
        ph = flask.request.form['ph']

        sulphates = flask.request.form['sulphates']
        alcohol = flask.request.form['alcohol']

        # Make DataFrame for model
        input_variables = [[fixedAcidity, volatileAcidity, citricAcid, residualSugar, chlorides, freeSulfur, totalSulfur, density, ph, sulphates, alcohol]]

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
        #prediction = 1001

        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Fixed Acidity':fixedAcidity,
                                                     'Volatile Acidity':volatileAcidity,
                                                     'Citric Acid':citricAcid,
                                                     'Residual sugar':residualSugar,
                                                     'Chlorides':chlorides,
                                                     'Free sulfur dioxide':freeSulfur,
                                                     'Total sulfur dioxide':totalSulfur,
                                                     'Density':density,
                                                     'PH':ph,
                                                     'Sulphates':sulphates,
                                                     'Alcohol':alcohol
                                                     },
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()
