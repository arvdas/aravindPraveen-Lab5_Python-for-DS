import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
import joblib
from flask import Flask, request, render_template, jsonify
import pandas as pd

app = Flask(__name__)  # Initialize the flask App
model = joblib.load('model.pkl')  # loading the trained model

fuelTypes = ['Petrol', 'Diesel', 'CNG']
sellerTypes = ['Dealer', 'Individual']
transmissions = ['Manual', 'Automatic']
owners = ['0', '1', '3']


@app.route('/')  # Homepage
def index():
    return render_template('index.html', fuelTypes=fuelTypes, sellerTypes=sellerTypes, transmissions=transmissions,
                           owners=owners)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = int(data['age'])
    fuelType = data['fuelType']
    sellerType = data['sellerType']
    transmission = data['transmission']
    owner = int(data['owner'])

    input_data = pd.DataFrame({
        'AGE': [age],
        'Fuel_Types': [fuelType],
        'Seller_Type': [sellerType],
        'Transmission': [transmission],
        'Owner': [owner]
    })
    input_data = pd.get_dummies(input_data, drop_first=True)
    predicted_price = model.predict(input_data)[0]
    response = {'predicted_price': predicted_price}
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
