from flask import Flask, request, jsonify
from inference import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.get_json()
    input_data = data.get('input', [])
    result = predict(input_data)
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)