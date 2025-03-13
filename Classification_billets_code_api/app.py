import pickle
from flask import Flask, request, jsonify, render_template

# Charger le modèle entraîné 
filename = 'best_model.pkl'
with open(filename, 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def render_default():
    return render_template("index.html")

@app.route('/api/prediction', methods=['POST'])
def prediction():
    # Récupérer les données d'entrée depuis la requête POST
    data = request.get_json()  

    # Convertir les données en un tableau NumPy
    # columns:  ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
    billet = [
        data['diagonal'],
        data['height_left'], 
        data['height_right'],
        data['margin_low'],
        data['margin_up'],
        data['length']
    ]
 
    print("Feature received: ", billet)

    # Effectuer la prédiction sur les données d'entrée
    prediction = model.predict([billet])[0]
    probability = model.predict_proba([billet])[0][1]

    # Préparer la réponse de l'API
    response = {
        'prediction': 'Vrai' if prediction else 'Faux',
        'probability': probability
    }

    # Renvoyer la réponse au format JSON
    return jsonify(response)

if __name__ == '__main__':
    
    app.run(host='localhost', port=5001, debug=True)

    
