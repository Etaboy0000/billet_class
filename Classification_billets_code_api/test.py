import requests
import json

# les donnees d'entree pour la prediction

data = {
    'diagonal': 100,
    'height_left': 102,
    'height_right': 10,
    'margin_low': 3,
    'margin_up': 2,
    'length': 8
}

# envoie de la requete POST
response = requests.post('http://localhost:5001/api/prediction',json=data)

#recuperer la reponse JSON
prediction = response.json()

# afficher les resultat

print('Prediction:', prediction['prediction'])
print('Probability:', prediction['probability'])

