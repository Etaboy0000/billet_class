// document.getElementById('billetForm').addEventListener('submit', function(event) 
function Manage_click()

{
    // event.preventDefault();

    // Récupérer les valeurs des champs du formulaire
    const diagonal = document.getElementById('diagonal').value;
    const height_left = document.getElementById('height_left').value;
    const height_right = document.getElementById('height_right').value;
    const margin_low = document.getElementById('margin_low').value;
    const margin_up = document.getElementById('margin_up').value;
    const length = document.getElementById('length').value;

    // Créer l'objet de données à envoyer
    const data = {
        diagonal: parseFloat(diagonal),
        height_left: parseFloat(height_left),
        height_right: parseFloat(height_right),
        margin_low: parseFloat(margin_low),
        margin_up: parseFloat(margin_up),
        length: parseFloat(length)
    };

    // Envoyer la requête POST à l'API
    $.ajax(
        {
            url : 'http://localhost:5001/api/prediction', 
        type: 'POST',

        headers: {
            'Content-Type': 'application/json'
        },
        data: JSON.stringify(data),
        
        success: function(data) {
            // Afficher les résultats dans la page
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = `
                <h2>Résultat:</h2>
                <p>Prédiction: ${data.prediction}</p>
                <p>Probabilité: ${data.probability}</p>
            `;
        },
        error: function(xhr, status, error) {
            console.error('Erreur:', error);
            document.getElementById('result').textContent = 'Erreur lors de la requête. Veuillez réessayer.';
        },
        cache : false,
        processData: false
    });
    
}


// fetch('http://localhost:5001/api/prediction', {
//     method: 'POST',
//     headers: {
//         'Content-Type': 'application/json'
//     },
//     body: JSON.stringify(data)
// })
// .then(response => {
//     if (!response.ok) {
//         throw new Error(`HTTP error! status: ${response.status}`);
//     }
//     return response.json();
// })
// .then(data => {
//     const resultDiv = document.getElementById('result');
//     resultDiv.innerHTML = `
//         <h2>Résultat:</h2>
//         <p>Prédiction: ${data.prediction}</p>
//         <p>Probabilité: ${data.probability}</p>
//     `;
// })
// .catch(error => {
//     console.error('Erreur:', error);
//     document.getElementById('result').textContent = `Erreur lors de la requête : ${error.message}`;
// });
