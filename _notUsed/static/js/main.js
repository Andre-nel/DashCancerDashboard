// add the JavaScript code to handle form submission, make an AJAX request to 
// the server, and update the prediction result

// This JavaScript code listens for the form submission event, prevents the default form
// submission behavior, and sends an AJAX request to the server with the input feature values.
// The server returns the prediction probabilities, which the code then uses to update the progress
// bar, showing the probability of a malignant diagnosis.

// With this code, you have now built an interactive dashboard layout and design, incorporated
// the logistic regression model into the dashboard, and enabled users to input feature values
// to receive a cancer diagnosis prediction.

document.getElementById('prediction-form').addEventListener('submit', function (event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(prediction_proba => {
            console.log('Data received from server prediction_proba:', prediction_proba); // Add this line
            updatePredictionResult(prediction_proba);
        })
        .catch(error => {
            console.error('Error:', error);
        });
});

function updatePredictionResult(prediction_proba) {
    const progressBar = document.querySelector('#prediction-result .progress-bar');
    const progress = document.querySelector('#prediction-result .progress');
    const malignant_proba = prediction_proba[0][1] * 100;

    progressBar.style.width = `${malignant_proba}%`;
    progressBar.setAttribute('aria-valuenow', malignant_proba);
    progressBar.textContent = `Malignant: ${malignant_proba.toFixed(2)}%`;

    progress.style.display = 'block';
}

// function updateProgressBar(malignantProbability) {
//     const progressBar = document.querySelector('.progress-bar');
//     progressBar.style.width = `${malignantProbability * 100}%`;
//     progressBar.setAttribute('aria-valuenow', malignantProbability * 100);
//     progressBar.textContent = `Malignant: ${(malignantProbability * 100).toFixed(2)}%`;
// }


// document.getElementById("prediction-form").addEventListener("submit", function (event) {
//     event.preventDefault();

//     // Get the form data
//     const formData = new FormData(event.target);

//     // Send the request to the server
//     fetch("/predict", {
//         method: "POST",
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => {
//         console.log('Data received from server:', data); // Add this line
//         updateProgressBar(data[0][1]);
//     })
//     .catch(error => {
//         console.error('Error:', error);
//     });

// });
