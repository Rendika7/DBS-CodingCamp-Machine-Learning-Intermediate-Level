const imageUpload = document.getElementById('imageUpload');
const selectedImage = document.getElementById('selectedImage');
const result = document.getElementById('result');
let model;

async function loadModel() {
    try {
        model = await tf.loadGraphModel('modeltfjs/model.json'); // Path to your converted model
        console.log("Model loaded successfully:", model);
    } catch (error) {
        console.error("Error loading the model:", error);
    }
}


async function classifyImage() {
    try {
        const imageElement = selectedImage;
        const tensorImg = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .expandDims();
        const predictions = await model.predict(tensorImg).data();
        console.log(predictions); // Untuk memeriksa output prediksi
        displayResults(predictions);
    } catch (error) {
        console.error("Error during classification:", error);
        result.innerText = "Error during classification.";
    }
}


function displayResults(predictions) {
    const classNames = [
"battery", "biological", "brown-glass", "cardboard", "clothes", 
"green-glass", "metal", "paper", "plastic", "shoes", 
"trash", "white-glass"
    ]; // Update with your class names



    const maxPredictionIndex = predictions.indexOf(Math.max(...predictions));
    const predictedClass = classNames[maxPredictionIndex];
    result.innerText = `Prediction: ${predictedClass}`;
}

imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();
    
    reader.onload = function(event) {
        selectedImage.src = event.target.result;
        selectedImage.onload = () => classifyImage();
    };
    
    if (file) {
        reader.readAsDataURL(file);
    }
});

// Load the model when the page is loaded
window.onload = () => {
    loadModel();
};


console.log(predictions); // Untuk memeriksa output prediksi