
const form = document.getElementById("prediction-form");
const result = document.getElementById("result");

form.addEventListener("submit", async function (event) {
    event.preventDefault();

    result.textContent = "Predicting price...";

    const laptopData = {
        Laptop: document.getElementById("laptop").value.trim(),
        Status: document.getElementById("status").value,
        Brand: document.getElementById("brand").value.trim(),
        Model: document.getElementById("model").value.trim(),
        CPU: document.getElementById("cpu").value.trim(),
        RAM: Number(document.getElementById("ram").value),
        Storage: Number(document.getElementById("storage").value),
        Storage_type: document.getElementById("storage-type").value,
        GPU: document.getElementById("gpu").value.trim(),
        Screen: Number(document.getElementById("screen").value),
        Touch: document.getElementById("touch").value
    };

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(laptopData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        result.textContent =
            "Estimated Laptop Price: " + data.predicted_price;

    } catch (error) {
        result.textContent =
            "Error: " + error.message +
            ". Make sure the API is running.";
    }
});
