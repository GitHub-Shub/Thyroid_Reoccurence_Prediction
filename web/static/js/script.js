document.getElementById("predictionForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    const formData = new FormData(this);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch("/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.error) {
            alert("Error: " + result.error);
            return;
        }

        document.getElementById("results").classList.remove("hidden");
        document.getElementById("probability").textContent = 
            `Predicted Recurrence Probability: ${(result.probability * 100).toFixed(2)}%`;

        const riskDiv = document.getElementById("riskLevel");
        riskDiv.textContent = result.risk_level;
        riskDiv.className = `risk-indicator ${result.risk_class}`;
    } catch (err) {
        console.error("Error making prediction:", err);
        alert("Error making prediction: " + err.message);
    }
});
