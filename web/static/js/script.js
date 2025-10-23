document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const data = Object.fromEntries(formData.entries());
    
    try {
        const response = await fetch('/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        document.getElementById('probability').textContent = 
            `Recurrence Probability: ${(result.probability * 100).toFixed(2)}%`;
        
        const riskLevel = document.getElementById('riskLevel');
        riskLevel.textContent = result.risk_level;
        riskLevel.className = `risk-indicator ${result.risk_class}`;
        
        document.getElementById('results').classList.remove('hidden');
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error making prediction. Please try again.');
    }
});