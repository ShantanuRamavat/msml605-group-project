async function predict() {

    const data = {
        area_sq_ft: parseFloat(document.getElementById("area").value),
        bedrooms: parseInt(document.getElementById("bedrooms").value),
        bathrooms: parseInt(document.getElementById("bathrooms").value),
        stories: parseInt(document.getElementById("stories").value),
        guestroom: 0,
        basement: 0,
        airconditioning: 1,
        prefarea: 0,
        parking: 1,
        furnishingstatus_semi_furnished: 1,
        furnishingstatus_unfurnished: 0
    };

    const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    document.getElementById("result").innerText =
        "Price: $" + result.predicted_price +
        " | Latency: " + result.latency_ms + " ms";
}