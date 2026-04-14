import urllib.request, json, time, statistics

URL = "http://abfc7cee14a5549279d871c549811c66-236180841.us-east-1.elb.amazonaws.com"

house = {
    "area_sq_ft": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3,
    "guestroom": 0, "basement": 0, "airconditioning": 1, "prefarea": 1,
    "parking": 2, "furnishingstatus_semi_furnished": 0,
    "furnishingstatus_unfurnished": 0
}

payload = json.dumps(house).encode()
latencies = []

print("Sending 100 requests to real-time endpoint...")
for i in range(100):
    req = urllib.request.Request(
        URL + "/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    latencies.append((time.perf_counter() - start) * 1000)

latencies.sort()
print(f"Requests : 100")
print(f"Mean     : {statistics.mean(latencies):.2f}ms")
print(f"P50      : {latencies[49]:.2f}ms")
print(f"P95      : {latencies[94]:.2f}ms")
print(f"P99      : {latencies[98]:.2f}ms")
print(f"Max      : {latencies[-1]:.2f}ms")