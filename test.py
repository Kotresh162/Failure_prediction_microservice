import requests

url = "http://127.0.0.1:8000/predict"

test_data = [
    [3, 0, 4, 4, 3, 0, 32, 3, 3],  
    [640, 7, 7, 5, 7, 4, 33, 3, 3],  
    [3, 0, 4, 4, 3, 0, 32, 3, 3],  
    [640, 7, 7, 5, 7, 4, 33, 3, 3],  
    [100, 1, 7, 7, 5, 6, 35, 2, 3],  
    [100, 7, 4, 4, 7, 2, 43, 5, 3],  
    [11, 3, 6, 6, 3, 2, 76, 6, 3],  
    [0, 7, 4, 3, 1, 6, 45, 3, 3],  
    [4, 7, 4, 6, 6, 0, 88, 2, 3],  
    [35, 6, 4, 4, 2, 1, 46, 3, 4],
]

for i, data in enumerate(test_data):
    response = requests.post(url, json={"features": data})
    print(f"Sample {i+1}: {data} -> Prediction: {response.json()}")
