import requests

url = "http://127.0.0.1:5000/predict"

state = [0, 0, 1, 0, 0]  # example drone state

res = requests.post(url, json={"state": state})

print("Action from AI:", res.json())