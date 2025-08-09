import requests

url = 'http://127.0.0.1:5000/results'

values = {'waist size': 27, 'neck size': 20, 'waist to height ratio': 47}

r = requests.post(url, json=values)

print(r.json())

