import requests
import json

if __name__ != '__main__': exit(1)

response = requests.get('http://localhost:5000/search', {
    'query': "test"
})

response_data = response.json()

print(json.dumps(response_data))

print()
print('------- ALL REPS: --------')
print()

print(json.dumps(requests.get('http://localhost:5000/allreps').json()))
