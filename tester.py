import requests
import json

from node import print_search_results

if __name__ != '__main__': exit(1)

response = requests.get('http://localhost:5000/search', {
    'query': "test"
})

response_data = response.json()

print()
print('------- returned search results: -------')
print()

print_search_results(response_data)

print()
print('------- ALL REPS: --------')
print()

print(json.dumps(requests.get('http://localhost:5000/api/allreps').json()))
