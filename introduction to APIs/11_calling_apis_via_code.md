# Module 11: Calling APIs via Code

## Concept

Every request made through Postman across this course is really just a UI wrapper around code. This module shows the same calls as actual Python, using the requests library, to connect the Postman experience to what real applications do under the hood.

## Code equivalents of concepts covered

GET request (Module 1):
import requests
response = requests.get("https://api.github.com/users/SARGE2899")
print(response.status_code)
print(response.json())

POST request with a JSON body (Module 3):
payload = {"title": "My first API test", "body": "Learning how POST works", "userId": 1}
response = requests.post("https://jsonplaceholder.typicode.com/posts", json=payload)
print(response.status_code, response.json())

Bearer token auth (Module 6):
headers = {"Authorization": "Bearer YOUR_KEY_HERE"}
response = requests.get("https://api.stripe.com/v1/customers", headers=headers)

Query params (Module 2):
params = {"q": "language:python stars:>10000"}
response = requests.get("https://api.github.com/search/repositories", params=params)

## Mapping table

| Postman concept | Code equivalent |
|---|---|
| Method dropdown (GET/POST/etc) | requests.get(), requests.post(), requests.put(), requests.delete() |
| URL bar | The url argument |
| Body tab (raw JSON) | The json= argument |
| Authorization tab | The headers= argument, e.g. {"Authorization": "Bearer ..."} |
| Status code (top right) | response.status_code |
| Response body | response.json() (parsed) or response.text (raw) |
| Response Headers tab | response.headers |

## Security note

The Stripe key is never hardcoded in this script. It is read from an environment variable at runtime, and the .env file (if used locally) is excluded from version control via .gitignore. Even so, this pattern is only safe for local scripts/servers — a static frontend (e.g. GitHub Pages) can never hold a real secret key, since all its JavaScript is visible to any visitor's browser. Real secrets must only ever be used server-side.

## Key takeaway

> Postman is a learning and testing tool — real products call APIs from actual application code, using the same underlying concepts (method, URL, headers, body) in whichever language the backend is written in. Understanding the concepts in Postman first, then seeing them as code, closes the gap between "I can test an API" and "I understand what engineering is actually building" — a genuinely useful credibility signal in PM conversations about API-dependent features.

## Full test results (script run)

| Test | Status | Result |
|---|---|---|
| GET (GitHub) | 200 | Your profile data |
| POST (JSONPlaceholder) | 201 | Created resource with new id: 101 |
| GET + Bearer auth (Stripe) | 200 | Empty customer list, auth accepted |

All three calls ran successfully through the actual Python script (11_api_calls_demo.py), 
confirming every concept from Modules 1 through 9 (GET, POST, status codes, and Bearer 
token auth) works identically whether triggered from Postman or from real application code.
