"""Module 11 demo: the same calls made via Postman throughout this course,
now as actual Python using the requests library.

Requires: pip install requests
Requires STRIPE_TEST_KEY to be set as an environment variable before running
Test 3 (never hardcode the key here).
"""

import os

import requests


def test_github_get():
    print("\n--- Test 1: GET github user ---")
    response = requests.get("https://api.github.com/users/SARGE2899")
    print(f"status: {response.status_code}")
    data = response.json()
    print(f"login: {data.get('login')}")
    print(f"public_repos: {data.get('public_repos')}")


def test_jsonplaceholder_post():
    print("\n--- Test 2: POST create a post ---")
    payload = {
        "title": "Calling APIs via code",
        "body": "Final module test",
        "userId": 1,
    }
    response = requests.post(
        "https://jsonplaceholder.typicode.com/posts", json=payload
    )
    print(f"status: {response.status_code}")
    print(response.json())


def test_stripe_bearer_auth():
    print("\n--- Test 3: GET stripe customers (Bearer auth) ---")
    stripe_key = os.environ.get("STRIPE_TEST_KEY")
    if not stripe_key:
        print("ERROR: STRIPE_TEST_KEY environment variable is not set. "
              "Set it before running this test, e.g.:\n"
              "  export STRIPE_TEST_KEY=sk_test_...   (macOS/Linux)\n"
              "  $env:STRIPE_TEST_KEY = 'sk_test_...' (PowerShell)")
        return

    headers = {"Authorization": f"Bearer {stripe_key}"}
    response = requests.get("https://api.stripe.com/v1/customers", headers=headers)
    print(f"status: {response.status_code}")
    print(response.json())


if __name__ == "__main__":
    test_github_get()
    test_jsonplaceholder_post()
    test_stripe_bearer_auth()
