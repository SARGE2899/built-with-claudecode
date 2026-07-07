# Module 6: API Key Auth (Bearer Token)

## Concept

Some APIs require proof of identity even for read-only (GET) requests, if the data is private/sensitive. This is enforced via an Authorization header containing an API key or token — commonly as a "Bearer token."

Contrast with Module 1 (GitHub): public data needed no auth at all. Stripe customer data is private, so every request needs a valid key.

## Test 1: Authenticated request

Request: GET https://api.stripe.com/v1/customers
Authorization: Bearer sk_test_... (secret key, stored as a Postman variable, never pasted in plain text or shared)

Status: 200 OK
Response:
{
  "object": "list",
  "data": [],
  "has_more": false,
  "url": "/v1/customers"
}

Empty data array — no customers created yet, but the request itself succeeded, proving the key is valid.

## Test 2: Same request, no auth

Request: GET https://api.stripe.com/v1/customers
Authorization: none

Status: 401 Unauthorized
Response:
{
  "error": {
    "message": "You did not provide an API key. You need to provide your API key in the Authorization header, using Bearer auth (e.g. 'Authorization: Bearer YOUR_SECRET_KEY').",
    "type": "invalid_request_error"
  }
}

## Key takeaway

> Public + Read = usually open (GitHub). Private data = needs proof of identity regardless of the HTTP method (Stripe, even on GET). A well-designed API rejects missing/invalid auth with a clear 401 error that tells you exactly how to fix it, rather than failing silently or vaguely — good API design principle to recognize when evaluating third-party APIs for a product integration.

## Security note

API keys are never committed to version control, pasted into shared documents, or exposed in logs/screenshots. In Postman, store them as variables (ideally marked "secret" type) rather than pasting directly into request fields.
