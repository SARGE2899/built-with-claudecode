# Module 8: Error Handling — 4xx vs 5xx

## Concept

- 4xx = client error — you did something wrong (bad request, missing auth, wrong URL, rate limit exceeded)
- 5xx = server error — their infrastructure broke, not your fault

Status codes are the first thing to check when debugging any integration issue — before looking at the body at all.

## Test 1: 404 Not Found (GitHub)

Request: GET https://api.github.com/users/thisusernamedefinitelydoesnotexist123456
Status: 404 Not Found
Response:
{"message":"Not Found","documentation_url":"https://docs.github.com/rest","status":"404"}

Clean, correct behavior — the resource genuinely doesn't exist, and GitHub says so directly with a docs link for more detail. This is what a well-designed 404 looks like.

## Test 2: Invalid ID, but NOT a proper error (JSONPlaceholder)

Request: GET https://jsonplaceholder.typicode.com/posts/abc
Status: 200 OK
Response: {}

Notable inconsistency: "abc" is not a valid post ID, yet the API returned 200 OK (success) with just an empty object, instead of a proper 404 or 400. This is a real, common API design flaw — not every API correctly signals "this doesn't exist" via status code. Some silently return empty/null data with a success code, leaving it to the caller to check whether the response is meaningfully empty.

## Test 3: Rate limit status (GitHub)

Request: GET https://api.github.com/rate_limit
Status: 200 OK
Response (core resource):
{
  "limit": 60,
  "remaining": 54,
  "used": 6,
  "reset": 1783443310
}

Unauthenticated GitHub requests are capped at 60/hour. Every test made throughout this course against GitHub has been silently consuming this quota. When remaining hits 0, further requests return 403 Forbidden until the reset timestamp passes.

## Key takeaway

> Status codes should be the first thing checked, not the response body. But status codes are only as reliable as the API author made them — a 200 does not always guarantee real data (see Test 2). When integrating any third-party API for a product, it's worth explicitly testing edge cases like invalid IDs to see whether the API signals failure correctly, rather than assuming standard HTTP semantics are followed everywhere. Rate limits (Test 3) are also something to factor into any integration's technical design — hitting them mid-flow causes real user-facing failures.
