# Module 9: Headers Deep Dive

## Concept

Every API response carries headers — metadata about the response, separate from the body. Headers reveal things the body doesn't: caching rules, rate limit status, response format, and freshness signals.

## Test

Request: GET https://api.github.com/users/SARGE2899
Status: 200 OK

## Key headers observed

| Header | Value | Meaning |
|---|---|---|
| content-type | application/json; charset=utf-8 | Officially declares the body format |
| x-ratelimit-limit | 60 | Total hourly quota for unauthenticated requests |
| x-ratelimit-remaining | 52 | Requests left this hour |
| x-ratelimit-used | 8 | Running count of requests used |
| x-ratelimit-reset | 1783443310 | Unix timestamp when quota resets |
| cache-control | public, max-age=60, s-maxage=60 | This response can be cached for 60 seconds |
| etag | W/"5c1b2951081af6ea947368290ca1040ad2bcb8de3dab00733cff85be25435589" | A fingerprint of this exact response |

## Why cache-control and etag matter for PM work

cache-control tells you the provider's own guidance on data freshness — polling an API faster than its max-age wastes calls (and rate limit budget) for no real benefit. This is directly relevant when scoping how often a product should sync/poll a third-party API.

etag enables "conditional requests" — a client can ask "has this changed since etag X?" and receive a cheap 304 Not Modified response instead of the full payload if nothing changed. This is a real efficiency pattern worth knowing when discussing sync architecture with engineering — it avoids re-fetching unchanged data.

## Rate limits, made visible via headers

Rather than needing a separate call to /rate_limit (Module 8), GitHub returns current quota status as headers on every single response. This is the standard, smarter pattern for real integrations — check headers on each response rather than making dedicated quota-check calls.

## Key takeaway

> Response bodies contain data; response headers contain metadata about that data (format, caching rules, rate limit status, freshness fingerprints). A well-designed API exposes rate limit status on every response via headers, not just a dedicated endpoint — letting integrations self-regulate without wasting calls just to check quota.
