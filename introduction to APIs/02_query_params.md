# Module 2: Query Parameters

## Concept

There are two ways a URL can carry information about what you want:

- **Path params** — part of the URL path itself, identifies *which specific resource*.
  Example: /users/SARGE2899 → "give me the resource with this exact ID/name."
- **Query params** — the ?key=value part after the URL, used to *filter, search, or modify* a request. Multiple params are chained with &.
  Example: ?q=language:python&sort=stars → "search, and narrow results by these conditions."

**Rule of thumb:** asking for one specific known thing → path param. Filtering/searching a collection → query param.

## Test

Request: GET https://api.github.com/search/repositories?q=language:python+stars:>10000
Status: 200 OK

## Response shape

{
    "total_count": 956,
    "incomplete_results": false,
    "items": [ ... ]
}

- total_count: number of matching repos that exist in total (not all returned at once — see Module 10 on pagination)
- items: the actual array of matching repo objects

## Notable pattern: field duplication for backward compatibility

Each repo object has both stargazers_count and stargazers, forks_count and forks — same data under two names. Sign of an API that evolved over time while keeping old fields for existing integrations. Worth recognizing when evaluating any API's maturity.

## Key takeaway

> Path params say "which one." Query params say "filtered/sorted how." Search/list endpoints commonly wrap results in an outer object with metadata (total_count) plus an array (items) rather than returning a bare list.
