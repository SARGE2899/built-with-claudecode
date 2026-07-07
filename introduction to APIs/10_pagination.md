# Module 10: Pagination

## Concept

APIs rarely return an entire dataset in one response — especially for accounts with many results (in Module 2, GitHub's search had 956 total matches but only returned a page at a time). Pagination controls how many results come back per request, and how to fetch the next chunk.

Common pagination controls:
- per_page (or limit) — how many results per response
- page — which page/chunk to fetch
- Some APIs instead use a link header (see below) or a cursor/token-based system rather than page numbers

## Test

Request 1: GET https://api.github.com/users/SARGE2899/repos?per_page=3
Status: 200 OK
Returned 3 repos: academic-codebase, ai-email-agent, AI-powered-PRD-generator

Request 2: GET https://api.github.com/users/SARGE2899/repos?per_page=3&page=2
Status: 200 OK
Returned 3 different repos: AlgorithmsCourse-PrincetonUniversity, built-with-claudecode, DataStructures-and-Algorithms

Same endpoint, same per_page, different page — confirmed two completely different slices of the same underlying repo list.

## The link header (real pagination mechanism)

GitHub also returns a link response header containing ready-to-use URLs for next/prev/first/last pages, e.g.:
<https://api.github.com/user/repos?page=2&per_page=3>; rel="next"

This is the more robust pattern than manually incrementing page numbers — a real integration should follow the URL the API provides rather than assuming page+1 always works (some APIs use opaque cursor tokens instead of numbered pages specifically to prevent that assumption).

## Key takeaway

> Never assume an API returns "everything" in one call — always check for pagination fields or headers, especially before building a feature that depends on a complete dataset (e.g. "show all repos" would silently show only page 1 without proper pagination handling). When scoping an integration with engineering, ask specifically how the API paginates: page numbers, offset/limit, or cursor tokens — each has different implications for how the frontend/backend need to loop through results.
