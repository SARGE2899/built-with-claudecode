# Module 1: GET Requests, Status Codes, and JSON

## Concept

- **GET** is the HTTP method used to *read* data. It doesn't create, change, or delete anything on the server — this makes it the "safe" verb.
- Every API call returns a **status code**: a 3-digit number telling you what happened.
  - `2xx` → success
  - `4xx` → client error (you did something wrong — bad request, not authorized, not found)
  - `5xx` → server error (something broke on their end)
- Responses are typically returned as **JSON** (JavaScript Object Notation) — a lightweight, human-readable data format built from key-value pairs.

## Why this GET call worked with zero setup

1. **GET = read-only.** APIs are generally relaxed about allowing GET requests without proof of identity, since nothing is being changed.
2. **The data was public.** GitHub profile info (username, repo count, followers) is public by design — anyone can request it.

**General rule:** Public + Read = usually open. Private or Write (POST/PUT/PATCH/DELETE) = usually needs authentication.

## Test

**Request:** GET https://api.github.com/users/SARGE2899
**Status:** 200 OK

**Response sample (trimmed for readability — full raw response saved separately in responses/01_github_user_response.json):**
{
    "login": "SARGE2899",
    "id": 78431276,
    "public_repos": 13,
    "followers": 1,
    "following": 6,
    "created_at": "2021-02-02T15:22:31Z"
}

## Decoding JSON

- { } → curly braces = a single object (one "thing")
- [ ] → square brackets = a list of objects (multiple "things")
- Inside: "key": value pairs, like one row of a spreadsheet where every column has a label.

| Key | Value | Meaning |
|---|---|---|
| login | "SARGE2899" | username |
| id | 78431276 | unique internal ID GitHub assigned |
| public_repos | 13 | number of public repos |
| created_at | "2021-02-02T15:22:31Z" | account creation timestamp |

## Notable pattern: self-referencing URLs

The response included fields like followers_url, repos_url, gists_url — other API endpoints handed back to you, telling you "if you want more detail, go here next."

## Key takeaway

> The URL identifies which resource you want. The HTTP method (GET, POST, etc.) identifies what action you want to take on it. GET on public data = no auth needed.
