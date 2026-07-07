# Module 4: PUT vs PATCH — Updating a Resource

## Concept

Both update an existing resource, but differently:
- PUT = replace the entire resource. You must send all fields — anything omitted would be wiped in a real system.
- PATCH = update only the fields you send. Everything else stays untouched.

## Test 1: PUT

Request: PUT https://jsonplaceholder.typicode.com/posts/1
Body:
{
    "id": 1,
    "title": "Updated title only",
    "body": "Updated body only",
    "userId": 1
}

Response:
{
  "id": 1,
  "title": "Updated title only",
  "body": "Updated body only",
  "userId": 1
}

All fields sent were all fields returned — full replacement.

## Test 2: PATCH

Request: PATCH https://jsonplaceholder.typicode.com/posts/1
Body:
{
    "title": "Only updating the title"
}

Response:
{
  "userId": 1,
  "id": 1,
  "title": "Only updating the title",
  "body": "quia et suscipit\nsuscipit recusandae consequuntur expedita et cum\nreprehenderit molestiae ut ut quas totam\nnostrum rerum est autem sunt rem eveniet architecto"
}

Only title was sent, but the original body text survived untouched — proof that PATCH only modifies what you explicitly include. (Note: this body text is placeholder Lorem-Ipsum-style filler from JSONPlaceholder's seeded fake data, not anything meaningful.)

## Key takeaway

> PUT = full replacement, send everything. PATCH = partial update, send only what changed. This matters for real product decisions: PATCH lets a frontend send small, efficient updates (e.g. just a changed profile field) without re-sending the entire object every time.
