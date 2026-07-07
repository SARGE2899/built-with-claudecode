# Module 5: DELETE — Removing a Resource

## Concept

DELETE asks the server to remove a resource entirely. No request body is needed — the URL alone identifies which resource to delete.

## Test

Request: DELETE https://jsonplaceholder.typicode.com/posts/1
Response: {}
(empty object — the expected response shape for a successful delete)

## Follow-up test: GET the same resource after deleting

Request: GET https://jsonplaceholder.typicode.com/posts/1
Response:
{
  "userId": 1,
  "id": 1,
  "title": "sunt aut facere repellat provident occaecati excepturi optio reprehenderit",
  "body": "quia et suscipit\nsuscipit recusandae consequuntur expedita et cum\nreprehenderit molestiae ut ut quas totam\nnostrum rerum est autem sunt rem eveniet architecto"
}

The post is still there, completely unchanged — not even reflecting the earlier PUT/PATCH edits, just the original seed data.

## Why this happened

JSONPlaceholder is a simulation, not a real backend. It accepts the DELETE request and returns a correct-looking empty response, but never actually removes anything from its underlying data — there's no real database behind it, just canned responses per resource ID.

## Key takeaway

> A successful-looking response (200/204, empty body) only confirms the request was accepted — it does not prove the underlying data actually changed. In real systems, this gap between "API said success" and "data actually changed" is a genuine bug category worth testing for. As a PM, "returns 200" is not sufficient acceptance criteria on its own — verify with a follow-up GET that the change actually persisted.
