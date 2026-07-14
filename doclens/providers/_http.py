from __future__ import annotations

import time

import httpx

RETRY_STATUSES = {429, 500, 502, 503, 504}


def post_with_retry(client: httpx.Client, url: str, headers: dict, json: dict,
                    retries: int = 3) -> httpx.Response:
    delay = 2
    for attempt in range(retries + 1):
        resp = client.post(url, headers=headers, json=json)
        if resp.status_code not in RETRY_STATUSES or attempt == retries:
            return resp
        time.sleep(delay)
        delay *= 2
    return resp
