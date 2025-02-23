"""
title: Jira Search
description: This tool allows you to search issues from Jira.
repository: https://github.com/RomainNeup/open-webui-utilities
author: @romainneup
author_url: https://github.com/RomainNeup
funding_url: https://github.com/sponsors/RomainNeup
version: 0.0.2
changelog:
- 0.0.1 - Initial code base.
- 0.0.2 - Implement Jira search
"""


import base64
import json
from typing import Any, Awaitable, Callable, Dict
import requests
from pydantic import BaseModel, Field


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Awaitable[None]]):
        self.event_emitter = event_emitter
        pass

    async def emit_status(self, description: str, done: bool, error: bool = False):
        await self.event_emitter(
            {
                "data": {
                    "description": f"{done and (error and '❌' or '✅') or '🔎'} {description}",
                    "status": done and "complete" or "in_progress",
                    "done": done,
                },
                "type": "status",
            }
        )

    async def emit_message(self, content: str):
        await self.event_emitter({"data": {"content": content}, "type": "message"})

    async def emit_source(self, name: str, url: str, content: str, html: bool = False):
        await self.event_emitter(
            {
                "type": "citation",
                "data": {
                    "document": [content],
                    "metadata": [{"source": url, "html": html}],
                    "source": {"name": name},
                },
            }
        )


class Jira:
    def __init__(self, username: str, api_key: str, base_url: str):
        self.base_url = base_url
        self.headers = self.authenticate(username, api_key)
        pass

    def authenticate(self, username: str, api_key: str):
        auth_string = f"{username}:{api_key}"
        encoded_auth_string = base64.b64encode(auth_string.encode("utf-8")).decode(
            "utf-8"
        )
        return {"Authorization": "Basic " + encoded_auth_string}

    def get(self, endpoint: str, params: Dict[str, Any]):
        url = f"{self.base_url}/rest/api/3/{endpoint}"
        response = requests.get(url, params=params, headers=self.headers)
        return response.json()

    def search(self, query: str):
        endpoint = "search"
        terms = query.split()
        if terms:
            cql_terms = " OR ".join([f'text ~ "{term}"' for term in terms])
        else:
            cql_terms = f'text ~ "{query}"'
        params = {"jql": f"{cql_terms}", "maxResults": 5}
        rawResponse = self.get(endpoint, params)
        response = []
        for item in rawResponse["issues"]:
            response.append(item["key"])
        return response


class Tools:
    def __init__(self):
        self.valves = self.Valves()
        pass

    class Valves(BaseModel):
        username: str = Field("", description="Your username here")
        api_key: str = Field("", description="Your API key here")
        base_url: str = Field("", description="Your Jira base URL here")

    async def search_jira(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: dict = {},
    ) -> str:
        """
        Search for a query on Jira. This returns the result of the search on Jira.
        Use it to search for a query on Jira. When a user mentions a search on Jira, this must be used.
        Note: This returns a list of issues that match the search query.
        :param query: The text to search for on Jira. MUST be a string.
        :return: A list of search results from Jira in JSON format (key). If no results are found, an empty list is returned.
        """
        jira = Jira(self.valves.username, self.valves.api_key, self.valves.base_url)
        event_emitter = EventEmitter(__event_emitter__)

        await event_emitter.emit_status(
            f"Searching for '{query}' on Jira...", False
        )
        try:
            searchResponse = jira.search(query)
            await event_emitter.emit_status(
                f"Search for '{query}' on Jira complete. ({len(searchResponse)} results found)",
                True,
            )
            return json.dumps(searchResponse)
        except Exception as e:
            await event_emitter.emit_status(
                f"Failed to search for '{query}' on Jira: {e}", True, True
            )
            return f"Error: {e}"
