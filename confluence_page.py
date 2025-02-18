"""
title: Confluence page
description: This tool allows you to retrieve content from a specific page on Confluence.
repository: https://github.com/RomainNeup/open-webui-utilities
author: @romainneup
author_url: https://github.com/RomainNeup
funding_url: https://github.com/sponsors/RomainNeup
requirements: markdownify
version: 0.1.0
changelog:
- 0.0.1 - Initial code base.
- 0.0.2 - Fix Valves variables
- 0.1.0 - Split Confuence search and Confluence get page
"""

import base64
import json
import requests
from typing import Awaitable, Callable, Dict, Any
from pydantic import BaseModel, Field
from markdownify import markdownify


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


class Confluence:
    def __init__(self, username: str, api_key: str, base_url: str):
        self.base_url = base_url
        self.headers = self.authenticate(username, api_key)
        pass

    def get(self, endpoint: str, params: Dict[str, Any]):
        url = f"{self.base_url}/rest/api/{endpoint}"
        response = requests.get(url, params=params, headers=self.headers)
        return response.json()

    def get_page(self, page_id: str):
        endpoint = f"content/{page_id}"
        params = {"expand": "body.view", "include-version": "false"}
        result = self.get(endpoint, params)
        return {
            "id": result["id"],
            "title": result["title"],
            "body": markdownify(result["body"]["view"]["value"]),
            "link": f"{self.base_url}{result['_links']['webui']}",
        }

    def authenticate(self, username: str, api_key: str):
        auth_string = f"{username}:{api_key}"
        encoded_auth_string = base64.b64encode(auth_string.encode("utf-8")).decode(
            "utf-8"
        )
        return {"Authorization": "Basic " + encoded_auth_string}


class Tools:
    def __init__(self):
        self.valves = self.Valves()
        pass

    class Valves(BaseModel):
        username: str = Field("example@example.com", description="Your username here")
        api_key: str = Field("ABCD1234", description="Your API key here")
        base_url: str = Field(
            "https://example.atlassian.net/wiki",
            description="The base URL of your Confluence instance",
        )

    # Get content from Confluence
    async def get_confluence_page(
        self,
        page_id: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: dict = {}
    ) -> Dict[str, Any]:
        """
        Get the content of a page on Confluence. This returns the content of a specific page on Confluence.
        Use it to get the content of a specific page on Confluence. When a user requests a specific page, this must be used.
        :param page_id: The ID of the page on Confluence
        :return: The content of the page on Confluence in JSON format (title, body, link). If the page is not found, an error message is returned.
        """
        confluence = Confluence(self.valves.username, self.valves.api_key, self.valves.base_url)
        event_emitter = EventEmitter(__event_emitter__)
        await event_emitter.emit_status(f"Retrieving page '{page_id}' from Confluence...", False)
        try:
            result = confluence.get_page(page_id)
            await event_emitter.emit_status(f"Retrieved page '{page_id}' from Confluence.", True)
            await event_emitter.emit_source(result["title"], result["link"], result["body"])
            return json.dumps(result)
        except Exception as e:
            await event_emitter.emit_status(f"Failed to retrieve page '{page_id}': {e}.", True, True)
            return f"Error: {e}"
