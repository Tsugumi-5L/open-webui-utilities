"""
title: Jira
description: This tool allows you to search for and retrieve content from Jira.
repository: https://github.com/RomainNeup/open-webui-utilities
author: @romainneup
author_url: https://github.com/RomainNeup
funding_url: https://github.com/sponsors/RomainNeup
version: 0.0.1
changelog:
- 0.0.1 - Initial code base.
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

    def post(self, endpoint: str, data: Dict[str, Any]):
        url = f"{self.base_url}/rest/api/3/{endpoint}"
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()

    def get_issue(self, issue_id: str):
        endpoint = f"issue/{issue_id}"
        result = self.get(
            endpoint,
            {"fields": "summary,description,status", "expand": "renderedFields"},
        )

        return {
            "title": result["fields"]["summary"],
            "description": result["renderedFields"]["description"],
            "status": result["fields"]["status"]["name"],
            "link": f"{self.base_url}/browse/{issue_id}",
        }

    def search(self, query: str):
        endpoint = "search"
        params = {"jql": f"text ~ '{query}'", "maxResults": 5}
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

    async def get_issue(
        self,
        issue_id: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: dict = {},
    ):
        """
        Get a Jira issue by its ID. The response includes the title, description as HTML, status, and link to the issue.
        :param issue_id: The ID of the issue.
        :return: A response in JSON format (title, description, status, link).
        """
        jira = Jira(self.valves.username, self.valves.api_key, self.valves.base_url)
        event_emitter = EventEmitter(__event_emitter__)
        try:
            await event_emitter.emit_status(f"Getting issue {issue_id}", False)
            response = jira.get_issue(issue_id)
            await event_emitter.emit_source(
                response["title"], response["link"], response["description"], True
            )
            await event_emitter.emit_status(f"Got issue {issue_id}", True)
            return json.dumps(response)
        except Exception as e:
            await event_emitter.emit_status(
                f"Failed to get issue {issue_id}: {e}", True, True
            )
            return f"Error: {e}"
