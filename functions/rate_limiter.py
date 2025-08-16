"""
title: Rate Limit Filter
author: justinh-rahb with improvements by Yanyutin753
author_url: https://github.com/justinh-rahb
funding_url: https://github.com/open-webui
version: 0.2.1
license: MIT
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        requests_per_minute: Optional[int] = Field(
            default=10, description="Maximum number of requests allowed per minute."
        )
        requests_per_hour: Optional[int] = Field(
            default=50, description="Maximum number of requests allowed per hour."
        )
        sliding_window_limit: Optional[int] = Field(
            default=100,
            description="Maximum number of requests allowed within the sliding window.",
        )
        sliding_window_minutes: Optional[int] = Field(
            default=180, description="Duration of the sliding window in minutes."
        )
        global_limit: bool = Field(
            default=True,
            description="Whether to apply the limits globally to all models.",
        )
        enabled_for_admins: bool = Field(
            default=True,
            description="Whether rate limiting is enabled for admin users.",
        )

    def __init__(self):
        self.file_handler = False
        self.valves = self.Valves()
        self.user_requests = {}

    def prune_requests(self, user_id: str, model_id: str):
        now = time.time()

        if user_id not in self.user_requests:
            self.user_requests[user_id] = {}  # This remains a dict of model requests

        if self.valves.global_limit:
            # Clear all request timestamps for the user.
            self.user_requests[user_id] = {
                k: [
                    req
                    for req in v
                    if (
                        (self.valves.requests_per_minute is not None and now - req < 60)
                        or (
                            self.valves.requests_per_hour is not None
                            and now - req < 3600
                        )
                        or (now - req < self.valves.sliding_window_minutes * 60)
                    )
                ]
                for k, v in self.user_requests[user_id].items()
            }
        else:
            # Clear request timestamps for the specified model only.
            if model_id not in self.user_requests[user_id]:
                return

            self.user_requests[user_id][model_id] = [
                req
                for req in self.user_requests[user_id][model_id]
                if (
                    (self.valves.requests_per_minute is not None and now - req < 60)
                    or (self.valves.requests_per_hour is not None and now - req < 3600)
                    or (now - req < self.valves.sliding_window_minutes * 60)
                )
            ]

    def rate_limited(
        self, user_id: str, model_id: str
    ) -> Tuple[bool, Optional[int], int]:
        self.prune_requests(user_id, model_id)

        if self.valves.global_limit:
            user_reqs = self.user_requests.get(user_id, {})
            requests_last_minute = sum(
                1
                for reqs in user_reqs.values()
                for req in reqs
                if time.time() - req < 60
            )
            if requests_last_minute >= self.valves.requests_per_minute:
                earliest_request = min(
                    req
                    for reqs in user_reqs.values()
                    for req in reqs
                    if time.time() - req < 60
                )
                return (
                    True,
                    int(60 - (time.time() - earliest_request)),
                    requests_last_minute,
                )

            requests_last_hour = sum(
                1
                for reqs in user_reqs.values()
                for req in reqs
                if time.time() - req < 3600
            )
            if requests_last_hour >= self.valves.requests_per_hour:
                earliest_request = min(
                    req
                    for reqs in user_reqs.values()
                    for req in reqs
                    if time.time() - req < 3600
                )
                return (
                    True,
                    int(3600 - (time.time() - earliest_request)),
                    requests_last_hour,
                )

            sliding_window_seconds = self.valves.sliding_window_minutes * 60
            requests_in_window = sum(
                1
                for reqs in user_reqs.values()
                for req in reqs
                if time.time() - req < sliding_window_seconds
            )
            if requests_in_window >= self.valves.sliding_window_limit:
                earliest_request = min(
                    req
                    for reqs in user_reqs.values()
                    for req in reqs
                    if time.time() - req < sliding_window_seconds
                )
                return (
                    True,
                    int(sliding_window_seconds - (time.time() - earliest_request)),
                    requests_in_window,
                )

        # Process requests for a specific model.
        if (
            user_id not in self.user_requests
            or model_id not in self.user_requests[user_id]
        ):
            return False, None, 0

        user_reqs = self.user_requests[user_id][model_id]
        requests_last_minute = sum(1 for req in user_reqs if time.time() - req < 60)
        if requests_last_minute >= self.valves.requests_per_minute:
            earliest_request = min(req for req in user_reqs if time.time() - req < 60)
            return (
                True,
                int(60 - (time.time() - earliest_request)),
                requests_last_minute,
            )

        requests_last_hour = sum(1 for req in user_reqs if time.time() - req < 3600)
        if requests_last_hour >= self.valves.requests_per_hour:
            earliest_request = min(req for req in user_reqs if time.time() - req < 3600)
            return (
                True,
                int(3600 - (time.time() - earliest_request)),
                requests_last_hour,
            )

        sliding_window_seconds = self.valves.sliding_window_minutes * 60
        requests_in_window = sum(
            1 for req in user_reqs if time.time() - req < sliding_window_seconds
        )
        if requests_in_window >= self.valves.sliding_window_limit:
            earliest_request = min(
                req for req in user_reqs if time.time() - req < sliding_window_seconds
            )
            return (
                True,
                int(sliding_window_seconds - (time.time() - earliest_request)),
                requests_in_window,
            )

        return False, None, len(user_reqs)

    def log_request(self, user_id: str, model_id: str):
        if user_id not in self.user_requests:
            self.user_requests[user_id] = {}
        if model_id not in self.user_requests[user_id]:
            self.user_requests[user_id][model_id] = []
        self.user_requests[user_id][model_id].append(time.time())

    def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        print(f"inlet:{__name__}")
        print(f"inlet:body:{body}")
        print(f"inlet:user:{__user__}")

        if __user__ is not None and (
            __user__.get("role") != "admin" or self.valves.enabled_for_admins
        ):
            user_id = __user__["id"]
            model_id = __model__["id"] if __model__ is not None else "default_model"
            rate_limited, wait_time, request_count = self.rate_limited(
                user_id, model_id
            )
            if rate_limited:
                current_time = datetime.now()
                future_time = current_time + timedelta(seconds=wait_time)
                future_time_str = future_time.strftime("%I:%M %p")

                raise Exception(
                    f"Rate limit exceeded. You have made {request_count} requests to model '{model_id}'. Please try again at {future_time_str}."
                )

            self.log_request(user_id, model_id)
        return body
