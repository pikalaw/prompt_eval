import asyncio
from devtools import debug
import functools
from google.api_core.exceptions import (
    DeadlineExceeded,
    InternalServerError,
    ResourceExhausted,
)
import logging
from typing import (
    Any,
    Awaitable,
    Callable,
    ParamSpec,
    TypeVar,
)


P = ParamSpec("P")
R = TypeVar("R")
F = Callable[P, Awaitable[R]]


MAX_SEC_TO_WAIT_ON_RESOURCE_EXHAUSTED = 60 * 5
MAX_INTERNAL_SERVER_ERRORS = 10


semaphore = asyncio.BoundedSemaphore(value=20)


def limit_concurrency(*, concurrency: int):
    def decorator(func: F) -> F:

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def retry_on_resource_exhausted(func: F) -> F:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        wait_time = 1
        while True:
            try:
                return await func(*args, **kwargs)
            except ResourceExhausted as e:
                logging.warning(
                    debug.format(
                        "Resource exhausted",
                        exception=e,
                        wait_time_to_retry=wait_time,
                    )
                )
                await asyncio.sleep(wait_time)
                if wait_time < MAX_SEC_TO_WAIT_ON_RESOURCE_EXHAUSTED:
                    wait_time *= 2

    return wrapper


def retry_on_internal_server_error(func: F) -> F:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        wait_time = 1
        try_count = 1
        while True:
            try:
                return await func(*args, **kwargs)
            except (DeadlineExceeded, InternalServerError) as e:
                if try_count >= MAX_INTERNAL_SERVER_ERRORS:
                    raise

                logging.warning(
                    debug.format(
                        "Internal Server Error",
                        exception=f"{e}",
                        wait_time_to_retry=wait_time,
                    )
                )
                await asyncio.sleep(wait_time)
                wait_time *= 2
                try_count += 1

    return wrapper
