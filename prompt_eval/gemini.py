import google.generativeai as genai
from .decorators import (
    retry_on_resource_exhausted,
    retry_on_internal_server_error,
)


@retry_on_resource_exhausted
@retry_on_internal_server_error
async def generate_content(
    *,
    model: str,
    prompt: str,
    input: str,
) -> str:
    m = genai.GenerativeModel(
        model,
        system_instruction=prompt,
    )
    response = await m.generate_content_async(input)

    if not response.parts:
        raise ValueError(f"Empty response: {response}")

    return response.text
