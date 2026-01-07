import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

"""
export DIAL_API_KEY="<SECRET>" &&
py -m venv .venv &&
source .venv/bin/activate &&
pip install -r requirements.txt &&
python3 -m task.t1.no_grounding.py
"""

#TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }


# Create AzureChatOpenAI client and token tracker
# Note: if your environment requires a different deployment name change azure_deployment below
llm_client = AzureChatOpenAI(
    azure_endpoint=DIAL_URL,
    openai_api_key=API_KEY,
    azure_deployment="gpt-5-mini-2025-08-07",
    api_version="2024-12-01-preview",
    temperature=1,
)

token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    """Format a list of user dictionaries into a human-readable, non-JSON string for LLM context.

    Example output:
    User:
      id: 1
      name: John
      surname: Doe
      about_me: loves hiking

    This function intentionally avoids raw JSON (double quotes) to reduce risk of the model
    treating keys/values as JSON in prompts.
    """
    parts: list[str] = []
    for user in context:
        lines = ["User:"]
        for k, v in user.items():
            # Convert value to string, replace double quotes to avoid raw JSON, and strip newlines
            try:
                val_str = str(v)
            except Exception:
                val_str = ""
            val_str = val_str.replace('"', "'")
            val_str = val_str.replace("\n", " ")
            lines.append(f"  {k}: {val_str}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    # 1. Create messages array with system prompt and user message
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

    # 2. Generate response (use `ainvoke`)
    try:
        response = await llm_client.ainvoke(messages)
    except Exception as e:
        print(f"LLM invocation failed: {e}")
        return ""

    # 3. Get usage from response metadata
    total_tokens = 0
    try:
        metadata = getattr(response, "metadata", {}) or {}
        token_usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
        if token_usage and isinstance(token_usage, dict):
            total_tokens = int(token_usage.get("total_tokens", 0))
    except Exception:
        total_tokens = 0

    # 4. Add tokens to token_tracker
    token_tracker.add_tokens(total_tokens)

    # 5. Print response content and total tokens
    content = ""
    # response may have .content or .text or be a Generation object
    if hasattr(response, "content"):
        content = response.content
    elif hasattr(response, "text"):
        content = response.text
    else:
        # Try to stringify
        content = str(response)

    print("=== Response ===")
    print(content)
    print(f"Tokens used for this call: {total_tokens}")

    # 5. return response content
    return content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        # 1. Get all users (use UserClient)
        client = UserClient()
        try:
            users = client.get_all_users()
        except Exception as e:
            print(f"Failed to fetch users: {e}")
            return

        # 2. Split all users on batches (100 users in 1 batch).
        batch_size = 100
        user_batches: list[list[dict[str, Any]]] = [users[i:i + batch_size] for i in range(0, len(users), batch_size)]

        # 3. Prepare tasks for async run of response generation for users batches
        tasks = []
        for batch in user_batches:
            context_str = join_context(batch)
            user_prompt = USER_PROMPT.format(context=context_str, query=user_question)
            tasks.append(generate_response(BATCH_SYSTEM_PROMPT, user_prompt))

        # 4. Run tasks asynchronously
        results = await asyncio.gather(*tasks)

        # 5. Filter results on 'NO_MATCHES_FOUND'
        filtered = [r for r in results if r and r.strip() != "NO_MATCHES_FOUND"]

        # 5. If results after filtration are present:
        if filtered:
            combined_results = "\n\n".join(filtered)
            # create augmented final prompt
            final_context = combined_results
            final_prompt = USER_PROMPT.format(context=final_context, query=user_question)

            print("\n--- Combining and deduplicating results ---")
            final_answer = await generate_response(FINAL_SYSTEM_PROMPT, final_prompt)
            print("\n--- Final Answer ---")
            print(final_answer)
        else:
            print("No users found matching")

        # 7. Print usage summary
        print("\n=== Usage summary ===")
        print(token_tracker.get_summary())


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation
