import asyncio
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

"""
export DIAL_API_KEY="<SECRET>" &&
py -m venv .venv &&
source .venv/bin/activate &&
pip install -r requirements.txt &&
python3 -m task.t2.input_vector_based.py
"""

#TODO:
# Before implementation open the `vector_based_grounding.png` to see the flow of app

# Provide System prompt. Goal is to explain LLM that in the user message will be provide rag context that is retrieved
# based on user question and user question and LLM need to answer to user based on provided context
SYSTEM_PROMPT = """
You are a RAG-powered assistant that answers user questions about users based ONLY on the information
presented in the provided RAG CONTEXT and conversation history.

INSTRUCTIONS:
- Use information from `RAG CONTEXT` to answer the `USER QUESTION`.
- Cite sources when referencing specific information from the context where appropriate.
- If no relevant information is present in `RAG CONTEXT`, state that you cannot answer the question.
- Do not hallucinate or invent facts not present in the context.
- Be concise and helpful.
"""

# Should consist retrieved context and user question
USER_PROMPT = """
## RAG CONTEXT:
{context}

## USER QUESTION:
{query}
"""


def format_user_document(user: dict[str, Any]) -> str:
    # Prepare context from users JSONs in the same way as in `no_grounding.py` `join_context` method (collect as one string)
    parts: list[str] = ["User:"]
    for k, v in user.items():
        try:
            val_str = str(v)
        except Exception:
            val_str = ""
        val_str = val_str.replace('"', "'")
        val_str = val_str.replace("\n", " ")
        parts.append(f"  {k}: {val_str}")
    return "\n".join(parts)


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        # 1. Get all users (use UserClient)
        client = UserClient()
        try:
            users = client.get_all_users()
        except Exception as e:
            print(f"Failed to fetch users: {e}")
            users = []

        # 2. Prepare array of Documents where page_content is `format_user_document(user)`
        documents: list[Document] = []
        for user in users:
            content = format_user_document(user)
            metadata = {"user_id": user.get("id")}
            documents.append(Document(page_content=content, metadata=metadata))

        # 3. call `_create_vectorstore_with_batching` (async) and setup it as obj var `vectorstore`
        if documents:
            try:
                self.vectorstore = await self._create_vectorstore_with_batching(documents, batch_size=100)
            except Exception as e:
                print(f"Failed to create vectorstore: {e}")
                self.vectorstore = None

        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        # 1. Split all `documents` on batches (100 documents in 1 batch).
        doc_batches: list[list[Document]] = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

        # 2. Iterate through document batches and create array with tasks that will generate FAISS vector stores from documents
        tasks = []
        for batch in doc_batches:
            # FAISS.afrom_documents is async according to the API notes
            tasks.append(FAISS.afrom_documents(documents=batch, embedding=self.embeddings))

        # 3. Gather tasks with asyncio
        vectorstores = await asyncio.gather(*tasks)

        # 4. Create `final_vectorstore` via merge of all vector stores
        if not vectorstores:
            raise Exception("No vectorstores were created")

        final_vs = vectorstores[0]
        if len(vectorstores) > 1:
            # Merge additional vectorstores into the first one by calling the instance method `merge_from`.
            # The FAISS.merge_from is an instance method that expects the target vectorstore as an argument.
            for vs in vectorstores[1:]:
                try:
                    final_vs.merge_from(vs)
                except Exception as e:
                    # If merge_from fails for some reason, raise an informative error.
                    raise Exception(f"Failed to merge vectorstores: {e}")

        # 5. Return `final_vectorstore`
        return final_vs

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        if not self.vectorstore:
            print("Vectorstore is not initialized")
            return ""

        # 1. Make similarity search
        try:
            results = await self.vectorstore.asimilarity_search_with_relevance_scores(query, k=k)
        except Exception:
            # fallback to sync if async method not available
            results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)

        # 2. Create `context_parts` empty array
        context_parts: list[str] = []

        # 3. Iterate through retrieved relevant docs (tuple (doc, relevance_score))
        for item in results:
            # item may be (doc, score) or Generation
            try:
                doc, score = item
            except Exception:
                # Unexpected format
                continue

            # Score filtering (depending on scoring convention; include by threshold)
            try:
                numeric_score = float(score)
            except Exception:
                numeric_score = 0.0

            # Append if passes threshold
            if numeric_score >= score:
                print(f"[score={numeric_score}] {doc.page_content[:200]}...")
                context_parts.append(doc.page_content)
            else:
                print(f"[skipped score={numeric_score}] {doc.page_content[:100]}...")

        # 4. Return joined context
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        # Make augmentation for USER_PROMPT via `format` method
        augmented = USER_PROMPT.format(context=context, query=query)
        print("--- Augmented prompt ---")
        print(augmented)
        return augmented

    def generate_answer(self, augmented_prompt: str) -> str:
        # 1. Create messages array with system prompt and user prompt
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=augmented_prompt)]

        # 2. Generate response
        try:
            response = self.llm_client.invoke(messages)
        except Exception as e:
            print(f"LLM invocation failed: {e}")
            return ""

        # 3. Return response content
        content = ""
        if hasattr(response, "content"):
            content = response.content
        elif hasattr(response, "text"):
            content = response.text
        else:
            content = str(response)

        return content


async def main():

    # 1. Create AzureOpenAIEmbeddings
    # embedding model 'text-embedding-3-small-1'
    # I would recommend to set up dimensions as 384
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        openai_api_key=API_KEY,
        azure_deployment="text-embedding-3-small-1",
        api_version="2024-12-01-preview",
        # dimensions may be optional; if supported, uncomment
        # dimensions=384,
    )

    # 2. Create AzureChatOpenAI
    llm_client = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        openai_api_key=API_KEY,
        azure_deployment="gpt-5-mini-2025-08-07",
        api_version="2024-12-01-preview",
        temperature=1,
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break

            # 1. Retrieve context
            context = await rag.retrieve_context(user_question, k=10, score=0.1)

            # 2. Make augmentation
            augmented = rag.augment_prompt(user_question, context)

            # 3. Generate answer and print it
            answer = rag.generate_answer(augmented)
            print("\n--- Answer ---")
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce
