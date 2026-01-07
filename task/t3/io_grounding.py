import asyncio
from typing import Any, Optional, Dict, List

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

"""
export DIAL_API_KEY="<SECRET>" &&
py -m venv .venv &&
source .venv/bin/activate &&
pip install -r requirements.txt &&
python3 -m task.t3.in_out_grounding
"""

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)



# Combines vector search with structured output and real-time data retrieval.

# **How it works:**
# - Uses vector similarity for initial filtering
# - Structures LLM output with Pydantic models
# - Fetches live user data for final results
# - Auto-updates vector store with new/deleted users

# **Pros:**
# - Best of both worlds: semantic search + live data
# - Structured, parseable outputs
# - Automatic data synchronization

# **Cons:**
# - Most complex implementation
# - Higher computational overhead


# Pydantic model for structured output
class HobbyUsers(BaseModel):
    """Structured output for hobby-based user grouping"""
    hobbies: Dict[str, List[int]] = Field(
        description="Dictionary mapping hobbies to lists of user IDs"
    )


class HobbiesSearchingWizard:
    """
    A wizard that searches users by hobbies using vector search and provides
    their full information grouped by hobbies.
    """
    
    def __init__(self):
        """Initialize the wizard with embeddings, LLM, vector store, and user client"""
        # Initialize embeddings model
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            azure_deployment="text-embedding-3-small-1",
            api_version="2024-12-01-preview"
        )
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            azure_deployment="gpt-4o-mini",
            api_version="2024-08-01-preview",
            temperature=0.0  # For consistent extraction
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name="user_hobbies",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db_hobbies"
        )
        
        # Initialize user client
        self.user_client = UserClient()
        
        # Track current user IDs in vector store
        self.current_user_ids: set = set()
        
        # Flag to track if vector store is initialized
        self.initialized = False
    
    async def _initialize_vector_store(self):
        """Initialize vector store with all current users"""
        try:
            # Get all users from the service
            users = self.user_client.get_all_users()
            
            # Create documents from users (only id and about_me)
            documents = []
            for user in users:
                if user.get("about_me"):
                    doc = Document(
                        page_content=user["about_me"],
                        metadata={"user_id": user["id"]},
                        id=str(user["id"])  # Use user ID as document ID
                    )
                    documents.append(doc)
                    self.current_user_ids.add(user["id"])
            
            # Add documents to vector store
            if documents:
                await self.vector_store.aadd_documents(documents)
                print(f"Initialized vector store with {len(documents)} users")
            
            self.initialized = True
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            self.initialized = False
    
    async def _update_vector_store(self):
        """Update vector store by adding new users and removing deleted ones"""
        try:
            # Get current users from the service
            current_users = self.user_client.get_all_users()
            current_ids = {user["id"] for user in current_users}
            
            # Find deleted users
            deleted_ids = self.current_user_ids - current_ids
            if deleted_ids:
                # Delete from vector store
                self.vector_store.delete(ids=[str(uid) for uid in deleted_ids])
                print(f"Deleted {len(deleted_ids)} users from vector store")
            
            # Find new users
            new_ids = current_ids - self.current_user_ids
            if new_ids:
                # Add new users to vector store
                new_documents = []
                for user in current_users:
                    if user["id"] in new_ids and user.get("about_me"):
                        doc = Document(
                            page_content=user["about_me"],
                            metadata={"user_id": user["id"]},
                            id=str(user["id"])
                        )
                        new_documents.append(doc)
                
                if new_documents:
                    await self.vector_store.aadd_documents(new_documents)
                    print(f"Added {len(new_documents)} new users to vector store")
            
            # Update current user IDs
            self.current_user_ids = current_ids
            
        except Exception as e:
            print(f"Error updating vector store: {e}")
    
    async def search_users_by_hobbies(self, query: str, top_k: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search users by hobbies based on the query
        
        Args:
            query: Natural language query about hobbies
            top_k: Number of top similar users to retrieve
            
        Returns:
            Dictionary mapping hobbies to lists of full user information
        """
        # Initialize vector store if not already done
        if not self.initialized:
            await self._initialize_vector_store()
        else:
            # Update vector store with latest users
            await self._update_vector_store()
        
        # Perform vector similarity search
        similar_docs = await self.vector_store.asimilarity_search(
            query=query,
            k=top_k
        )
        
        if not similar_docs:
            return {}
        
        # Prepare context for LLM
        context_users = []
        for doc in similar_docs:
            user_id = doc.metadata.get("user_id")
            about_me = doc.page_content
            context_users.append(f"User ID: {user_id}\nAbout: {about_me}")
        
        context = "\n---\n".join(context_users)
        
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=HobbyUsers)
        
        # Create prompt template
        system_template = """You are a hobby extraction expert. Your task is to analyze user profiles and extract hobbies related to the query.

Given a user query about hobbies and a list of user profiles, you need to:
1. Identify hobbies mentioned in the query
2. Find users who have those hobbies based on their "About" sections
3. Group users by specific hobbies

{format_instructions}

Important rules:
- Only include users whose profiles clearly mention hobbies related to the query
- Be specific with hobby names (e.g., "hiking", "rock climbing", "camping" instead of just "outdoor activities")
- A user can appear under multiple hobbies if they mention multiple relevant activities
- Only return user IDs, not any personal information
"""
        
        system_prompt = SystemMessagePromptTemplate.from_template(
            system_template,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Create chat prompt
        chat_prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            ("human", "Query: {query}\n\nUser Profiles:\n{context}")
        ])
        
        # Create chain
        chain = chat_prompt | self.llm | parser
        
        # Get structured response
        try:
            result: HobbyUsers = await chain.ainvoke({
                "query": query,
                "context": context
            })
            
            # Perform output grounding - fetch full user info and verify IDs
            grounded_result = {}
            
            for hobby, user_ids in result.hobbies.items():
                grounded_users = []
                for user_id in user_ids:
                    try:
                        # Fetch full user information - now calling synchronously
                        user_info = self.user_client.get_user(user_id)
                        grounded_users.append(user_info)
                    except Exception as e:
                        # Skip invalid user IDs
                        print(f"Could not fetch user {user_id}: {e}")
                        continue
                
                if grounded_users:
                    grounded_result[hobby] = grounded_users
            
            return grounded_result
            
        except Exception as e:
            print(f"Error in hobby extraction: {e}")
            return {}


async def main():
    """Main function to demonstrate the Hobbies Searching Wizard"""
    
    # Create wizard instance
    wizard = HobbiesSearchingWizard()
    
    print("Hobbies Searching Wizard")
    print("=" * 60)
    print("This wizard searches users by hobbies and provides their full info.")
    print("Type 'quit' to exit.\n")
    
    # Interactive mode
    while True:
        try:
            # Get user input
            query = input("\nEnter your hobby search query: ").strip()
            
            # Check for exit command
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                print("Please enter a valid query.")
                continue
            
            print(f"\nSearching for: {query}")
            print("Processing...")
            
            # Search for users
            result = await wizard.search_users_by_hobbies(query)
            
            if result:
                print(f"\nFound users in {len(result)} hobby categories:")
                for hobby, users in result.items():
                    print(f"\n{hobby.upper()} ({len(users)} users):")
                    for user in users[:3]:  # Show first 3 users per hobby
                        print(f"  - {user.get('name', 'N/A')} {user.get('surname', 'N/A')} (ID: {user.get('id', 'N/A')})")
                        if user.get('about_me'):
                            about_preview = user['about_me'][:100] + "..." if len(user['about_me']) > 100 else user['about_me']
                            print(f"    About: {about_preview}")
                    if len(users) > 3:
                        print(f"  ... and {len(users) - 3} more users")
            else:
                print("No users found matching your query.")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")


if __name__ == "__main__":
    asyncio.run(main())
