import os
from typing import List, Dict, Any, Optional
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from .faiss_indexer import FAISSIndexBuilder

logger = logging.getLogger(__name__)


class MessageRAGEngine:
    """RAG Engine that works without RetrievalQA"""

    def __init__(
        self,
        index_builder: FAISSIndexBuilder,
        model_name: str = "gpt-4-turbo",
        temperature: float = 0.2,
    ):
        """Initialize RAG engine"""

        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Please set OPENAI_API_KEY environment variable")

        self.model_name = model_name
        self.index_builder = index_builder

        # Initialize LLM
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=500)

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()

        # Build vectorstore
        self.vectorstore = self._build_vectorstore()

        logger.info(f"RAG Engine ready with {model_name}")

    def _build_vectorstore(self) -> FAISS:
        """Build FAISS vectorstore"""

        documents = []
        for msg in self.index_builder.messages:
            content = (
                f"User: {msg.user_name}\n" f"Date: {msg.timestamp}\n" f"Message: {msg.message}"
            )

            doc = Document(
                page_content=content,
                metadata={"user_name": msg.user_name, "timestamp": msg.timestamp},
            )
            documents.append(doc)

        vectorstore = FAISS.from_documents(documents=documents, embedding=self.embeddings)

        logger.info(f"Created vectorstore with {len(documents)} documents")
        return vectorstore

    def _get_prompt_for_question(self, question: str) -> str:
        """Get appropriate prompt based on question type"""

        question_lower = question.lower()

        # Counting questions
        if any(word in question_lower for word in ["how many", "count", "total", "number"]):
            return """You are analyzing messages from a luxury concierge service.
Count the requested information.

Context:
{context}

Question: {question}

Instructions:
1. Count each unique occurrence
2. List what you found
3. Provide the exact total

Answer:"""

        # Analysis questions
        if any(word in question_lower for word in ["analyze", "pattern", "popular", "trend"]):
            return """You are analyzing messages from a luxury concierge service.
Analyze patterns and trends.

Context:
{context}

Question: {question}

Instructions:
1. Look for patterns across messages
2. Identify trends and preferences
3. Provide specific examples

Answer:"""

        # Default QA
        return """You are analyzing messages from a luxury concierge service.
Answer based ONLY on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer only from the context
- Mention specific client names
- Include dates and locations
- Say "I don't have that information" if not in context

Answer:"""

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using manual RAG pipeline"""

        try:
            # Step 1: Retrieve relevant documents
            relevant_docs = self.vectorstore.similarity_search(question, k=7)
            print(relevant_docs)
            if not relevant_docs:
                return {
                    "answer": "I couldn't find any relevant information.",
                    "relevant_messages": [],
                }

            # Step 2: Create context from documents
            context_parts = []
            for doc in relevant_docs:
                context_parts.append(doc.page_content)
            context = "\n\n".join(context_parts)

            # Step 3: Get appropriate prompt
            prompt_template = self._get_prompt_for_question(question)

            # Step 4: Format prompt with context
            prompt = prompt_template.format(context=context, question=question)

            # Step 5: Get answer from LLM
            response = self.llm.invoke(prompt)

            # Extract answer (handling different response types)
            if hasattr(response, "content"):
                answer = response.content
            else:
                answer = str(response)

            # Step 6: Format relevant messages
            relevant_messages = []
            for doc in relevant_docs[:3]:  # Top 3
                content = doc.page_content
                message = content.split("Message: ")[-1] if "Message: " in content else content

                relevant_messages.append(
                    {"user": doc.metadata.get("user_name", "Unknown"), "message": message}
                )

            return {"answer": answer.strip(), "relevant_messages": relevant_messages}

        except Exception as e:
            logger.error(f"Error: {e}")

            # Fallback to basic search
            try:
                results = self.index_builder.search(question, k=1)
                if results:
                    msg, _ = results[0]
                    answer = f"Based on the messages, {msg.user_name} mentioned: {msg.message}"
                else:
                    answer = "I couldn't find relevant information."

                return {"answer": answer, "relevant_messages": []}
            except:
                return {"answer": f"Error processing question: {str(e)}", "relevant_messages": []}
