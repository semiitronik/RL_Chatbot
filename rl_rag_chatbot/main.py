# ============================================================
# SECTION 1: IMPORTS
# ============================================================

import json
import math
import os
import uuid
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField
from sqlmodel import Session, SQLModel, create_engine, select
from openai import OpenAI


# ============================================================
# SECTION 2: APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="RL-Style RAG Chatbot API",
    version="1.0.0",
    description=(
        "A FastAPI backend that demonstrates document ingestion, "
        "embeddings-based retrieval, RAG, and a reviewer sub-agent."
    ),
)


# ============================================================
# SECTION 3: ENVIRONMENT VARIABLES / CLIENT SETUP
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///rag_chatbot.db")

client = OpenAI(api_key=OPENAI_API_KEY)
engine = create_engine(DATABASE_URL, echo=False)


# ============================================================
# SECTION 4: DATABASE MODELS
# ============================================================

class Document(SQLModel, table=True):
    """
    Stores top-level document metadata.
    """
    id: str = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    title: str
    created_at: datetime = SQLField(default_factory=datetime.utcnow)


class Chunk(SQLModel, table=True):
    """
    Stores each chunk of a document plus its embedding.
    """
    id: str = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: str = SQLField(index=True)
    chunk_index: int
    text: str
    embedding_json: str
    created_at: datetime = SQLField(default_factory=datetime.utcnow)


class Conversation(SQLModel, table=True):
    """
    Stores a chat session for a specific document.
    """
    id: str = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: str = SQLField(index=True)
    created_at: datetime = SQLField(default_factory=datetime.utcnow)


class Message(SQLModel, table=True):
    """
    Stores messages exchanged during a conversation.
    """
    id: str = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    conversation_id: str = SQLField(index=True)
    role: str
    content: str
    created_at: datetime = SQLField(default_factory=datetime.utcnow)


# ============================================================
# SECTION 5: REQUEST / RESPONSE SCHEMAS
# ============================================================

class IngestRequest(BaseModel):
    title: str = Field(..., description="A friendly title for the document")
    text: str = Field(..., min_length=1, description="The full raw text of the document")


class IngestResponse(BaseModel):
    document_id: str
    title: str
    chunk_count: int
    message: str


class ChatRequest(BaseModel):
    document_id: str
    question: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    top_k: int = Field(default=4, ge=1, le=8)


class RetrievedChunk(BaseModel):
    chunk_id: str
    chunk_index: int
    score: float
    preview: str


class ChatResponse(BaseModel):
    conversation_id: str
    document_id: str
    answer: str
    citations: List[str]
    retrieved_chunks: List[RetrievedChunk]


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    created_at: str


# ============================================================
# SECTION 6: DATABASE STARTUP / SESSION HELPERS
# ============================================================

def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


@app.on_event("startup")
def on_startup() -> None:
    create_db_and_tables()


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


# ============================================================
# SECTION 7: TEXT CHUNKING
# ============================================================

def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 200) -> List[str]:
    """
    Splits a document into overlapping chunks for retrieval.

    Why overlap matters:
    It preserves context when a useful sentence spans two chunk boundaries.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(cleaned):
        end = start + chunk_size
        chunks.append(cleaned[start:end])
        start += chunk_size - overlap

    return chunks


# ============================================================
# SECTION 8: EMBEDDING HELPERS
# ============================================================

def create_embedding(text: str) -> List[float]:
    """
    Creates an embedding vector for one input string.
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    return response.data[0].embedding


# ============================================================
# SECTION 9: VECTOR SIMILARITY HELPERS
# ============================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Computes cosine similarity between two vectors.
    """
    if len(a) != len(b):
        return -1.0

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return -1.0

    return dot_product / (magnitude_a * magnitude_b)


# ============================================================
# SECTION 10: RETRIEVAL LAYER
# ============================================================

def retrieve_top_chunks(
    session: Session,
    document_id: str,
    question: str,
    top_k: int = 4,
) -> List[Tuple[Chunk, float]]:
    """
    True retrieval step for RAG:
    1. Embed the user's question
    2. Load all chunks for the document
    3. Compare each chunk embedding to the question embedding
    4. Return the top-k most similar chunks
    """
    question_embedding = create_embedding(question)

    chunks = session.exec(
        select(Chunk).where(Chunk.document_id == document_id)
    ).all()

    scored_chunks: List[Tuple[Chunk, float]] = []

    for chunk in chunks:
        chunk_embedding = json.loads(chunk.embedding_json)
        score = cosine_similarity(question_embedding, chunk_embedding)
        scored_chunks.append((chunk, score))

    scored_chunks.sort(key=lambda pair: pair[1], reverse=True)
    return scored_chunks[:top_k]


# ============================================================
# SECTION 11: CONVERSATION HISTORY HELPERS
# ============================================================

def load_recent_messages(
    session: Session,
    conversation_id: str,
    limit: int = 6,
) -> List[Message]:
    """
    Loads the most recent messages in a conversation.
    """
    messages = session.exec(
        select(Message).where(Message.conversation_id == conversation_id)
    ).all()

    messages.sort(key=lambda msg: msg.created_at)
    return messages[-limit:]


# ============================================================
# SECTION 12: PRIMARY ANSWER AGENT
# ============================================================

def answer_agent(
    question: str,
    retrieved: List[Tuple[Chunk, float]],
    history: List[Message],
) -> str:
    """
    The main agent that drafts an answer using retrieved chunks.
    """
    context_blocks: List[str] = []
    for chunk, score in retrieved:
        context_blocks.append(
            f"[Chunk {chunk.chunk_index} | chunk_id={chunk.id} | score={score:.4f}]\n{chunk.text}"
        )

    history_text = "\n".join(
        f"{message.role.upper()}: {message.content}" for message in history
    )

    joined_context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are the primary answer agent for a document question-answering system.

Instructions:
1. Use the retrieved document context as your main source of truth.
2. If the answer is not clearly supported by the context, say so.
3. Be concise, accurate, and useful.
4. Do not invent facts not grounded in the retrieved chunks.

RECENT CONVERSATION:
{history_text}

RETRIEVED CONTEXT:
{joined_context}

USER QUESTION:
{question}
""".strip()

    response = client.responses.create(
        model="gpt-5.4",
        input=prompt,
    )

    return response.output_text


# ============================================================
# SECTION 13: REVIEWER / CITATION SUB-AGENT
# ============================================================

def reviewer_sub_agent(
    question: str,
    draft_answer: str,
    retrieved: List[Tuple[Chunk, float]],
) -> Tuple[str, List[str]]:
    """
    A sub-agent that reviews the draft answer and adds grounded citations.

    It:
    - checks the draft against the retrieved context
    - removes or softens unsupported claims
    - adds inline chunk citations like [chunk-2]
    """
    context_blocks: List[str] = []
    citation_labels: List[str] = []

    for chunk, score in retrieved:
        label = f"chunk-{chunk.chunk_index}"
        citation_labels.append(label)
        context_blocks.append(
            f"[{label} | chunk_id={chunk.id} | score={score:.4f}]\n{chunk.text}"
        )

    joined_context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are a review-and-citation sub-agent.

Your tasks:
1. Review the draft answer.
2. Remove or soften any claim not clearly supported by the retrieved context.
3. Return a final answer with inline citations like [chunk-1].
4. Use only citations that are actually relevant.
5. Keep the answer clear and concise.

USER QUESTION:
{question}

DRAFT ANSWER:
{draft_answer}

RETRIEVED CONTEXT:
{joined_context}
""".strip()

    response = client.responses.create(
        model="gpt-5.4",
        input=prompt,
    )

    final_answer = response.output_text
    used_citations = [
        label for label in citation_labels
        if f"[{label}]" in final_answer
    ]

    return final_answer, used_citations


# ============================================================
# SECTION 14: ROOT / HEALTH ENDPOINTS
# ============================================================

@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "RAG chatbot API is running."
    }


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {
        "status": "ok"
    }


# ============================================================
# SECTION 15: DOCUMENT INGESTION ENDPOINT
# ============================================================

@app.post("/documents/ingest", response_model=IngestResponse)
def ingest_document(
    request: IngestRequest,
    session: Session = Depends(get_session),
) -> IngestResponse:
    """
    Ingests a document into the system:
    1. stores document metadata
    2. chunks the text
    3. creates embeddings for each chunk
    4. stores chunks + embeddings in the database
    """
    chunks = chunk_text(request.text)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Document text is empty after cleaning."
        )

    document = Document(title=request.title)
    session.add(document)
    session.commit()
    session.refresh(document)

    for index, chunk_value in enumerate(chunks):
        embedding = create_embedding(chunk_value)
        chunk = Chunk(
            document_id=document.id,
            chunk_index=index,
            text=chunk_value,
            embedding_json=json.dumps(embedding),
        )
        session.add(chunk)

    session.commit()

    return IngestResponse(
        document_id=document.id,
        title=document.title,
        chunk_count=len(chunks),
        message="Document ingested, chunked, embedded, and stored successfully.",
    )


# ============================================================
# SECTION 16: CHAT ENDPOINT
# ============================================================

@app.post("/chat", response_model=ChatResponse)
def chat_with_document(
    request: ChatRequest,
    session: Session = Depends(get_session),
) -> ChatResponse:
    """
    Full RAG flow:
    1. Validate the document
    2. Create or reuse a conversation
    3. Retrieve relevant chunks using embeddings
    4. Draft an answer with the primary agent
    5. Review and cite it with the reviewer sub-agent
    6. Save chat messages
    7. Return answer + retrieval metadata
    """
    document = session.get(Document, request.document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    if request.conversation_id:
        conversation = session.get(Conversation, request.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        conversation = Conversation(document_id=request.document_id)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

    history = load_recent_messages(session, conversation.id)

    retrieved = retrieve_top_chunks(
        session=session,
        document_id=request.document_id,
        question=request.question,
        top_k=request.top_k,
    )

    if not retrieved:
        raise HTTPException(
            status_code=400,
            detail="No retrievable chunks found for this document."
        )

    draft_answer = answer_agent(
        question=request.question,
        retrieved=retrieved,
        history=history,
    )

    final_answer, citations = reviewer_sub_agent(
        question=request.question,
        draft_answer=draft_answer,
        retrieved=retrieved,
    )

    session.add(
        Message(
            conversation_id=conversation.id,
            role="user",
            content=request.question,
        )
    )
    session.add(
        Message(
            conversation_id=conversation.id,
            role="assistant",
            content=final_answer,
        )
    )
    session.commit()

    retrieved_payload = [
        RetrievedChunk(
            chunk_id=chunk.id,
            chunk_index=chunk.chunk_index,
            score=round(score, 4),
            preview=chunk.text[:180].replace("\n", " "),
        )
        for chunk, score in retrieved
    ]

    return ChatResponse(
        conversation_id=conversation.id,
        document_id=request.document_id,
        answer=final_answer,
        citations=citations,
        retrieved_chunks=retrieved_payload,
    )


# ============================================================
# SECTION 17: DOCUMENT LIST ENDPOINT
# ============================================================

@app.get("/documents", response_model=List[DocumentSummary])
def list_documents(
    session: Session = Depends(get_session),
) -> List[DocumentSummary]:
    """
    Lists all documents currently stored in the database.
    """
    documents = session.exec(select(Document)).all()

    return [
        DocumentSummary(
            document_id=document.id,
            title=document.title,
            created_at=document.created_at.isoformat(),
        )
        for document in documents
    ]


# ============================================================
# SECTION 18: OPTIONAL DEBUG ENDPOINT FOR CONVERSATION HISTORY
# ============================================================

@app.get("/conversations/{conversation_id}")
def get_conversation_messages(
    conversation_id: str,
    session: Session = Depends(get_session),
) -> List[Dict[str, str]]:
    """
    Returns all messages in a conversation.
    Helpful for demos and debugging.
    """
    conversation = session.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = session.exec(
        select(Message).where(Message.conversation_id == conversation_id)
    ).all()

    messages.sort(key=lambda msg: msg.created_at)

    return [
        {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at.isoformat(),
        }
        for message in messages
    ]
