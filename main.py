"""
Compliance Auditor FastAPI Application
Integrates Neo4j (Graph DB), Qdrant (Vector DB), and Ollama (LLM) for compliance checking
"""

import logging
import os
import re
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import httpx
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Configuration
TENANT_ID = os.getenv("TENANT_ID", "DefaultTenant")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama2")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Compliance Auditor API", version="1.0.0")

# Initialize Neo4j driver
neo4j_driver = None
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    logger.info("✓ Neo4j driver initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Neo4j driver: {e}")
    neo4j_driver = None

# Initialize Qdrant client
qdrant_client = None
try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info("✓ Qdrant client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Qdrant client: {e}")
    qdrant_client = None


# Pydantic models for OpenAI API compatibility
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    messages: List[Message]
    stream: bool = Field(default=False)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# Helper functions
def create_openai_response(content: str, model: str) -> ChatCompletionResponse:
    """Create OpenAI-compatible response format"""
    return ChatCompletionResponse(
        id=f"chatcmpl-{datetime.now().timestamp()}",
        created=int(datetime.now().timestamp()),
        model=model,
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }]
    )


def query_neo4j_requirements(query: str, limit: int = 5) -> str:
    """
    IMPROVED: Query Neo4j for requirements and EUR-Lex documents with enhanced logging and error handling.
    
    Args:
        query: User query string
        limit: Maximum number of results to return
        
    Returns:
        Formatted context string with sources
    """
    if not neo4j_driver:
        logger.warning("⚠️ Neo4j driver not available - skipping graph queries")
        return ""
    
    # Extract keywords from query
    keywords = re.findall(r'\b[a-zäöüß]{4,}\b', query.lower())
    logger.info(f"🔍 Neo4j search with keywords: {keywords[:5]}")
    
    context = ""
    found_requirements = 0
    found_eurlex = 0
    
    # Query 1: Requirements nodes
    try:
        with neo4j_driver.session() as session:
            # Build query for requirements
            cypher_query = """
            MATCH (r:Requirement)
            WHERE any(keyword IN $keywords WHERE toLower(r.text) CONTAINS keyword)
            RETURN r.id as id, r.text as text, r.source as source, r.line as line
            LIMIT $limit
            """
            
            result_requirements = session.run(cypher_query, keywords=keywords[:5], limit=limit)
            
            for record in result_requirements:
                found_requirements += 1
                source = record.get('source', 'Unknown')
                line = record.get('line', 'N/A')
                text = record.get('text', '')
                req_id = record.get('id', '')
                
                context += f"📄 Quelle: {source}"
                if line and line != 'N/A':
                    context += f", Zeile {line}"
                context += f"\n"
                
                if req_id:
                    context += f"[SOLL-ANFORDERUNG: {req_id}]\n"
                context += f"{text}\n\n"
        
        logger.info(f"✓ Found {found_requirements} requirements in Neo4j")
    
    except Exception as e:
        logger.error(f"❌ Error querying requirements: {e}", exc_info=True)
    
    # Query 2: EUR-Lex articles
    try:
        with neo4j_driver.session() as session:
            # Build query for EUR-Lex articles
            cypher_query = """
            MATCH (a:Article)-[:PART_OF]->(d:Document)
            WHERE any(keyword IN $keywords WHERE 
                toLower(a.text) CONTAINS keyword OR 
                toLower(a.title) CONTAINS keyword)
            RETURN a.number as number, a.title as title, a.text as text, 
                   d.celex as celex, d.title as doc_title
            LIMIT $limit
            """
            
            result_eurlex = session.run(cypher_query, keywords=keywords[:5], limit=limit)
            
            for record in result_eurlex:
                found_eurlex += 1
                celex_info = record.get('celex', 'N/A')
                article_num = record.get('number', 'N/A')
                article_title = record.get('title', '')
                article_text = record.get('text', '')
                doc_title = record.get('doc_title', '')
                
                context += f"📋 Quelle: CELEX:{celex_info}, Artikel {article_num}\n"
                if doc_title:
                    context += f"Dokument: {doc_title}\n"
                if article_title:
                    context += f"Titel: {article_title}\n"
                context += f"{article_text}\n\n"
        
        logger.info(f"✓ Found {found_eurlex} EUR-Lex nodes in Neo4j")
    
    except Exception as e:
        logger.error(f"❌ Error querying EUR-Lex nodes: {e}", exc_info=True)
    
    # Summary logging
    total_found = found_requirements + found_eurlex
    logger.info(f"📊 Neo4j query complete: {total_found} total results")
    
    if not context:
        logger.warning("⚠️ Neo4j query returned NO results - check if database is populated!")
    
    return context


def query_qdrant_documents(query: str, limit: int = 5) -> tuple[str, List[str]]:
    """
    Query Qdrant vector database for relevant document chunks.
    
    Args:
        query: User query string
        limit: Maximum number of results to return
        
    Returns:
        Tuple of (formatted context string, list of sources)
    """
    if not qdrant_client:
        logger.warning("⚠️ Qdrant client not available - skipping vector search")
        return "", []
    
    context = ""
    sources = []
    
    try:
        # Get embedding for query (simplified - in production use actual embedding model)
        # For now, we'll simulate with a placeholder
        logger.info("Querying Qdrant vector database...")
        
        # Simulated search - in real implementation, use actual embeddings
        # search_results = qdrant_client.search(
        #     collection_name="documents",
        #     query_vector=query_embedding,
        #     limit=limit
        # )
        
        # For this implementation, we'll note that Qdrant integration would go here
        logger.info("✓ Qdrant query completed (implementation placeholder)")
        
    except Exception as e:
        logger.error(f"❌ Error querying Qdrant: {e}", exc_info=True)
    
    return context, sources


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Compliance Auditor API",
        "version": "1.0.0",
        "tenant": TENANT_ID
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "neo4j": "connected" if neo4j_driver else "disconnected",
        "qdrant": "connected" if qdrant_client else "disconnected",
        "ollama": "configured"
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    IMPROVED: OpenAI-compatible chat completion endpoint with enhanced source tracking and validation.
    Integrates Neo4j graph data and Qdrant vector search with comprehensive logging.
    """
    
    # Extract user query from messages
    user_query = ""
    for msg in request.messages:
        if msg.role == "user":
            user_query = msg.content
            break
    
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Initialize tracking variables
    context = ""
    has_neo4j_data = False
    has_qdrant_data = False
    neo4j_sources = []
    qdrant_sources = []
    
    logger.info("=== CHAT REQUEST ===")
    logger.info(f"User Query: {user_query[:100]}...")
    logger.info(f"Model: {request.model}")
    logger.info(f"Stream: {request.stream}")
    
    # Query Neo4j for structured requirements and EUR-Lex documents
    logger.info("Querying Neo4j...")
    neo4j_context = query_neo4j_requirements(user_query, limit=5)
    
    if neo4j_context:
        has_neo4j_data = True
        context += "=== STRUKTURIERTE SOLL-ANFORDERUNGEN & EUR-LEX DOKUMENTE (Neo4j Graph Database) ===\n"
        context += neo4j_context + "\n"
        
        # Extract sources from Neo4j context
        neo4j_sources = re.findall(r'(?:📄|📋)\s*Quelle:\s*([^\n]+)', neo4j_context)
        logger.info(f"✓ Neo4j data found: {len(neo4j_sources)} sources")
    else:
        logger.warning("⚠️ NO Neo4j data found - graph database might be empty!")
    
    # Query Qdrant for document chunks
    logger.info("Querying Qdrant...")
    qdrant_context, qdrant_sources = query_qdrant_documents(user_query, limit=5)
    
    if qdrant_context:
        has_qdrant_data = True
        context += "=== IST-DOKUMENTATION (Qdrant Vector Database) ===\n"
        context += qdrant_context + "\n"
        logger.info(f"✓ Qdrant data found: {len(qdrant_sources)} sources")
    else:
        logger.info("ℹ️ No Qdrant data found")
    
    # Validate that we have some context
    if not context.strip():
        logger.error("❌ NO CONTEXT FOUND - both Neo4j and Qdrant returned empty!")
        return create_openai_response(
            "⚠️ Keine Informationen in der Datenbank gefunden. Bitte prüfen Sie:\n"
            "1. Wurden Dokumente mit /ingest importiert?\n"
            "2. Wurden EUR-Lex Dokumente mit /ingest/eurlex importiert?\n"
            "3. Ist die Neo4j Datenbank befüllt?",
            model=request.model
        )
    
    # Log context summary
    all_sources = neo4j_sources + qdrant_sources
    logger.info(f"=== CONTEXT SUMMARY ===")
    logger.info(f"Neo4j sources: {len(neo4j_sources)}")
    logger.info(f"Qdrant sources: {len(qdrant_sources)}")
    logger.info(f"Total context length: {len(context)} chars")
    logger.info(f"Available sources: {all_sources[:5]}")
    
    # Build source list for system prompt
    source_list = "\n".join([f"- {s}" for s in all_sources[:10]])
    if not source_list:
        source_list = "Keine spezifischen Quellen verfügbar"
    
    # Enhanced system prompt with mandatory source citation
    system_prompt = f"""Du bist ein Compliance-Auditor für {TENANT_ID} mit Spezialisierung auf EU-Regulierung.

🔴 KRITISCHE REGEL: JEDE Aussage MUSS mit exakten Quellen belegt werden! 🔴

VERFÜGBARE QUELLEN IN DIESEM KONTEXT:
{source_list}

AUFGABE: Prüfe IST-Dokumente gegen SOLL-Anforderungen aus EU-Verordnungen und internen Richtlinien.

🔴 QUELLENANGABE IST PFLICHT! 🔴

VORSCHRIFT FÜR QUELLENZITATION:
==========================================
**JEDE EINZELNE AUSSAGE** muss mit **PRÄZISEN QUELLEN** belegt werden:

EUR-LEX Artikel (aus Neo4j oder Qdrant):
✅ RICHTIG: "Gemäß [CELEX:32022R2554, Artikel 5] müssen..."
✅ RICHTIG: "Artikel 5 der DORA-Verordnung [CELEX:32022R2554, Artikel 5] fordert..."
❌ FALSCH: "Die DORA-Verordnung fordert..." (fehlende CELEX + Artikel-Nummer!)

PDF-Dokumente:
✅ RICHTIG: "Laut IST-Dokumentation [Policy_IKT.pdf, Seite 3, Zeile 45] ist..."
❌ FALSCH: "Die Dokumentation besagt..." (keine Quelle!)

Interne Requirements (aus Neo4j):
✅ RICHTIG: "Anforderung [REQ-001] aus [Requirements.pdf, Zeile 120] legt fest..."
✅ RICHTIG: "[SOLL-ANFORDERUNG: REQ-IKT-042] fordert..."

REGELN:
1. **Neo4j EUR-LEX Artikel** haben VORRANG als Rechtsgrundlage
2. **Bei JEDER Bewertung Quellen nennen**
3. **Keine Aussage ohne Quelle**
4. **Verwende die Emojis**: 📄 für PDF, 📋 für EUR-Lex, 🔗 für URLs

KONTEXT (enthält alle verfügbaren Quellen):
{context}

ANTWORT-FORMAT:
1. EUR-LEX Rechtsgrundlage: "[CELEX:..., Artikel X] - Titel"
2. Bewertung MIT exakter Quelle [Dokument, Seite, Zeile]
3. Lücken MIT Quellenangaben

🔴 WICHTIG: Wenn du KEINE passende Quelle findest:
"⚠️ Zu dieser Frage finde ich keine Informationen in den verfügbaren Quellen."
"""
    
    # Prepare messages for Ollama
    ollama_messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add conversation history
    for msg in request.messages:
        ollama_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Handle streaming vs non-streaming responses
    if request.stream:
        return await handle_streaming_response(
            request, ollama_messages, all_sources, source_list
        )
    else:
        return await handle_non_streaming_response(
            request, ollama_messages, all_sources, source_list
        )


async def handle_non_streaming_response(
    request: ChatCompletionRequest,
    ollama_messages: List[Dict[str, str]],
    all_sources: List[str],
    source_list: str
) -> ChatCompletionResponse:
    """Handle non-streaming chat completion"""
    
    used_model = request.model
    
    try:
        # Call Ollama API
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": used_model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens
                    }
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            msg = result.get("message", {}).get("content", "Error generating response")
            logger.info(f"LLM response length: {len(msg)} chars")
            
            # Validate: Were sources cited?
            has_celex = "[CELEX:" in msg or "CELEX:" in msg
            has_pdf = ".pdf" in msg
            has_req = "[REQ" in msg or "[SOLL-ANFORDERUNG" in msg
            
            if not (has_celex or has_pdf or has_req):
                logger.warning("⚠️ LLM response contains NO SOURCE CITATIONS!")
                msg += "\n\n" + "="*60
                msg += "\n⚠️ **WARNUNG: Fehlende Quellenzitate!**"
                msg += f"\n\n**Verfügbare Quellen:**\n{source_list}"
                msg += "\n" + "="*60
            else:
                logger.info(f"✓ LLM response contains sources: CELEX={has_celex}, PDF={has_pdf}, REQ={has_req}")
            
            return create_openai_response(msg, used_model)
    
    except httpx.HTTPError as e:
        logger.error(f"❌ Ollama API error: {e}")
        raise HTTPException(status_code=502, detail=f"LLM service error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def handle_streaming_response(
    request: ChatCompletionRequest,
    ollama_messages: List[Dict[str, str]],
    all_sources: List[str],
    source_list: str
) -> StreamingResponse:
    """Handle streaming chat completion with source validation"""
    
    used_model = request.model
    
    async def generate():
        full_response = ""
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": used_model,
                        "messages": ollama_messages,
                        "stream": True,
                        "options": {
                            "temperature": request.temperature,
                            "num_predict": request.max_tokens
                        }
                    }
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                import json
                                chunk = json.loads(line)
                                if "message" in chunk:
                                    content = chunk["message"].get("content", "")
                                    full_response += content
                                    
                                    # Stream in OpenAI format
                                    yield f"data: {json.dumps({
                                        'id': f'chatcmpl-{datetime.now().timestamp()}',
                                        'object': 'chat.completion.chunk',
                                        'created': int(datetime.now().timestamp()),
                                        'model': used_model,
                                        'choices': [{
                                            'index': 0,
                                            'delta': {'content': content},
                                            'finish_reason': None
                                        }]
                                    })}\n\n"
                            except json.JSONDecodeError:
                                continue
                
                # Log complete response
                logger.info(f"Streaming response complete: {len(full_response)} chars")
                
                # Validate sources
                has_celex = "[CELEX:" in full_response or "CELEX:" in full_response
                has_pdf = ".pdf" in full_response
                has_req = "[REQ" in full_response or "[SOLL-ANFORDERUNG" in full_response
                
                if not (has_celex or has_pdf or has_req):
                    logger.warning("⚠️ Streaming response contains NO SOURCE CITATIONS!")
                    
                    # Append warning
                    warning_msg = "\n\n" + "="*60
                    warning_msg += "\n⚠️ **WARNUNG: Fehlende Quellenzitate!**"
                    warning_msg += f"\n\n**Verfügbare Quellen:**\n{source_list}"
                    warning_msg += "\n" + "="*60
                    
                    import json
                    yield f"data: {json.dumps({
                        'id': f'chatcmpl-{datetime.now().timestamp()}',
                        'object': 'chat.completion.chunk',
                        'created': int(datetime.now().timestamp()),
                        'model': used_model,
                        'choices': [{
                            'index': 0,
                            'delta': {'content': warning_msg},
                            'finish_reason': None
                        }]
                    })}\n\n"
                else:
                    logger.info(f"✓ Streaming response contains sources: CELEX={has_celex}, PDF={has_pdf}, REQ={has_req}")
                
                # Send final chunk
                import json
                yield f"data: {json.dumps({
                    'id': f'chatcmpl-{datetime.now().timestamp()}',
                    'object': 'chat.completion.chunk',
                    'created': int(datetime.now().timestamp()),
                    'model': used_model,
                    'choices': [{
                        'index': 0,
                        'delta': {},
                        'finish_reason': 'stop'
                    }]
                })}\n\n"
                yield "data: [DONE]\n\n"
        
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}", exc_info=True)
            import json
            yield f"data: {json.dumps({
                'error': {
                    'message': str(e),
                    'type': 'server_error'
                }
            })}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


# Shutdown handler
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if neo4j_driver:
        neo4j_driver.close()
        logger.info("Neo4j driver closed")
    logger.info("Application shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
