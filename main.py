# test

import os, logging, time, hashlib, uuid, posixpath, asyncio, re
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field
from collections import deque
from urllib.parse import urljoin, urlparse
import html
import json
import requests
import pypdf
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from fastapi.responses import StreamingResponse
from neo4j import GraphDatabase

# BeautifulSoup Import mit Fallback
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup4 not available. Install with: pip install beautifulsoup4")

# ==================== CONFIGURATION ====================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "compliance2025")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "llama3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
TENANT_ID = os.getenv("TENANT_ID", "UNKNOWN")
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")
MAINT_TOKEN = os.getenv("MAINT_TOKEN", "")
KNOWLEDGE_PATH = "/app/data/knowledge"
SOLL_PATH = f"{KNOWLEDGE_PATH}/SOLL"
IST_PATH = f"{KNOWLEDGE_PATH}/IST"
COLLECTION_NAME = f"audit_{TENANT_ID}".lower().replace("-", "_")
VECTOR_SIZE = 768
DEEP_HEALTH = os.getenv("DEEP_HEALTH_REQUIRE_OLLAMA", "true").lower() == "true"
ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "http://localhost:3210").split(",")]
MAX_CHUNKS_PER_FILE = int(os.getenv("MAX_CHUNKS_PER_FILE", "200"))
MAX_FILE_SIZE_MB = 10
MAX_PAGES_PDF = 500
MAX_MESSAGES = 50
MAX_QUERY_CHARS = 8000

qdrant_client = None
neo4j_driver = None
service_status = "starting"

# ==================== LOGGING ====================
def _setup_logger(name, file_path):
    l = logging.getLogger(name)
    l.setLevel(logging.INFO)
    if getattr(l, "_configured", False):
        return l
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    l.addHandler(sh)
    if ENABLE_FILE_LOGGING:
        try:
            fh = RotatingFileHandler(file_path, maxBytes=10*1024*1024, backupCount=5)
            fh.setFormatter(fmt)
            l.addHandler(fh)
        except Exception as e:
            print(f"[{name}] WARNING: File logging disabled: {e}")
    l._configured = True
    return l

logger = _setup_logger("AI-AUDIT", "/app/logs/audit.log")
audit_events = _setup_logger("AI-AUDIT-EVENTS", "/app/logs/audit_events.log")

# ==================== FASTAPI APP ====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== HELPER FUNCTIONS ====================
def create_openai_response(content: str, model: str = DEFAULT_MODEL):
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

def get_canonical_path(raw_path: str) -> str:
    p = raw_path.replace("\\", "/")
    p = posixpath.normpath(p)
    return p

def get_embedding(text: str):
    if not text:
        return []
    for _ in range(3):
        try:
            res = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
                timeout=15
            )
            if res.status_code == 200:
                return res.json().get("embedding")
        except:
            time.sleep(1)
    return []

def generate_deterministic_uuid(tenant, canon_path, idx):
    return str(uuid.UUID(hex=hashlib.md5(f"{tenant}|{canon_path}|{idx}".encode()).hexdigest()))

# ==================== DATA CLASSES ====================
@dataclass
class RequirementNode:
    """Datenklasse für einen Requirements-Knoten"""
    id: str
    type: str
    number: str
    title: str
    content: str
    source: str
    line_number: int
    parent_id: Optional[str] = None
    level: int = 0
    references: List[str] = field(default_factory=list)

@dataclass
class EURLexNode:
    """Datenklasse für EUR-Lex Dokument-Knoten"""
    id: str
    node_type: str
    number: str
    title: str
    content: str
    html_content: str
    url: str
    celex: str
    parent_id: Optional[str] = None
    level: int = 0
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

# ==================== GENERAL EUR-LEX SCRAPER ====================
class EURLexScraper:
    """Generischer Scraper für EUR-Lex Dokumente"""
    BASE_URL = "https://eur-lex.europa.eu"
    
    # Mehrsprachige Struktur-Patterns
    STRUCTURE_PATTERNS = {
        # Deutsch
        "titel": r"TITEL\s+([IVX]+)",
        "kapitel": r"KAPITEL\s+([IVX]+)",
        "abschnitt": r"(?:Abschnitt|ABSCHNITT)\s+([\dIVX]+)",
        "artikel": r"Artikel\s+(\d+)",
        # Englisch
        "title": r"TITLE\s+([IVX]+)",
        "chapter": r"CHAPTER\s+([IVX]+)",
        "section": r"(?:Section|SECTION)\s+([\dIVX]+)",
        "article": r"Article\s+(\d+)",
    }
    
    HIERARCHY_LEVELS = {
        "document": 0,
        "titel": 1, "title": 1,
        "kapitel": 2, "chapter": 2,
        "abschnitt": 3, "section": 3,
        "artikel": 4, "article": 4
    }
    
    def __init__(self, base_url: str, follow_anchors: bool = True, max_depth: int = 1):
        """
        Args:
            base_url: Die EUR-Lex URL des Dokuments
            follow_anchors: Ob Anker-Links (#) zur gleichen Seite verfolgt werden sollen
            max_depth: Maximale Tiefe (1 = nur Hauptseite + direkte Anker)
        """
        self.base_url = base_url
        self.follow_anchors = follow_anchors
        self.max_depth = max_depth
        self.visited_urls = set()
        self.scraped_nodes = []
        self.url_queue = deque()
        self.celex = self._extract_celex(base_url)
        self.document_name = self._extract_document_name(base_url)
    
    def _extract_celex(self, url: str) -> str:
        """Extrahiert CELEX-Nummer aus URL"""
        match = re.search(r'CELEX:(\d+[A-Z]\d+)', url)
        if match:
            return match.group(1)
        match = re.search(r'uri=([^&]+)', url)
        if match:
            return match.group(1).replace(':', '_')
        return hashlib.md5(url.encode()).hexdigest()[:16]
    
    def _extract_document_name(self, url: str) -> str:
        """Extrahiert Dokumentnamen aus URL"""
        if "CELEX:" in url:
            match = re.search(r'CELEX:(\d+[A-Z]\d+)', url)
            return match.group(1) if match else "EUR-Lex Document"
        if "uri=" in url:
            match = re.search(r'uri=([^&]+)', url)
            uri = match.group(1) if match else "Unknown"
            return uri.split(':')[-1]
        return "EUR-Lex Document"
    
    def normalize_url(self, url: str) -> str:
        """Normalisiert URL ohne Fragment für Vergleich"""
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            base += f"?{parsed.query}"
        return base
    
    def is_same_document_page(self, url: str) -> bool:
        """Prüft ob URL zum gleichen Dokument gehört (nur mit anderem Anker)"""
        if not url:
            return False
        if url.startswith("#"):
            return True
        normalized_url = self.normalize_url(url)
        normalized_base = self.normalize_url(self.base_url)
        return normalized_url == normalized_base and self.celex in url
    
    def extract_article_structure(self, soup: BeautifulSoup, base_url: str) -> List[EURLexNode]:
        """Extrahiert Artikel aus EUR-Lex HTML"""
        nodes = []
        articles = soup.find_all('div', class_='eli-subdivision')
        logger.info(f"Found {len(articles)} eli-subdivision elements")
        
        for element in articles:
            try:
                title_elem = element.find('p', class_='oj-ti-art')
                if not title_elem:
                    continue
                
                title_text = title_elem.get_text(strip=True)
                
                # Mehrsprachige Artikel-Erkennung
                artikel_match = re.search(r'(?:Artikel|Article)\s+(\d+)', title_text, re.IGNORECASE)
                if not artikel_match:
                    continue
                
                artikel_nr = artikel_match.group(1)
                
                subtitle_elem = element.find('p', class_='oj-sti-art')
                subtitle = subtitle_elem.get_text(strip=True) if subtitle_elem else ""
                full_title = f"{title_text} - {subtitle}" if subtitle else title_text
                
                content_parts = []
                normal_paragraphs = element.find_all('p', class_='oj-normal')
                for p in normal_paragraphs:
                    p_text = p.get_text(strip=True)
                    if p_text and len(p_text) > 10:
                        content_parts.append(p_text)
                
                table_cells = element.find_all('td')
                for td in table_cells:
                    td_text = td.get_text(strip=True)
                    if td_text and len(td_text) > 10:
                        content_parts.append(td_text)
                
                content = "\n\n".join(content_parts)
                element_id = element.get('id', f'artikel_{artikel_nr}')
                
                node = EURLexNode(
                    id=f"{self.celex}::Article_{artikel_nr}",
                    node_type="article",
                    number=artikel_nr,
                    title=full_title[:500],
                    content=content[:10000],
                    html_content=str(element)[:15000],
                    url=f"{self.normalize_url(base_url)}#{element_id}",
                    celex=self.celex,
                    level=4,
                    metadata={
                        "source": "EUR-Lex",
                        "element_id": element_id,
                        "has_subtitle": bool(subtitle)
                    }
                )
                node.references = self._extract_references(content)
                nodes.append(node)
                logger.info(f"✓ Extracted Article {artikel_nr}: {full_title[:60]}")
            
            except Exception as e:
                logger.error(f"Error extracting article: {e}")
                continue
        
        return nodes
    
    def extract_hierarchical_structure(self, soup: BeautifulSoup, base_url: str) -> List[EURLexNode]:
        """Extrahiert Titel, Kapitel, Abschnitte"""
        nodes = []
        structure_elements = soup.find_all(['p', 'div'], class_=re.compile(r'(oj-ti-section|oj-title|oj-chapter)'))
        
        for elem in structure_elements:
            text = elem.get_text(strip=True)
            for struct_type, pattern in self.STRUCTURE_PATTERNS.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    number = match.group(1)
                    title = text[match.end():].strip()
                    elem_id = elem.get('id', f'{struct_type}_{number}')
                    
                    # Normalisiere Typ auf Englisch
                    normalized_type = struct_type
                    if struct_type in ["titel", "title"]:
                        normalized_type = "title"
                    elif struct_type in ["kapitel", "chapter"]:
                        normalized_type = "chapter"
                    elif struct_type in ["abschnitt", "section"]:
                        normalized_type = "section"
                    elif struct_type in ["artikel", "article"]:
                        normalized_type = "article"
                    
                    node = EURLexNode(
                        id=f"{self.celex}::{normalized_type}_{number}",
                        node_type=normalized_type,
                        number=number,
                        title=title[:500],
                        content=text[:2000],
                        html_content=str(elem)[:5000],
                        url=f"{self.normalize_url(base_url)}#{elem_id}",
                        celex=self.celex,
                        level=self.HIERARCHY_LEVELS.get(normalized_type, 99),
                        metadata={"source": "EUR-Lex"}
                    )
                    nodes.append(node)
                    logger.info(f"✓ Extracted {normalized_type} {number}: {title[:50]}")
                    break
        
        return nodes
    
    def _extract_references(self, text: str) -> List[str]:
        """Extrahiert Referenzen zu anderen Artikeln (mehrsprachig)"""
        references = []
        patterns = [
            r'(?:Artikel|Article)\s+(\d+)',
            r'gemäß\s+(?:Artikel|Article)\s+(\d+)',
            r'(?:according to|pursuant to)\s+(?:Artikel|Article)\s+(\d+)',
        ]
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ref = f"{self.celex}::Article_{match.group(1)}"
                if ref not in references:
                    references.append(ref)
        return references
    
    def scrape_page(self, url: str, depth: int = 0) -> List[EURLexNode]:
        """Scrapt eine einzelne EUR-Lex Seite"""
        normalized = self.normalize_url(url)
        if depth > self.max_depth or normalized in self.visited_urls:
            logger.info(f"Skipping {url} (depth={depth}, visited={normalized in self.visited_urls})")
            return []
        
        self.visited_urls.add(normalized)
        logger.info(f"Scraping (depth={depth}): {url}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'de-DE,de;q=0.9,en-US,en;q=0.8',
            }
            response = requests.get(url, timeout=30, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"HTTP {response.status_code} for {url}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            nodes = []
            
            # Dokument-Metadaten (nur bei depth=0)
            if depth == 0:
                doc_title_elem = soup.find('title')
                doc_title = doc_title_elem.get_text(strip=True) if doc_title_elem else self.document_name
                
                doc_node = EURLexNode(
                    id=f"{self.celex}::document",
                    node_type="document",
                    number=self.celex,
                    title=doc_title,
                    content=f"EUR-Lex Document {self.celex}",
                    html_content="",
                    url=url,
                    celex=self.celex,
                    level=0,
                    metadata={
                        "source": "EUR-Lex",
                        "scraped_at": time.time(),
                        "base_url": url
                    }
                )
                nodes.append(doc_node)
            
            # Extrahiere Struktur
            structure_nodes = self.extract_hierarchical_structure(soup, url)
            nodes.extend(structure_nodes)
            
            # Extrahiere Artikel
            article_nodes = self.extract_article_structure(soup, url)
            nodes.extend(article_nodes)
            
            # Folge Anker-Links
            if self.follow_anchors and depth < self.max_depth:
                links = soup.find_all('a', href=True)
                logger.info(f"Found {len(links)} links, checking for document anchors...")
                for link in links:
                    href = link.get('href')
                    full_url = urljoin(url, href)
                    if self.is_same_document_page(full_url):
                        normalized_link = self.normalize_url(full_url)
                        if normalized_link not in self.visited_urls:
                            logger.info(f"  → Following anchor: {full_url}")
                            self.url_queue.append((full_url, depth + 1))
            
            return nodes
        
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}", exc_info=True)
            return []
    
    def scrape_document(self) -> List[EURLexNode]:
        """Hauptmethode zum Scrapen eines EUR-Lex Dokuments"""
        self.url_queue.append((self.base_url, 0))
        all_nodes = []
        
        logger.info(f"=== Starting EUR-Lex scraping from: {self.base_url} ===")
        logger.info(f"CELEX: {self.celex}, Follow anchors: {self.follow_anchors}, Max depth: {self.max_depth}")
        
        while self.url_queue:
            url, depth = self.url_queue.popleft()
            nodes = self.scrape_page(url, depth)
            all_nodes.extend(nodes)
            
            # Rate limiting
            if self.url_queue:
                time.sleep(1)
        
        logger.info(f"✓ Scraping complete: {len(all_nodes)} nodes from {len(self.visited_urls)} unique pages")
        return all_nodes

# ==================== CROSS-DOCUMENT LINKING (VERBESSERT) ====================
def create_cross_document_links(celex: str):
    """
    Erstellt automatisch Verknüpfungen zwischen Dokumenten (VERBESSERTE VERSION)
    Analysiert das neue Dokument und sucht nach:
    1. Expliziten Referenzen (z.B. "gemäß Verordnung (EU) 2022/2554")
    2. Artikel-Referenzen zwischen Dokumenten
    3. Thematischen Verbindungen basierend auf Keywords
    """
    if not neo4j_driver:
        logger.warning("Neo4j driver not available for cross-linking")
        return None
    
    try:
        with neo4j_driver.session() as session:
            logger.info(f"=== Creating cross-document links for {celex} ===")
            stats = {
                "explicit_references": 0,
                "cross_article_references": 0,
                "thematic_links": 0,
                "document_relationships": 0
            }
            
            # 1. EXPLIZITE CELEX-REFERENZEN (VERBESSERT)
            logger.info("Step 1: Searching for explicit CELEX references...")
            
            # Hole alle anderen CELEX-Nummern
            other_docs = session.run(
                """
                MATCH (d:EURLexDocument {tenant: $tenant})
                WHERE d.celex <> $celex
                RETURN d.celex AS celex
                """,
                celex=celex,
                tenant=TENANT_ID
            )
            other_celex_list = [record["celex"] for record in other_docs]
            logger.info(f"Found {len(other_celex_list)} other documents: {other_celex_list}")
            
            # Suche nach jedem anderen CELEX im Content
            for other_celex in other_celex_list:
                # Extrahiere Nummer aus CELEX (z.B. "32022R2554" -> "2022/2554")
                celex_patterns = []
                # Pattern 1: Volle CELEX
                celex_patterns.append(other_celex)
                # Pattern 2: Jahr/Nummer Format (z.B. 2022/2554)
                match = re.search(r'(\d{4})[A-Z](\d+)', other_celex)
                if match:
                    year_num = f"{match.group(1)}/{match.group(2)}"
                    celex_patterns.append(year_num)
                
                logger.info(f"Searching for patterns: {celex_patterns}")
                
                result = session.run(
                    """
                    MATCH (new_node:EURLexNode {celex: $celex, tenant: $tenant})
                    MATCH (other_doc:EURLexDocument {celex: $other_celex, tenant: $tenant})
                    WHERE new_node.content IS NOT NULL
                      AND (
                        any(pattern IN $patterns WHERE new_node.content CONTAINS pattern)
                        OR any(pattern IN $patterns WHERE new_node.title CONTAINS pattern)
                      )
                    MERGE (new_node)-[r:CITES_DOCUMENT]->(other_doc)
                    ON CREATE SET r.created_at = datetime(),
                                  r.link_type = 'explicit_celex_reference',
                                  r.found_patterns = $patterns
                    RETURN count(r) AS links
                    """,
                    celex=celex,
                    other_celex=other_celex,
                    patterns=celex_patterns,
                    tenant=TENANT_ID
                )
                link_count = result.single()["links"]
                if link_count > 0:
                    logger.info(f"✓ Found {link_count} references to {other_celex}")
                    stats["explicit_references"] += link_count
            
            logger.info(f"Total explicit references: {stats['explicit_references']}")
            
            # 2. ARTIKEL-ZU-ARTIKEL REFERENZEN (VERBESSERT)
            logger.info("Step 2: Creating cross-article references...")
            
            result = session.run(
                """
                MATCH (new_article:EURLexNode {celex: $celex, type: 'article', tenant: $tenant})
                WHERE new_article.content IS NOT NULL
                // Hole alle Artikel aus ANDEREN Dokumenten
                MATCH (other_article:EURLexNode {type: 'article', tenant: $tenant})
                WHERE other_article.celex <> $celex
                // Prüfe ob die Artikel-Nummer im Content erwähnt wird
                WITH new_article, other_article
                WHERE toLower(new_article.content) CONTAINS 'artikel ' + other_article.number
                   OR toLower(new_article.content) CONTAINS 'article ' + other_article.number
                   OR toLower(new_article.title) CONTAINS 'artikel ' + other_article.number
                   OR toLower(new_article.title) CONTAINS 'article ' + other_article.number
                MERGE (new_article)-[r:CROSS_REFERENCES]->(other_article)
                ON CREATE SET r.created_at = datetime(),
                              r.link_type = 'article_reference',
                              r.referenced_article = other_article.number,
                              r.target_celex = other_article.celex
                RETURN count(DISTINCT r) AS links
                """,
                celex=celex,
                tenant=TENANT_ID
            )
            stats["cross_article_references"] = result.single()["links"]
            logger.info(f"✓ Created {stats['cross_article_references']} cross-article references")
            
            # 3. THEMATISCHE VERBINDUNGEN (mit mehr Keywords)
            logger.info("Step 3: Creating thematic links...")
            keywords = [
                # Englisch
                'risk management', 'operational resilience', 'third party', 'ict service',
                'incident reporting', 'testing', 'oversight', 'governance', 'business continuity',
                'threat intelligence', 'vulnerability', 'security', 'backup', 'recovery',
                'critical service', 'service provider', 'contractual arrangement',
                'digital operational', 'cyber threat', 'information security',
                # Deutsch
                'risikomanagement', 'betriebsstabilität', 'drittanbieter', 'ikt-dienst',
                'meldung von vorfällen', 'aufsicht', 'geschäftskontinuität',
                'bedrohungsinformationen', 'sicherheit', 'datensicherung', 'wiederherstellung',
                'kritische dienste', 'dienstleister', 'vertragliche vereinbarung',
                'digitale betriebsstabilität', 'cyberbedrohung', 'informationssicherheit'
            ]
            
            result = session.run(
                """
                MATCH (new_article:EURLexNode {celex: $celex, type: 'article', tenant: $tenant})
                WHERE new_article.content IS NOT NULL AND length(new_article.content) > 100
                // Finde Artikel in anderen Dokumenten
                MATCH (other_article:EURLexNode {type: 'article', tenant: $tenant})
                WHERE other_article.celex <> $celex
                  AND other_article.content IS NOT NULL
                  AND length(other_article.content) > 100
                // Zähle übereinstimmende Keywords
                WITH new_article, other_article,
                     [kw IN $keywords WHERE 
                      toLower(new_article.content) CONTAINS toLower(kw)
                      AND toLower(other_article.content) CONTAINS toLower(kw)] AS matching_keywords
                WITH new_article, other_article, matching_keywords
                WHERE size(matching_keywords) >= 3  // Mindestens 3 gemeinsame Keywords
                MERGE (new_article)-[r:RELATED_TO]->(other_article)
                ON CREATE SET r.created_at = datetime(),
                              r.link_type = 'thematic_similarity',
                              r.keyword_matches = size(matching_keywords),
                              r.matching_keywords = matching_keywords[..5]  // Erste 5 Keywords
                RETURN count(DISTINCT r) AS links
                """,
                celex=celex,
                keywords=keywords,
                tenant=TENANT_ID
            )
            stats["thematic_links"] = result.single()["links"]
            logger.info(f"✓ Created {stats['thematic_links']} thematic links")
            
            # 4. BIDIREKTIONALE VERLINKUNG
            logger.info("Step 4: Creating bidirectional links...")
            
            result = session.run(
                """
                // Finde alle Artikel aus ALTEN Dokumenten, die auf das NEUE verweisen könnten
                MATCH (old_article:EURLexNode {type: 'article', tenant: $tenant})
                WHERE old_article.celex <> $celex
                  AND old_article.content IS NOT NULL
                // Finde Artikel im NEUEN Dokument
                MATCH (new_article:EURLexNode {celex: $celex, type: 'article', tenant: $tenant})
                // Prüfe ob alte Artikel das neue Dokument erwähnen
                WITH old_article, new_article
                WHERE toLower(old_article.content) CONTAINS 'artikel ' + new_article.number
                   OR toLower(old_article.content) CONTAINS 'article ' + new_article.number
                MERGE (old_article)-[r:CROSS_REFERENCES]->(new_article)
                ON CREATE SET r.created_at = datetime(),
                              r.link_type = 'article_reference',
                              r.referenced_article = new_article.number,
                              r.target_celex = new_article.celex
                RETURN count(DISTINCT r) AS links
                """,
                celex=celex,
                tenant=TENANT_ID
            )
            backward_links = result.single()["links"]
            logger.info(f"✓ Created {backward_links} backward references")
            stats["cross_article_references"] += backward_links
            
            # 5. DOKUMENT-ZU-DOKUMENT BEZIEHUNGEN
            logger.info("Step 5: Creating document-level relationships...")
            result = session.run(
                """
                MATCH (new_doc:EURLexDocument {celex: $celex, tenant: $tenant})
                MATCH (other_doc:EURLexDocument {tenant: $tenant})
                WHERE other_doc.celex <> $celex
                // Zähle alle Verbindungen zwischen den Dokumenten
                OPTIONAL MATCH (new_node:EURLexNode {celex: $celex, tenant: $tenant})
                  -[rel:CITES_DOCUMENT|CROSS_REFERENCES|RELATED_TO]->
                  (other_node:EURLexNode {celex: other_doc.celex, tenant: $tenant})
                // Zähle auch Rück-Verbindungen
                OPTIONAL MATCH (other_existing:EURLexNode {celex: other_doc.celex, tenant: $tenant})
                  -[back_rel:CITES_DOCUMENT|CROSS_REFERENCES|RELATED_TO]->
                  (new_existing:EURLexNode {celex: $celex, tenant: $tenant})
                WITH new_doc, other_doc,
                     count(DISTINCT rel) + count(DISTINCT back_rel) AS total_connections
                WHERE total_connections > 0
                MERGE (new_doc)-[r:RELATED_REGULATION]-(other_doc)
                SET r.connection_strength = total_connections,
                    r.last_analyzed = datetime()
                RETURN count(DISTINCT r) AS links, sum(total_connections) AS total_conn
                """,
                celex=celex,
                tenant=TENANT_ID
            )
            result_single = result.single()
            stats["document_relationships"] = result_single["links"]
            total_connections = result_single["total_conn"] or 0
            logger.info(f"✓ Created {stats['document_relationships']} document relationships with {total_connections} total connections")
            
            # Gesamtstatistik
            total_links = sum(stats.values())
            logger.info(f"=== Cross-linking complete: {total_links} total links created ===")
            logger.info(f"Details: {stats}")
            
            return stats
    
    except Exception as e:
        logger.error(f"Error creating cross-document links: {e}", exc_info=True)
        import traceback
        logger.error(traceback.format_exc())
        return None

def rebuild_all_cross_document_links():
    """
    Baut ALLE Cross-Document Links neu auf
    Nützlich wenn Links fehlen oder neu berechnet werden sollen
    """
    if not neo4j_driver:
        logger.warning("Neo4j driver not available")
        return None
    
    try:
        with neo4j_driver.session() as session:
            logger.info("=== Rebuilding ALL cross-document links ===")
            
            # Lösche alte Links
            logger.info("Deleting old cross-document links...")
            session.run(
                """
                MATCH ()-[r:CITES_DOCUMENT|CROSS_REFERENCES|RELATED_TO|RELATED_REGULATION]->()
                DELETE r
                """
            )
            
            # Hole alle CELEX-Nummern
            result = session.run(
                """
                MATCH (d:EURLexDocument {tenant: $tenant})
                RETURN d.celex AS celex
                ORDER BY d.ingested_at
                """,
                tenant=TENANT_ID
            )
            celex_list = [record["celex"] for record in result]
            logger.info(f"Found {len(celex_list)} documents: {celex_list}")
            
            # Verlinke jedes Dokument
            total_stats = {
                "explicit_references": 0,
                "cross_article_references": 0,
                "thematic_links": 0,
                "document_relationships": 0
            }
            
            for celex in celex_list:
                logger.info(f"Processing {celex}...")
                stats = create_cross_document_links(celex)
                if stats:
                    for key in total_stats:
                        total_stats[key] += stats.get(key, 0)
            
            logger.info(f"=== Rebuild complete: {total_stats} ===")
            return total_stats
    
    except Exception as e:
        logger.error(f"Error rebuilding links: {e}", exc_info=True)
        return None

# ==================== EUR-LEX STORAGE FUNCTIONS ====================
def store_eurlex_nodes_in_neo4j(nodes: List[EURLexNode], source_url: str):
    """Speichert EUR-Lex Nodes in Neo4j"""
    if not neo4j_driver:
        logger.warning("Neo4j driver not available")
        return
    
    if not nodes:
        logger.warning("No nodes to store")
        return
    
    celex = nodes[0].celex
    doc_title = next((n.title for n in nodes if n.node_type == "document"), f"EUR-Lex {celex}")
    
    try:
        with neo4j_driver.session() as session:
            # Erstelle Haupt-Dokument
            session.run(
                """
                MERGE (d:EURLexDocument {celex: $celex, tenant: $tenant})
                SET d.url = $url,
                    d.title = $title,
                    d.ingested_at = datetime(),
                    d.node_count = $count,
                    d.source = 'EUR-Lex'
                """,
                celex=celex,
                tenant=TENANT_ID,
                url=source_url,
                title=doc_title,
                count=len(nodes)
            )
            
            # Erstelle alle Nodes
            for node in nodes:
                session.run(
                    """
                    MERGE (n:EURLexNode {id: $id, tenant: $tenant})
                    SET n.type = $type,
                        n.number = $number,
                        n.title = $title,
                        n.content = $content,
                        n.html_content = $html_content,
                        n.url = $url,
                        n.celex = $celex,
                        n.level = $level,
                        n.metadata = $metadata,
                        n.updated_at = datetime()
                    """,
                    id=node.id,
                    tenant=TENANT_ID,
                    type=node.node_type,
                    number=node.number,
                    title=node.title,
                    content=node.content[:5000],
                    html_content=node.html_content[:8000],
                    url=node.url,
                    celex=node.celex,
                    level=node.level,
                    metadata=json.dumps(node.metadata)
                )
            
            # Verknüpfe mit Haupt-Dokument
            session.run(
                """
                MATCH (d:EURLexDocument {celex: $celex, tenant: $tenant})
                MATCH (n:EURLexNode {celex: $celex, tenant: $tenant})
                MERGE (d)-[:CONTAINS]->(n)
                """,
                celex=celex,
                tenant=TENANT_ID
            )
            
            # Erstelle Referenz-Beziehungen (innerhalb des Dokuments)
            for node in nodes:
                if node.references:
                    for ref_id in node.references:
                        try:
                            session.run(
                                """
                                MATCH (source:EURLexNode {id: $source_id, tenant: $tenant})
                                MATCH (target:EURLexNode {id: $target_id, tenant: $tenant})
                                MERGE (source)-[:REFERENCES]->(target)
                                """,
                                source_id=node.id,
                                target_id=ref_id,
                                tenant=TENANT_ID
                            )
                        except:
                            pass
            
            logger.info(f"✓ Neo4j: Stored {len(nodes)} EUR-Lex nodes for {celex}")
    
    except Exception as e:
        logger.error(f"Neo4j error: {e}", exc_info=True)

def create_embeddings_for_eurlex_nodes(nodes: List[EURLexNode]):
    """Erstellt Embeddings für EUR-Lex Nodes in Qdrant"""
    if not qdrant_client:
        logger.warning("Qdrant client not available")
        return
    
    points = []
    for idx, node in enumerate(nodes):
        embed_text = f"{node.title}\n\n{node.content}"
        if len(embed_text.strip()) < 20:
            continue
        
        vec = get_embedding(embed_text[:4000])
        if vec and len(vec) == VECTOR_SIZE:
            point_id = generate_deterministic_uuid(TENANT_ID, node.id, 0)
            points.append(qmodels.PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "source": f"EUR-Lex {node.celex}",
                    "node_id": node.id,
                    "node_type": node.node_type,
                    "number": node.number,
                    "title": node.title,
                    "content": node.content[:2000],
                    "text": node.content[:2000],
                    "url": node.url,
                    "celex": node.celex,
                    "level": node.level,
                    "doc_type": "SOLL",
                    "tenant": TENANT_ID,
                    "path": f"/knowledge/SOLL/EUR-Lex/{node.celex}/{node.id}",
                    "topic": f"EUR-Lex {node.celex}"
                }
            ))
    
    if points:
        try:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )
            logger.info(f"✓ Qdrant: Created {len(points)} embeddings")
        except Exception as e:
            logger.error(f"Qdrant error: {e}")

# ==================== NEO4J QUERY FUNCTIONS (ERWEITERT MIT QUELLEN) ====================
def query_neo4j_requirements(query: str, limit: int = 5):
    """
    Erweiterte Query-Funktion für Requirements UND EUR-Lex Nodes mit detaillierten Quellen
    """
    if not neo4j_driver:
        return ""
    
    try:
        keywords = re.findall(r'\b[a-zäöüß]{4,}\b', query.lower())
        if not keywords:
            return ""
        
        with neo4j_driver.session() as session:
            context = ""
            
            # 1. Suche in klassischen Requirements (aus PDFs)
            result_requirements = session.run(
                """
                MATCH (r:Requirement {tenant: $tenant})
                WHERE ANY(kw IN $keywords WHERE toLower(r.content) CONTAINS kw OR toLower(r.title) CONTAINS kw)
                OPTIONAL MATCH (r)-[:HAS_SUBREQUIREMENT]->(sub:Requirement)
                RETURN r.id AS id, r.title AS title, r.content AS content,
                       r.source AS source, r.type AS type, r.line_number AS line_number,
                       collect(sub.id) AS subreqs
                LIMIT $limit
                """,
                tenant=TENANT_ID, keywords=keywords, limit=limit
            )
            
            for record in result_requirements:
                subreqs = ", ".join(record["subreqs"]) if record["subreqs"] else "keine"
                line_info = f", Zeile {record['line_number']}" if record.get('line_number') else ""
                
                context += f"[SOLL-ANFORDERUNG: {record['id']}]\n"
                context += f"Quelle: {record['source']}{line_info}\n"
                context += f"Titel: {record['title']}\n"
                context += f"Inhalt: {record['content']}\n"
                context += f"Unteranforderungen: {subreqs}\n"
                context += "[ENDE ANFORDERUNG]\n\n"
            
            # 2. Suche in EUR-Lex Nodes mit Cross-Document References UND verwandten Dokumenten
            result_eurlex = session.run(
                """
                MATCH (d:EURLexNode {tenant: $tenant})
                WHERE ANY(kw IN $keywords WHERE toLower(d.content) CONTAINS kw OR toLower(d.title) CONTAINS kw)
                // Finde Referenzen innerhalb des Dokuments
                OPTIONAL MATCH (d)-[:REFERENCES]->(ref:EURLexNode {type: 'article'})
                // Finde Cross-Document Referenzen
                OPTIONAL MATCH (d)-[cross:CROSS_REFERENCES|RELATED_TO]->(other:EURLexNode)
                // Finde das übergeordnete Dokument und seine Beziehungen
                OPTIONAL MATCH (d)<-[:CONTAINS]-(doc:EURLexDocument)-[doc_rel:RELATED_REGULATION]-(related_doc:EURLexDocument)
                WHERE doc_rel.connection_strength > 0
                RETURN d.id AS id, d.number AS number, d.title AS title,
                       d.content AS content, d.type AS type, d.url AS url,
                       d.celex AS celex,
                       collect(DISTINCT ref.number) AS internal_refs,
                       collect(DISTINCT {celex: other.celex, number: other.number, type: type(cross), title: other.title}) AS cross_refs,
                       collect(DISTINCT {celex: related_doc.celex, title: related_doc.title, strength: doc_rel.connection_strength}) AS related_docs
                LIMIT $limit
                """,
                tenant=TENANT_ID, keywords=keywords, limit=limit
            )
            
            for record in result_eurlex:
                internal_refs = ", ".join([r for r in record["internal_refs"] if r]) if record["internal_refs"] else "keine"
                celex_info = record.get('celex', 'N/A')
                
                context += f"[EUR-LEX {celex_info} - {record['type'].upper()}: {record['number']}]\n"
                context += f"Quelle: CELEX:{celex_info}, Artikel {record['number']}\n"
                context += f"Titel: {record['title']}\n"
                context += f"Inhalt: {record['content'][:1000]}\n"
                context += f"URL: {record['url']}\n"
                context += f"Interne Verweise: {internal_refs}\n"
                
                # Cross-Document Referenzen
                if record['cross_refs'] and any(r['celex'] for r in record['cross_refs']):
                    cross_info = []
                    for ref in record['cross_refs']:
                        if ref['celex']:
                            cross_info.append(f"→ CELEX:{ref['celex']} Art. {ref['number']} ({ref['type']}) - {ref.get('title', 'N/A')[:50]}")
                    if cross_info:
                        context += f"Dokument-übergreifende Verweise:\n" + "\n".join(cross_info) + "\n"
                
                # Verwandte Dokumente
                if record['related_docs'] and any(r['celex'] for r in record['related_docs']):
                    related_info = []
                    for rel in record['related_docs']:
                        if rel['celex']:
                            related_info.append(f"↔ CELEX:{rel['celex']} - {rel.get('title', 'N/A')[:60]} (Stärke: {rel['strength']})")
                    if related_info:
                        context += f"Verwandte Dokumente:\n" + "\n".join(related_info) + "\n"
                
                context += "[ENDE EUR-LEX ELEMENT]\n\n"
            
            return context
    
    except Exception as e:
        logger.error(f"Neo4j Query Error: {e}")
        return ""

# ==================== STARTUP/SHUTDOWN ====================
@app.on_event("startup")
async def startup_event():
    global qdrant_client, neo4j_driver, service_status
    logger.info("Startup: Initializing Qdrant + Neo4j...")
    
    # Qdrant Initialization
    for i in range(15):
        try:
            c = QdrantClient(host=QDRANT_HOST, port=6333, timeout=5)
            c.get_collections()
            vec = get_embedding("test")
            if not vec:
                logger.warning("Embedding failed. Retrying...")
            elif len(vec) != VECTOR_SIZE:
                logger.critical(f"DIMENSION MISMATCH: Got {len(vec)}, Expected {VECTOR_SIZE}.")
                service_status = "degraded"
                return
            else:
                if not c.collection_exists(COLLECTION_NAME):
                    c.create_collection(
                        collection_name=COLLECTION_NAME,
                        vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE)
                    )
                qdrant_client = c
                logger.info("Qdrant Ready.")
                break
        except Exception as e:
            logger.warning(f"Qdrant retry {i}: {e}")
            await asyncio.sleep(3)
    
    if not qdrant_client:
        logger.critical("Qdrant Startup Failed.")
        service_status = "unhealthy"
        return
    
    # Neo4j Initialization
    for i in range(10):
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            driver.verify_connectivity()
            
            with driver.session() as session:
                # Bestehende Constraints
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE (d.name, d.tenant) IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Requirement) REQUIRE (r.id, r.tenant) IS UNIQUE")
                session.run("CREATE INDEX IF NOT EXISTS FOR (r:Requirement) ON (r.tenant)")
                
                # Neue Constraints für EUR-Lex
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:EURLexDocument) REQUIRE (d.celex, d.tenant) IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:EURLexNode) REQUIRE (n.id, n.tenant) IS UNIQUE")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:EURLexNode) ON (n.tenant)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:EURLexNode) ON (n.celex)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:EURLexNode) ON (n.type)")
            
            neo4j_driver = driver
            logger.info("Neo4j Ready.")
            service_status = "healthy"
            return
        
        except Exception as e:
            logger.warning(f"Neo4j retry {i}: {e}")
            await asyncio.sleep(3)
    
    logger.error("Neo4j Startup Failed - Running in degraded mode")
    service_status = "degraded"

@app.on_event("shutdown")
async def shutdown_event():
    if neo4j_driver:
        neo4j_driver.close()

# ==================== PYDANTIC MODELS ====================
class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatReq(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False

class PurgeReq(BaseModel):
    path: str

class EURLexIngestReq(BaseModel):
    url: str
    follow_anchors: Optional[bool] = True
    max_depth: Optional[int] = 1

# ==================== ENDPOINTS ====================
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": DEFAULT_MODEL,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local"
        }]
    }

@app.get("/health")
def health(deep: bool = Query(False)):
    if service_status == "unhealthy" or qdrant_client is None:
        raise HTTPException(503, f"Status: {service_status}")
    
    base = {
        "status": service_status,
        "tenant": TENANT_ID,
        "neo4j": "connected" if neo4j_driver else "unavailable"
    }
    
    try:
        base["doc_count"] = qdrant_client.get_collection(COLLECTION_NAME).points_count
    except:
        base["doc_count"] = "unknown"
    
    if neo4j_driver:
        try:
            with neo4j_driver.session() as session:
                result = session.run("MATCH (r:Requirement {tenant: $tenant}) RETURN count(r) AS cnt", tenant=TENANT_ID)
                base["neo4j_requirements"] = result.single()["cnt"]
                
                result = session.run("MATCH (n:EURLexNode {tenant: $tenant}) RETURN count(n) AS cnt", tenant=TENANT_ID)
                base["neo4j_eurlex_nodes"] = result.single()["cnt"]
                
                result = session.run("MATCH ()-[r:RELATED_REGULATION]->() RETURN count(r) AS cnt")
                base["cross_document_links"] = result.single()["cnt"]
        except:
            base["neo4j_requirements"] = "error"
    
    if deep and DEEP_HEALTH:
        try:
            if requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3).status_code != 200:
                raise Exception()
        except:
            raise HTTPException(503, "Deep check failed")
    
    return base

@app.post("/v1/chat/completions")
async def chat(req: ChatReq):
    """
    Chat-Endpoint mit EUR-Lex Integration, Cross-Document Awareness und präzisen Quellenzitaten
    """
    if not req.messages or len(req.messages) > MAX_MESSAGES:
        return create_openai_response("Fehler: Zu viele Nachrichten oder keine Nachrichten.")
    
    used_model = req.model or DEFAULT_MODEL
    
    if qdrant_client is None:
        return create_openai_response("Systemfehler: DB nicht verbunden.", model=used_model)
    
    # Extrahiere User Query
    user_query = ""
    try:
        lc = req.messages[-1].content
        if isinstance(lc, str):
            user_query = lc
        elif isinstance(lc, list):
            user_query = " ".join([i.get("text", "") for i in lc if isinstance(i, dict) and i.get("type") == "text"])
    except:
        pass
    
    user_query = user_query.strip()
    
    if not user_query or len(user_query) > MAX_QUERY_CHARS:
        return create_openai_response("Fehler: Ungültige Anfrage.", model=used_model)
    
    # Kontext aufbauen
    context = ""
    
    # 1. Neo4j: Strukturierte SOLL-Anforderungen + EUR-Lex Nodes mit Cross-References
    neo4j_context = query_neo4j_requirements(user_query, limit=5)
    if neo4j_context:
        context += "=== STRUKTURIERTE SOLL-ANFORDERUNGEN & EUR-LEX DOKUMENTE (Neo4j) ===\n" + neo4j_context + "\n"
    
    # 2. Qdrant: Semantische Suche über alle Dokumente
    vec = get_embedding(user_query)
    if vec:
        try:
            hits = qdrant_client.search(collection_name=COLLECTION_NAME, query_vector=vec, limit=5)
            qdrant_context = ""
            
            for h in hits:
                try:
                    p = h.payload or {}
                    txt = p.get("text")
                    if txt:
                        doc_type = "SOLL" if p.get("path", "").find("/SOLL/") >= 0 else "IST"
                        
                        # Prüfe ob es ein EUR-Lex Node aus Qdrant ist
                        if p.get("node_type"):
                            celex = p.get("celex", "N/A")
                            qdrant_context += f"[EUR-LEX {celex} - {p.get('node_type').upper()}: {p.get('number', 'N/A')}]\n"
                            qdrant_context += f"Quelle: CELEX:{celex}, Artikel {p.get('number', 'N/A')}\n"
                            qdrant_context += f"Titel: {p.get('title', 'N/A')}\n"
                            qdrant_context += f"{txt}\n"
                            qdrant_context += f"URL: {p.get('url', 'N/A')}\n"
                            qdrant_context += "[ENDE EUR-LEX ELEMENT]\n\n"
                        else:
                            # Klassisches Dokument
                            source_info = p.get('source', 'N/A')
                            page_info = f", Seite {p.get('page_number')}" if p.get('page_number') else ""
                            
                            qdrant_context += f"[{doc_type}-DOKUMENT: {source_info}{page_info}]\n"
                            qdrant_context += f"Quelle: {source_info}{page_info}\n"
                            qdrant_context += f"Thema: {p.get('topic', 'N/A')}\n"
                            qdrant_context += f"{txt}\n"
                            qdrant_context += "[ENDE DOKUMENT]\n\n"
                except:
                    continue
            
            if qdrant_context:
                context += "=== SEMANTISCHE TREFFER (Qdrant) ===\n" + qdrant_context
        
        except Exception as e:
            logger.error(f"Qdrant fetch error: {e}")
    
    # Kein Kontext gefunden?
    if not context.strip():
        return create_openai_response("Keine Informationen im Kontext gefunden.", model=used_model)
    
    # System-Prompt mit EUR-Lex Integration, Cross-Document Awareness und QUELLENZITATION
    sys = f"""Du bist ein Compliance-Auditor für {TENANT_ID} mit Spezialisierung auf EU-Regulierung.

AUFGABE: Prüfe IST-Dokumente gegen SOLL-Anforderungen aus EU-Verordnungen und internen Richtlinien.

DATENQUELLEN:
1. **EUR-LEX DOKUMENTE** - Offizielle EU-Rechtsgrundlagen (Neo4j EURLexNode)
   - DORA (EU) 2022/2554, RTS und andere Verordnungen
   - Hierarchisch organisiert (Titel → Kapitel → Artikel)
   - **DOKUMENT-ÜBERGREIFENDE VERKNÜPFUNGEN**: Artikel können auf andere Verordnungen verweisen
   - Drei Arten von Verknüpfungen:
     * CITES_DOCUMENT: Explizite Erwähnung anderer Verordnungen
     * CROSS_REFERENCES: Direkte Artikel-zu-Artikel Referenzen zwischen Dokumenten
     * RELATED_TO: Thematisch verwandte Artikel in verschiedenen Dokumenten

2. **Interne SOLL-Anforderungen** - aus PDF-Dokumenten (Neo4j Requirement)
   - Organisationsspezifische Anforderungen
   - Verknüpft mit EU-Artikeln

3. **IST-Dokumentation** - aktuelle Umsetzung (Qdrant)
   - Bestehende Policies, Prozesse, Dokumentation

WICHTIG: QUELLENZITATION
==========================
**JEDE AUSSAGE** muss mit **PRÄZISEN QUELLEN** belegt werden:

EUR-LEX Artikel:
- Format: [CELEX:32022R2554, Artikel 5]
- Immer CELEX-Nummer UND Artikel-Nummer angeben
- Bei Querverweisen: "→ CELEX:32022R1234, Artikel 12"

PDF-Dokumente:
- Format: [Dokument.pdf, Seite 5] oder [Dokument.pdf, Zeile 120]
- Wenn verfügbar: Zeilen- UND Seitennummer

Interne Requirements:
- Format: [Requirement-ID: REQ-001]

REGELN:
1. **EUR-LEX Artikel** haben VORRANG als Rechtsgrundlage
   - Zitiere als: [CELEX:32022R2554, Artikel 5] mit Titel
   - Nenne die genaue CELEX-Nummer und Artikel-Nummer

2. **Dokument-übergreifende Zusammenhänge AKTIV nutzen**:
   - Wenn ein Artikel auf andere Dokumente verweist, NENNE diese Verbindungen
   - Zeige auf, wie verschiedene Verordnungen zusammenhängen
   - Beispiel: "DORA Artikel 28 verweist auf RTS Artikel 5 (CROSS_REFERENCES)"

3. **Interne SOLL-Anforderungen** ergänzen EUR-LEX
   - Zitiere als: [Requirement-ID: REQ-001] oder [Dokument.pdf, Seite X, Zeile Y]

4. **IST-Dokumente** prüfen gegen EUR-LEX + SOLL
   - Zitiere als: [Dokument.pdf, Seite X] oder [Dokument.pdf, Zeile Y]
   - Benenne KONKRET: Was fehlt? Was ist unvollständig?

5. **Hierarchische Beziehungen** beachten:
   - Bei EUR-LEX: Übergeordnete Kapitel/Titel nennen
   - Querverweise zu anderen Artikeln aufzeigen
   - **Dokument-übergreifende Referenzen hervorheben**

6. **Compliance-Bewertung**:
   - ✅ Konform: IST erfüllt Anforderung vollständig
   - ⚠️ Teilweise: IST erfüllt nur unvollständig
   - ❌ Nicht konform: Anforderung fehlt komplett
   - ❓ Unklar: Nicht genug IST-Dokumentation vorhanden

7. **IMMER Quellen zitieren**: Jede Aussage muss mit [CELEX:..., Artikel X], [Dokument.pdf, Seite Y] oder [REQ-ID] belegt sein

KONTEXT:
{context}

ANTWORT-FORMAT:
- Nenne zuerst den relevanten EUR-LEX Artikel mit **[CELEX:..., Artikel X]** und Titel
- **Falls vorhanden: Nenne Verknüpfungen zu anderen Dokumenten**
- Erkläre die Anforderung kurz
- Bewerte die IST-Umsetzung (✅/⚠️/❌/❓) mit **präziser Quellenzitation [Dokument.pdf, Seite X, Zeile Y]**
- Nenne konkrete Lücken oder Verbesserungen
- Verweise auf verwandte Artikel falls relevant (auch dokument-übergreifend)

BEISPIEL FÜR KORREKTE ZITATION:
"Gemäß [CELEX:32022R2554, Artikel 5] müssen Finanzunternehmen ein IKT-Risikomanagement-Framework implementieren. 
Dieser Artikel verweist auf [→ CELEX:32022R1234, Artikel 12] für technische Details.
Die IST-Dokumentation [Policy_IKT.pdf, Seite 3, Zeile 45-52] beschreibt das Framework, jedoch fehlt die Umsetzung von Punkt (c) ⚠️."
"""
    
    # Non-Streaming Response
    if not req.stream:
        try:
            r = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": used_model,
                    "messages": [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user_query},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=180,
            )
            
            if r.status_code != 200:
                return create_openai_response(f"LLM Fehler: {r.text[:100]}", model=used_model)
            
            msg = r.json().get("message", {}).get("content", "Error")
            
            # Warnung wenn keine Quellen zitiert wurden
            if "[CELEX:" not in msg and "[Requirement" not in msg and ".pdf" not in msg:
                msg += "\n\n⚠️ (WARNUNG: Fehlende Quellenzitate! Bitte konkrete CELEX-Nummern, Artikel, Dokumente mit Seiten-/Zeilennummern angeben.)"
            
            return create_openai_response(msg, model=used_model)
        
        except Exception as e:
            logger.error(f"Chat Crash: {e}")
            return create_openai_response(f"Kritischer Fehler: {e}", model=used_model)
    
    # Streaming Response
    async def stream_generator():
        created_ts = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        
        try:
            with requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": used_model,
                    "messages": [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user_query},
                    ],
                    "stream": True,
                    "options": {"temperature": 0.0},
                },
                stream=True,
                timeout=180,
            ) as r:
                
                if r.status_code != 200:
                    err_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": used_model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": f"LLM Fehler: {r.text[:100]}"},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    line = raw.strip()
                    
                    try:
                        data = json.loads(line)
                    except:
                        continue
                    
                    msg_part = ""
                    done = False
                    
                    if isinstance(data, dict):
                        msg = data.get("message") or {}
                        msg_part = msg.get("content") or ""
                        done = bool(data.get("done"))
                    
                    if msg_part:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": used_model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": msg_part},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    
                    if done:
                        final_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": used_model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        break
        
        except Exception as e:
            logger.error(f"Streaming Chat Crash: {e}")
            err_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": used_model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": f"Kritischer Fehler: {e}"},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(err_chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.post("/maintenance/purge")
async def purge_file(req: PurgeReq, x_token: str = Header(None, alias="X-Access-Token")):
    if x_token != MAINT_TOKEN:
        raise HTTPException(401, "Unauthorized")
    
    if qdrant_client is None:
        raise HTTPException(503, "DB unavailable")
    
    raw_path = get_canonical_path(req.path)
    if not raw_path.startswith(KNOWLEDGE_PATH):
        if "/knowledge/" in raw_path:
            parts = raw_path.split("/knowledge/", 1)
            raw_path = f"{KNOWLEDGE_PATH}/{parts[1]}"
        else:
            raise HTTPException(400, "Invalid path format")
    
    target_path = get_canonical_path(raw_path)
    
    if target_path != KNOWLEDGE_PATH and not target_path.startswith(KNOWLEDGE_PATH + "/"):
        audit_events.warning(f"TRAVERSAL BLOCKED {target_path}")
        raise HTTPException(403, "Denied")
    
    try:
        total_deleted = 0
        while True:
            points, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=qmodels.Filter(must=[
                    qmodels.FieldCondition(key="path", match=qmodels.MatchValue(value=target_path))
                ]),
                limit=500,
                offset=None,
                with_payload=False,
                with_vectors=False
            )
            
            if not points:
                break
            
            ids = [p.id for p in points]
            qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=qmodels.PointIdsList(points=ids))
            total_deleted += len(ids)
        
        neo4j_deleted = 0
        if neo4j_driver and "/SOLL/" in target_path:
            filename = os.path.basename(target_path)
            with neo4j_driver.session() as session:
                result = session.run(
                    "MATCH (d:Document {name: $name, tenant: $tenant}) OPTIONAL MATCH (d)-[:CONTAINS]->(r:Requirement) DETACH DELETE d, r RETURN count(r) AS deleted",
                    name=filename,
                    tenant=TENANT_ID
                )
                neo4j_deleted = result.single()["deleted"] or 0
        
        audit_events.info(f"PURGE_OK {target_path} qdrant={total_deleted} neo4j={neo4j_deleted}")
        return {
            "status": "purged",
            "path": target_path,
            "qdrant_count": total_deleted,
            "neo4j_count": neo4j_deleted
        }
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/ingest")
async def ingest(x_token: str = Header(None, alias="X-Access-Token")):
    if x_token != INGEST_TOKEN:
        raise HTTPException(401, "Unauthorized")
    
    if qdrant_client is None:
        raise HTTPException(503, "DB unavailable")
    
    logger.info("INGEST START")
    processed = 0
    stats = {"skipped": 0, "truncated": 0, "partial_fail": 0, "abort": 0, "neo4j_reqs": 0}
    
    for doc_type in ["SOLL", "IST"]:
        base_path = SOLL_PATH if doc_type == "SOLL" else IST_PATH
        logger.info(f"Processing {doc_type} at {base_path}")
        
        if not os.path.exists(base_path):
            logger.warning(f"Path missing: {base_path}")
            continue
        
        for root, _, files in os.walk(base_path):
            topic = os.path.basename(root)
            logger.info(f"Scanning {root} - Found {len(files)} files")
            
            for file in files:
                logger.info(f"Processing file: {file}")
                os_path = os.path.join(root, file)
                canon_path = get_canonical_path(os_path)
                
                try:
                    file_size = os.path.getsize(os_path)
                    logger.info(f"File size: {file_size} bytes")
                    
                    if file_size > (MAX_FILE_SIZE_MB * 1024 * 1024):
                        logger.warning(f"Skipped (too large): {file}")
                        stats["skipped"] += 1
                        continue
                    
                    content = ""
                    _, file_ext = os.path.splitext(file)
                    file_ext = file_ext.lower().strip()
                    logger.info(f"Detected extension: '{file_ext}'")
                    
                    # PDF-Handling
                    if file_ext == ".pdf":
                        logger.info(f"Reading PDF: {file}")
                        try:
                            r = pypdf.PdfReader(os_path)
                            logger.info(f"PDF pages: {len(r.pages)}")
                            
                            if len(r.pages) > MAX_PAGES_PDF:
                                logger.warning(f"Skipped (too many pages): {file}")
                                stats["skipped"] += 1
                                continue
                            
                            for p in r.pages:
                                content += (p.extract_text() or "")
                        
                        except Exception as pdf_err:
                            logger.error(f"PDF read error {file}: {pdf_err}")
                            stats["skipped"] += 1
                            continue
                    
                    # TXT/MD-Handling
                    elif file_ext in [".txt", ".md"]:
                        logger.info(f"Reading text file: {file}")
                        try:
                            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
                            for encoding in encodings:
                                try:
                                    with open(os_path, "r", encoding=encoding, errors='ignore') as f:
                                        content = f.read()
                                    logger.info(f"Successfully read with encoding: {encoding}")
                                    break
                                except:
                                    continue
                            
                            if not content:
                                logger.error(f"Could not read {file} with any encoding")
                                stats["skipped"] += 1
                                continue
                        
                        except Exception as txt_err:
                            logger.error(f"Text read error {file}: {txt_err}")
                            stats["skipped"] += 1
                            continue
                    
                    # Unsupported
                    else:
                        logger.warning(f"Skipped (unsupported format '{file_ext}'): {file}")
                        stats["skipped"] += 1
                        continue
                    
                    # Content-Check
                    if not content.strip():
                        logger.warning(f"Skipped (empty): {file}")
                        stats["skipped"] += 1
                        continue
                    
                    logger.info(f"Content length: {len(content)} chars")
                
                except Exception as read_err:
                    logger.error(f"File read error {file}: {read_err}")
                    stats["skipped"] += 1
                    continue
                
                # Qdrant Chunking
                logger.info(f"Creating chunks for {file}")
                existing_ids = set()
                next_offset = None
                
                try:
                    while True:
                        pts, next_offset = qdrant_client.scroll(
                            collection_name=COLLECTION_NAME,
                            scroll_filter=qmodels.Filter(must=[
                                qmodels.FieldCondition(key="path", match=qmodels.MatchValue(value=canon_path))
                            ]),
                            limit=1000,
                            offset=next_offset
                        )
                        
                        for p in pts:
                            existing_ids.add(p.id)
                        
                        if next_offset is None or not pts:
                            break
                
                except Exception as scroll_err:
                    logger.error(f"Qdrant scroll error: {scroll_err}")
                
                words = content.split()
                chunks = [" ".join(words[i:i+500]) for i in range(0, len(words), 500)]
                logger.info(f"Created {len(chunks)} chunks")
                
                is_truncated = False
                if len(chunks) > MAX_CHUNKS_PER_FILE:
                    chunks = chunks[:MAX_CHUNKS_PER_FILE]
                    is_truncated = True
                    stats["truncated"] += 1
                    logger.warning(f"Truncated to {MAX_CHUNKS_PER_FILE} chunks")
                
                points, new_ids, partial_fail = [], set(), False
                
                for idx, chk in enumerate(chunks):
                    vec = get_embedding(chk)
                    if vec:
                        pid = generate_deterministic_uuid(TENANT_ID, canon_path, idx)
                        new_ids.add(pid)
                        points.append(qmodels.PointStruct(
                            id=pid,
                            vector=vec,
                            payload={
                                "source": file,
                                "topic": topic,
                                "path": canon_path,
                                "text": chk,
                                "doc_type": doc_type
                            }
                        ))
                    else:
                        logger.warning(f"Embedding failed for chunk {idx}")
                        partial_fail = True
                
                if len(chunks) > 0 and not points:
                    logger.error(f"Abort {file}: No valid embeddings")
                    stats["abort"] += 1
                    continue
                
                upsert_ok = False
                if points:
                    try:
                        logger.info(f"Upserting {len(points)} points")
                        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
                        processed += len(points)
                        upsert_ok = True
                        logger.info(f"Upsert OK: {len(points)} points")
                    except Exception as upsert_err:
                        logger.error(f"Upsert fail: {upsert_err}")
                
                if upsert_ok and (not partial_fail) and (not is_truncated):
                    stale = list(existing_ids - new_ids)
                    if stale:
                        try:
                            logger.info(f"Deleting {len(stale)} stale points")
                            qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=qmodels.PointIdsList(points=stale))
                        except Exception as del_err:
                            logger.error(f"Delete error: {del_err}")
                else:
                    if partial_fail:
                        stats["partial_fail"] += 1
    
    logger.info(f"INGEST COMPLETE: processed={processed}, stats={stats}")
    audit_events.info(f"INGEST_DONE tenant={TENANT_ID} chunks={processed} stats={stats}")
    return {"status": "success", "processed": processed, "stats": stats}

@app.post("/ingest/eurlex")
async def ingest_eurlex(
    req: EURLexIngestReq,
    x_token: str = Header(None, alias="X-Access-Token")
):
    """
    Universeller Endpoint zum Scrapen von EUR-Lex Dokumenten mit automatischer Cross-Document Verknüpfung
    
    Unterstützt beliebige EUR-Lex URLs wie:
    - DORA: https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:32022R2554
    - RTS: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202401774
    - Andere EU-Verordnungen
    
    Body:
    {
        "url": "https://eur-lex.europa.eu/legal-content/...",
        "follow_anchors": true,  // Optional, default: true
        "max_depth": 1  // Optional, default: 1
    }
    
    AUTOMATISCHE VERKNÜPFUNG:
    Nach dem Import werden automatisch Verbindungen zu bestehenden Dokumenten erstellt:
    - Explizite CELEX-Referenzen
    - Artikel-zu-Artikel Cross-References
    - Thematische Ähnlichkeiten
    """
    if x_token != INGEST_TOKEN:
        raise HTTPException(401, "Unauthorized")
    
    if not BS4_AVAILABLE:
        raise HTTPException(503, "BeautifulSoup4 not installed")
    
    if not neo4j_driver or not qdrant_client:
        raise HTTPException(503, "Database services unavailable")
    
    logger.info(f"=== EUR-LEX INGEST START ===")
    logger.info(f"URL: {req.url}")
    logger.info(f"Follow anchors: {req.follow_anchors}, Max depth: {req.max_depth}")
    
    try:
        # Erstelle Scraper mit der übergebenen URL
        scraper = EURLexScraper(
            base_url=req.url,
            follow_anchors=req.follow_anchors,
            max_depth=req.max_depth
        )
        
        # Scrape das Dokument
        nodes = scraper.scrape_document()
        if not nodes:
            raise HTTPException(500, "No nodes extracted from EUR-Lex")
        
        # Speichere in Neo4j und Qdrant
        store_eurlex_nodes_in_neo4j(nodes, req.url)
        create_embeddings_for_eurlex_nodes(nodes)
        
        # AUTOMATISCHE CROSS-DOCUMENT VERKNÜPFUNG
        logger.info(f"=== Starting cross-document linking for {scraper.celex} ===")
        cross_link_stats = create_cross_document_links(scraper.celex)
        
               # Statistiken
        stats = {
            "total_nodes": len(nodes),
            "pages_visited": len(scraper.visited_urls),
            "by_type": {},
            "article_count": 0
        }
        
        for node in nodes:
            node_type = node.node_type
            stats["by_type"][node_type] = stats["by_type"].get(node_type, 0) + 1
            if node_type == "article":
                stats["article_count"] += 1
        
        audit_events.info(f"EURLEX_INGEST tenant={TENANT_ID} celex={scraper.celex} nodes={len(nodes)} articles={stats['article_count']} crosslinks={cross_link_stats}")
        
        return {
            "status": "success",
            "celex": scraper.celex,
            "document_name": scraper.document_name,
            "url": req.url,
            "statistics": stats,
            "cross_document_links": cross_link_stats,
            "nodes_stored": len(nodes)
        }
    
    except Exception as e:
        logger.error(f"EUR-Lex ingest error: {e}", exc_info=True)
        raise HTTPException(500, f"Ingest failed: {str(e)}")

@app.post("/ingest/eurlex/batch")
async def ingest_eurlex_batch(
    urls: List[str],
    follow_anchors: Optional[bool] = True,
    max_depth: Optional[int] = 1,
    x_token: str = Header(None, alias="X-Access-Token")
):
    """
    Batch-Import mehrerer EUR-Lex Dokumente mit automatischer Cross-Document Verknüpfung
    
    Body:
    {
        "urls": [
            "https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:32022R2554",
            "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202401774"
        ],
        "follow_anchors": true,
        "max_depth": 1
    }
    
    WICHTIG: Nach dem Import aller Dokumente wird automatisch 
    die komplette Cross-Document-Verknüpfung für ALLE importierten Dokumente durchgeführt.
    """
    if x_token != INGEST_TOKEN:
        raise HTTPException(401, "Unauthorized")
    
    if not BS4_AVAILABLE:
        raise HTTPException(503, "BeautifulSoup4 not installed")
    
    if not neo4j_driver or not qdrant_client:
        raise HTTPException(503, "Database services unavailable")
    
    if not urls or len(urls) == 0:
        raise HTTPException(400, "No URLs provided")
    
    if len(urls) > 10:
        raise HTTPException(400, "Maximum 10 URLs per batch")
    
    logger.info(f"=== EUR-LEX BATCH INGEST START: {len(urls)} documents ===")
    
    results = []
    imported_celex = []
    
    # Schritt 1: Importiere alle Dokumente
    for idx, url in enumerate(urls):
        logger.info(f"[{idx+1}/{len(urls)}] Processing {url}")
        
        try:
            scraper = EURLexScraper(
                base_url=url,
                follow_anchors=follow_anchors,
                max_depth=max_depth
            )
            
            nodes = scraper.scrape_document()
            if not nodes:
                logger.error(f"No nodes extracted from {url}")
                results.append({
                    "url": url,
                    "status": "failed",
                    "error": "No nodes extracted"
                })
                continue
            
            # Speichere in Neo4j und Qdrant
            store_eurlex_nodes_in_neo4j(nodes, url)
            create_embeddings_for_eurlex_nodes(nodes)
            
            imported_celex.append(scraper.celex)
            
            results.append({
                "url": url,
                "status": "success",
                "celex": scraper.celex,
                "nodes": len(nodes),
                "articles": sum(1 for n in nodes if n.node_type == "article")
            })
            
            logger.info(f"✓ Imported {scraper.celex}: {len(nodes)} nodes")
        
        except Exception as e:
            logger.error(f"Failed to import {url}: {e}")
            results.append({
                "url": url,
                "status": "failed",
                "error": str(e)
            })
    
    # Schritt 2: Cross-Document Linking für ALLE importierten Dokumente
    logger.info(f"=== Starting cross-document linking for {len(imported_celex)} documents ===")
    cross_link_stats = rebuild_all_cross_document_links()
    
    logger.info(f"=== BATCH INGEST COMPLETE ===")
    audit_events.info(f"EURLEX_BATCH_INGEST tenant={TENANT_ID} docs={len(imported_celex)} crosslinks={cross_link_stats}")
    
    return {
        "status": "success",
        "total_urls": len(urls),
        "imported_documents": len(imported_celex),
        "celex_list": imported_celex,
        "results": results,
        "cross_document_links": cross_link_stats,
        "config": {
            "follow_anchors": follow_anchors,
            "max_depth": max_depth
        }
    }

@app.post("/maintenance/rebuild-links")
async def rebuild_links(x_token: str = Header(None, alias="X-Access-Token")):
    """
    Baut alle Cross-Document Links neu auf
    
    Nützlich wenn:
    - Links fehlen oder falsch sind
    - Neue Dokumente importiert wurden ohne Linking
    - Die Linking-Logik verbessert wurde
    """
    if x_token != MAINT_TOKEN:
        raise HTTPException(401, "Unauthorized")
    
    if not neo4j_driver:
        raise HTTPException(503, "Neo4j unavailable")
    
    logger.info("=== MANUAL LINK REBUILD TRIGGERED ===")
    stats = rebuild_all_cross_document_links()
    
    if stats is None:
        raise HTTPException(500, "Link rebuild failed")
    
    audit_events.info(f"LINK_REBUILD tenant={TENANT_ID} stats={stats}")
    
    return {
        "status": "success",
        "message": "Cross-document links rebuilt",
        "statistics": stats
    }

@app.get("/documents/relationships")
async def get_document_relationships(
    celex: Optional[str] = None,
    x_token: str = Header(None, alias="X-Access-Token")
):
    """
    Zeigt Beziehungen zwischen Dokumenten an
    
    Query Parameter:
    - celex: Optional, filtert auf ein bestimmtes Dokument
    
    KORRIGIERT: Sucht jetzt korrekt nach RELATED_REGULATION (mit Unterstrich)
    """
    if x_token != INGEST_TOKEN and x_token != MAINT_TOKEN:
        raise HTTPException(401, "Unauthorized")
    
    if not neo4j_driver:
        raise HTTPException(503, "Neo4j unavailable")
    
    try:
        with neo4j_driver.session() as session:
            if not celex:
                # Alle Dokument-Beziehungen - KORRIGIERT
                result = session.run("""
                    MATCH (d1:EURLexDocument {tenant: $tenant})-[r:RELATED_REGULATION]->(d2:EURLexDocument)
                    WHERE d1.celex <> d2.celex 
                      AND r.connection_strength > 0
                    RETURN d1.celex AS doc1, 
                           d1.title AS title1,
                           d2.celex AS doc2, 
                           d2.title AS title2,
                           r.connection_strength AS strength,
                           r.last_analyzed AS last_analyzed
                    ORDER BY strength DESC
                """, tenant=TENANT_ID)
                
                relationships = []
                for record in result:
                    relationships.append({
                        "document1": {
                            "celex": record["doc1"],
                            "title": record["title1"]
                        },
                        "document2": {
                            "celex": record["doc2"],
                            "title": record["title2"]
                        },
                        "connection_strength": record["strength"],
                        "last_analyzed": str(record["last_analyzed"]) if record.get("last_analyzed") else None
                    })
                
                return {
                    "total_relationships": len(relationships),
                    "relationships": relationships
                }
            
            else:
                # Beziehungen für ein bestimmtes Dokument - KORRIGIERT
                result = session.run("""
                    MATCH (d:EURLexDocument {celex: $celex, tenant: $tenant})
                    
                    // Ausgehende Beziehungen
                    OPTIONAL MATCH (d)-[r1:RELATED_REGULATION]->(other:EURLexDocument)
                    WHERE r1.connection_strength > 0
                    
                    // Eingehende Beziehungen
                    OPTIONAL MATCH (d)<-[r2:RELATED_REGULATION]-(incoming:EURLexDocument)
                    WHERE r2.connection_strength > 0
                    
                    RETURN d.celex AS document,
                           d.title AS title,
                           d.url AS url,
                           collect(DISTINCT {
                               celex: other.celex,
                               title: other.title,
                               strength: r1.connection_strength,
                               direction: 'outgoing',
                               last_analyzed: r1.last_analyzed
                           }) AS outgoing_relations,
                           collect(DISTINCT {
                               celex: incoming.celex,
                               title: incoming.title,
                               strength: r2.connection_strength,
                               direction: 'incoming',
                               last_analyzed: r2.last_analyzed
                           }) AS incoming_relations
                """, celex=celex, tenant=TENANT_ID)
                
                record = result.single()
                if not record:
                    raise HTTPException(404, f"Document {celex} not found")
                
                # Kombiniere eingehende und ausgehende Beziehungen
                all_relations = []
                
                for rel in record["outgoing_relations"]:
                    if rel["celex"]:  # Filtere null-Werte
                        all_relations.append({
                            "celex": rel["celex"],
                            "title": rel["title"],
                            "connection_strength": rel["strength"],
                            "direction": rel["direction"],
                            "last_analyzed": str(rel["last_analyzed"]) if rel.get("last_analyzed") else None
                        })
                
                for rel in record["incoming_relations"]:
                    if rel["celex"]:  # Filtere null-Werte
                        all_relations.append({
                            "celex": rel["celex"],
                            "title": rel["title"],
                            "connection_strength": rel["strength"],
                            "direction": rel["direction"],
                            "last_analyzed": str(rel["last_analyzed"]) if rel.get("last_analyzed") else None
                        })
                
                # Sortiere nach Stärke
                all_relations.sort(key=lambda x: x["connection_strength"], reverse=True)
                
                return {
                    "document": record["document"],
                    "title": record["title"],
                    "url": record["url"],
                    "total_relationships": len(all_relations),
                    "related_documents": all_relations
                }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching relationships: {e}")
        raise HTTPException(500, str(e))

# ==================== ZUSÄTZLICHE HELPER ENDPOINTS ====================

@app.get("/documents")
async def list_documents(
    x_token: str = Header(None, alias="X-Access-Token")
):
    """
    Listet alle EUR-Lex Dokumente auf
    """
    if x_token != INGEST_TOKEN and x_token != MAINT_TOKEN:
        raise HTTPException(401, "Unauthorized")
    
    if not neo4j_driver:
        raise HTTPException(503, "Neo4j unavailable")
    
    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (d:EURLexDocument {tenant: $tenant})
                OPTIONAL MATCH (d)-[:CONTAINS]->(n:EURLexNode)
                RETURN d.celex AS celex,
                       d.title AS title,
                       d.url AS url,
                       d.ingested_at AS ingested_at,
                       d.node_count AS node_count,
                       count(n) AS actual_nodes,
                       count(CASE WHEN n.type = 'article' THEN 1 END) AS article_count
                ORDER BY d.ingested_at DESC
            """, tenant=TENANT_ID)
            
            documents = []
            for record in result:
                documents.append({
                    "celex": record["celex"],
                    "title": record["title"],
                    "url": record["url"],
                    "ingested_at": str(record["ingested_at"]) if record.get("ingested_at") else None,
                    "node_count": record["node_count"],
                    "actual_nodes": record["actual_nodes"],
                    "article_count": record["article_count"]
                })
            
            return {
                "total_documents": len(documents),
                "documents": documents
            }
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(500, str(e))

@app.get("/documents/{celex}")
async def get_document_details(
    celex: str,
    x_token: str = Header(None, alias="X-Access-Token")
):
    """
    Zeigt Details zu einem bestimmten Dokument
    """
    if x_token != INGEST_TOKEN and x_token != MAINT_TOKEN:
        raise HTTPException(401, "Unauthorized")
    
    if not neo4j_driver:
        raise HTTPException(503, "Neo4j unavailable")
    
    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (d:EURLexDocument {celex: $celex, tenant: $tenant})
                OPTIONAL MATCH (d)-[:CONTAINS]->(n:EURLexNode)
                RETURN d.celex AS celex,
                       d.title AS title,
                       d.url AS url,
                       d.ingested_at AS ingested_at,
                       d.source AS source,
                       collect({
                           id: n.id,
                           type: n.type,
                           number: n.number,
                           title: n.title,
                           url: n.url
                       }) AS nodes
            """, celex=celex, tenant=TENANT_ID)
            
            record = result.single()
            if not record:
                raise HTTPException(404, f"Document {celex} not found")
            
            # Gruppiere Nodes nach Typ
            nodes_by_type = {}
            for node in record["nodes"]:
                if node["id"]:  # Filtere null-Werte
                    node_type = node["type"]
                    if node_type not in nodes_by_type:
                        nodes_by_type[node_type] = []
                    nodes_by_type[node_type].append({
                        "id": node["id"],
                        "number": node["number"],
                        "title": node["title"],
                        "url": node["url"]
                    })
            
            return {
                "celex": record["celex"],
                "title": record["title"],
                "url": record["url"],
                "ingested_at": str(record["ingested_at"]) if record.get("ingested_at") else None,
                "source": record["source"],
                "total_nodes": sum(len(nodes) for nodes in nodes_by_type.values()),
                "nodes_by_type": nodes_by_type
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching document details: {e}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

