# CS2 - Compliance Auditor API

FastAPI-basierte Compliance-Auditor-Anwendung mit Neo4j Graph Database, Qdrant Vector Database und Ollama LLM Integration.

## 🎯 Funktionen

Diese Anwendung implementiert umfassende Verbesserungen für:

### ✅ Neo4j-Datennutzung mit verbessertem Logging
- Detailliertes Logging für jede Abfrage mit Emoji-Markern (📄📋🔗🔍⚠️📊)
- Keyword-Extraktion und -Logging
- Separate Try/Except-Blöcke für Requirements und EUR-Lex Queries
- Zählung gefundener Requirements und EUR-Lex Nodes
- Warnungen wenn KEINE Daten gefunden werden

### ✅ Quellenangaben in LLM-Antworten
- Explizites Source-Tracking für Neo4j und Qdrant
- Validierung ob Context Neo4j-Daten enthält
- Verbesserter System-Prompt mit ZWINGENDER Quellenzitation
- LLM-Antwort-Validierung (CELEX, PDF, REQ)
- Automatische Warnung bei fehlenden Quellen

### ✅ Streaming-Support
- Gleiche Validierungs- und Logging-Prinzipien auch für Streaming-Responses

## 🚀 Installation

```bash
# Repository klonen
git clone https://github.com/newcomer-cmotal/cs2.git
cd cs2

# Dependencies installieren
pip install -r requirements.txt
```

## ⚙️ Konfiguration

Umgebungsvariablen (optional):

```bash
export TENANT_ID="YourTenant"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export OLLAMA_BASE_URL="http://localhost:11434"
export DEFAULT_MODEL="llama2"
```

## 🧪 Tests

```bash
# Alle Tests ausführen
pytest test_main.py -v

# Mit Coverage
pytest test_main.py -v --cov=main
```

**Test-Ergebnisse:** 13/13 Tests bestehen ✅

## 🏃 Starten der Anwendung

```bash
python main.py
```

Die API ist dann verfügbar unter: `http://localhost:8000`

Swagger-Dokumentation: `http://localhost:8000/docs`

## 📡 API-Endpunkte

### Health Check
```bash
GET /
GET /health
```

### Chat Completion
```bash
POST /v1/chat/completions
```

**Beispiel Request:**
```json
{
  "model": "llama2",
  "messages": [
    {"role": "user", "content": "Was sind die DORA-Anforderungen für IKT-Risikomanagement?"}
  ],
  "stream": false
}
```

**Beispiel Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "llama2",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Gemäß [CELEX:32022R2554, Artikel 5] müssen Finanzunternehmen..."
    }
  }]
}
```

## 🔍 Logging-Beispiele

Die Anwendung produziert detaillierte Logs:

```
INFO - 🔍 Neo4j search with keywords: ['dora', 'risikomanagement', 'anforderungen']
INFO - ✓ Found 3 requirements in Neo4j
INFO - ✓ Found 2 EUR-Lex nodes in Neo4j
INFO - 📊 Neo4j query complete: 5 total results
INFO - === CONTEXT SUMMARY ===
INFO - Neo4j sources: 5
INFO - Qdrant sources: 0
INFO - Total context length: 1250 chars
INFO - ✓ LLM response contains sources: CELEX=True, PDF=False, REQ=True
```

## 📋 Quellenformat

Die Anwendung erzwingt folgende Quellenformate:

- **EUR-LEX Artikel:** `[CELEX:32022R2554, Artikel 5]`
- **PDF-Dokumente:** `[Policy_IKT.pdf, Seite 3, Zeile 45]`
- **Requirements:** `[REQ-001]` oder `[SOLL-ANFORDERUNG: REQ-IKT-042]`

## 🔒 Sicherheit

Alle Dependencies sind auf sichere Versionen aktualisiert:
- ✅ FastAPI 0.109.1+ (CVE-Fixes)
- ✅ qdrant-client 1.9.0+ (Input-Validierung)
- ✅ python-multipart 0.0.18+ (DoS/ReDoS-Fixes)
- ✅ CodeQL: Keine Sicherheitswarnungen

## 🤝 Contributing

Pull Requests sind willkommen!

## 📄 Lizenz

[Bitte Lizenz hinzufügen]