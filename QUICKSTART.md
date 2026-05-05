# Quick Start Guide

## 1. Local Setup (Windows)

### Prerequisites
- Python 3.8+ installed
- conversatIONs.csv in the project root

### Installation

```bash
# Clone or navigate to project
cd KaStack

# Run setup script (Windows)
.\run.bat

# Or manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd backend
python app.py
```

**Output:**
```
Starting RAG System Backend
API will be available at: http://localhost:5000
Frontend will be available at: http://localhost:5000
```

### Testing Locally

Open browser → http://localhost:5000

**API Endpoints to Test:**

```bash
# Health check
curl http://localhost:5000/health

# Get system stats
curl http://localhost:5000/api/stats

# Chat with bot
curl -X POST http://localhost:5000/api/chatbot ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What kind of person is this user?\"}"
```

## 2. System Statistics

After processing your conversations.csv:

```json
{
  "total_messages": 191592,
  "total_topics": 1,
  "total_checkpoints": 1916,
  "conversations_count": 25
}
```

## 3. Key Features

### Topic Detection
- Chronological message processing
- Semantic similarity-based topic changes
- Automatic keyword extraction per topic
- Summary generation

### Message Checkpoints
- Every 100 messages automatically
- Independent of topic changes
- Contains full message context
- Summary for quick reference

### Persona Extraction
- Habits (cooking, reading, sports, etc.)
- Personality traits (positive, emotional, humorous)
- Communication style (tone, expressiveness, message length)
- Personal facts (relationships, occupations, locations)

### RAG Retrieval
- Query-based message search
- Topic-level context retrieval
- Checkpoint-level summaries
- Multi-level relevance scoring

## 4. API Examples

### Query Retrieval
```bash
curl -X POST http://localhost:5000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "cooking and hobbies", "top_k": 5}'
```

**Response:**
```json
{
  "query": "cooking and hobbies",
  "results": {
    "messages": [
      {
        "message_idx": 42,
        "text": "I love cooking and reading books!",
        "similarity": 0.89
      }
    ],
    "topics": [...],
    "checkpoints": [...]
  }
}
```

### Chat with Chatbot
```bash
curl -X POST http://localhost:5000/api/chatbot \
  -H "Content-Type: application/json" \
  -d '{"question": "What are their habits?"}'
```

### Get Persona
```bash
curl http://localhost:5000/api/persona
curl http://localhost:5000/api/persona/user_1
```

## 5. Frontend Usage

### Dashboard Tabs

1. **Stats**: Overview of dataset statistics
2. **Topics**: List of detected topics with keywords
3. **Checkpoints**: 100-message summaries
4. **Persona**: User personality and habits

### Chat Interface

Type questions like:
- "What kind of person is this user?"
- "What are their habits?"
- "How do they communicate?"
- "What do they like to do?"

System responds with RAG-retrieved context + persona data.

## 6. File Structure

```
KaStack/
├── conversations.csv          # Input data
├── data/
│   ├── messages.json         # All messages (67 MB)
│   ├── topics.json           # Detected topics
│   └── checkpoints.json      # 100-msg checkpoints
├── src/
│   ├── conversation_processor.py   # Main pipeline
│   ├── persona_extractor.py        # Persona analysis
│   └── rag_system.py               # Retrieval system
├── backend/
│   ├── app.py                # Flask server
│   └── index.html            # Web UI
└── requirements.txt           # Dependencies
```

## 7. Performance Tips

### Speed Up Processing
- Use TF-IDF instead of embeddings (default)
- Process data once, reuse JSON files
- Cache RAG indexes

### Reduce Memory
- Stream messages instead of loading all
- Use checkpoint-level retrieval for summaries
- Disable full message logging

## 8. Troubleshooting

### Issue: "Module not found"
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Port 5000 already in use"
**Solution:**
```bash
# Change port in backend/app.py
app.run(port=5001)
```

### Issue: Slow startup
**Solution:**
- First run downloads NLTK data (~300MB)
- Subsequent runs are faster
- Data already processed in `data/` folder

### Issue: Large memory usage
**Solution:**
- Reduce checkpoint_interval in rag_system.py
- Use pagination for large datasets
- Stream processing instead of loading all

## 9. Next Steps

1. **Deploy to GitHub**: Push to repository
2. **Deploy to Heroku**: Use `git push heroku main`
3. **Create Demo Video**: Record system walkthrough
4. **Test End-to-End**: Verify all features work

## 10. Commands Reference

| Command | Purpose |
|---------|---------|
| `python src/conversation_processor.py` | Process conversations |
| `cd backend && python app.py` | Start Flask server |
| `python -m pytest tests/` | Run tests |
| `heroku logs --tail` | View deployment logs |
| `git push heroku main` | Deploy to Heroku |

---

**For detailed documentation**, see:
- [README.md](README.md) - Full system documentation
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
