# Conversation Analysis RAG System

An AI/ML system that builds a Retrieval-Augmented Generation (RAG) system from conversation data, extracts user personas, and provides intelligent chatbot capabilities.

## Features

- **RAG System with Topic Checkpoints**: Processes conversations chronologically, detects topic changes, and creates semantic summaries
- **100-Message Checkpoints**: Automatically creates summaries every 100 messages for easy navigation
- **Topic Detection**: Uses semantic similarity to detect when topics change in conversations
- **User Persona Extraction**: Extracts habits, personality traits, communication styles, and personal facts
- **Intelligent Chatbot**: Answers questions about user personalities, habits, and communication styles using RAG + Persona data

## System Architecture

```
conversations.csv
    ↓
ConversationParser (parse & structure)
    ↓
[Messages Stream]
    ↓
├─→ TopicDetector (detect topic changes using semantic similarity)
│       ↓
│   [Topic Checkpoints with summaries]
│
└─→ MessageCheckpointGenerator (every 100 messages)
        ↓
    [100-Message Checkpoints with summaries]
        ↓
        RAGSystem (build index)
        ↓
        PersonaExtractor (extract from messages)
        ↓
        [Persona Data - JSON]
        ↓
        Chatbot API (Flask)
        ↓
        Web UI
```

## How It Works

### Part 1: RAG System with Checkpoints

1. **Topic Detection Algorithm**:
   - Uses TF-IDF vectorization for semantic analysis
   - Applies sliding window similarity (every 5 messages)
   - Detects topic shifts when similarity < 0.4 threshold
   - Creates semantic summaries for each topic

2. **Topic Checkpoints**:
   - Auto-detected when conversation semantics change significantly
   - Each checkpoint contains: messages, keywords, text summary
   - Example output:
     ```
     Topic 1 → messages 0–25 → "Discussion about moving to Portland"
     Topic 2 → messages 26–60 → "Conversation about music and hobbies"
     ```

3. **100-Message Checkpoints**:
   - Fixed-size checkpoints every 100 messages
   - Independent of topic changes
   - Contains: message range, summary, full text

### Part 2: User Persona Extraction

Extracts structured persona data:

```json
{
  "habits": {
    "cooking": {"mentioned": true, "frequency": 5, "examples": [...]},
    "reading": {"mentioned": true, "frequency": 3, "examples": [...]}
  },
  "personality_traits": {
    "positive": {"score": 0.8, "frequency": 12},
    "emotional": {"score": 0.6, "frequency": 8}
  },
  "communication_style": {
    "tone": "casual",
    "expressiveness": "high",
    "average_message_length": 15.2
  },
  "personal_facts": {
    "relationships": ["parent", "friend"],
    "occupation": ["software engineer"]
  }
}
```

### Part 3: Chatbot

Questions answered:
- "What kind of person is this user?" → Personality + traits
- "What are their habits?" → Extracted habits with frequency
- "How do they talk?" → Communication style analysis

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repo-url>
cd KaStack
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare data:
```bash
# Place conversations.csv in project root
```

## Running the System

### Local Development

1. Start the backend:
```bash
cd backend
python app.py
```

2. Open browser:
```
http://localhost:5000
```

### Production Deployment (Heroku)

```bash
heroku create your-app-name
git push heroku main
heroku logs --tail
```

## API Endpoints

```
GET  /health                    - Health check
GET  /api/stats                 - System statistics
GET  /api/topics                - Get all detected topics
GET  /api/checkpoints           - Get all 100-message checkpoints
POST /api/retrieve              - Retrieve context (RAG)
GET  /api/persona               - Get all personas
GET  /api/persona/<user_id>     - Get specific user persona
POST /api/chatbot               - Main chatbot endpoint
POST /api/chat                  - Simple chat
```

### Example Requests

**Retrieve Context**:
```bash
curl -X POST http://localhost:5000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "cooking and travel", "top_k": 5}'
```

**Chat with Chatbot**:
```bash
curl -X POST http://localhost:5000/api/chatbot \
  -H "Content-Type: application/json" \
  -d '{"question": "What kind of person is this user?"}'
```

## Topic Detection Algorithm

The system detects topics using semantic similarity:

1. **Vectorization**: Convert all messages to TF-IDF vectors
2. **Sliding Window**: Every 5 messages, compare current window vs. next window
3. **Similarity Threshold**: If similarity < 0.4, mark as topic change
4. **Keywords Extraction**: Extract top keywords for each topic
5. **Summarization**: Generate text summary using first 2 sentences

### Algorithm Parameters
- `window_size`: 5 messages (for similarity calculation)
- `similarity_threshold`: 0.4 (topic change detection)
- `summary_sentences`: 2 (per topic)

## Persona Extraction Approach

Uses pattern matching and statistical analysis:

1. **Habit Extraction**: Regex patterns for common habits (food, sleep, exercise, etc.)
2. **Personality Detection**: Sentiment analysis + emotion keywords
3. **Communication Analysis**: Message length, tone, exclamations/questions
4. **Fact Extraction**: Relationships, occupations, locations mentioned
5. **Interest Inference**: Topic frequency analysis

### Example Extraction
```
Input: "I love cooking and reading. I work as a software engineer."
Output:
- Habits: cooking, reading
- Occupation: software engineer
- Personality: positive (loves)
- Communication: assertive (exclamation usage)
```

## Data Structure

### Conversations
```
Row: Full conversation text
Multiple "User X: message" entries
```

### Processed Output

**messages.json**:
```json
[
  {
    "user_id": 1,
    "text": "Hi there!",
    "global_idx": 0,
    "tokens": ["hi", "there"]
  }
]
```

**topics.json**:
```json
[
  {
    "topic_id": 0,
    "start_msg": 0,
    "end_msg": 25,
    "num_messages": 26,
    "keywords": ["portland", "moving", "city"],
    "summary": "User discussing moving to Portland..."
  }
]
```

**checkpoints.json**:
```json
[
  {
    "checkpoint_id": 0,
    "start_msg": 0,
    "end_msg": 99,
    "num_messages": 100,
    "summary": "Initial conversation checkpoint..."
  }
]
```

## Performance

- **Processing Speed**: ~0.5 seconds per 100 messages
- **Memory**: ~200MB for full dataset
- **RAG Retrieval**: <50ms for queries
- **Persona Extraction**: ~2-5 seconds for full analysis

## Key Design Decisions

1. **Topic Detection**: Uses semantic similarity rather than keyword clustering
   - Pros: Captures subtle topic changes, language-agnostic
   - Cons: Requires vectorization

2. **RAG System**: Dual index (topics + messages + checkpoints)
   - Enables multi-level retrieval
   - Provides context at different granularities

3. **Persona Extraction**: Rule-based patterns + statistical analysis
   - Pros: Fast, explainable, no external APIs needed
   - Cons: Limited to predefined patterns

4. **Deployment**: Stateless Flask API
   - Pros: Scalable, simple to deploy
   - Cons: Requires precomputed indexes

## Files Structure

```
KaStack/
├── conversations.csv           # Input conversation data
├── requirements.txt            # Python dependencies
├── backend/
│   ├── app.py                 # Flask API server
│   ├── index.html             # Frontend UI
│   └── venv/                  # Virtual environment
├── src/
│   ├── conversation_processor.py   # Main processing pipeline
│   ├── persona_extractor.py        # Persona extraction
│   └── rag_system.py               # RAG retrieval system
├── data/
│   ├── messages.json          # Processed messages
│   ├── topics.json            # Detected topics
│   └── checkpoints.json       # 100-message checkpoints
└── notebooks/
    └── analysis.ipynb         # Jupyter analysis notebook
```

## Limitations & Future Work

### Current Limitations
1. Topic detection threshold is fixed (0.4) - could be adaptive
2. Persona extraction uses predefined patterns - could use NLP models
3. Single-user extraction - could support multi-user comparison
4. No fine-tuning of topic boundaries

### Future Enhancements
- [ ] Dynamic topic threshold adjustment
- [ ] Named entity recognition for person/location extraction
- [ ] Sentiment trend analysis
- [ ] Multi-user comparison and clustering
- [ ] Fine-grained emotion detection
- [ ] Temporal analysis of habit changes

## Testing

```bash
# Run tests
pytest tests/

# Analyze single conversation
python -m src.conversation_processor
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/xyz`
3. Commit changes: `git commit -m "Add xyz"`
4. Push to branch: `git push origin feature/xyz`
5. Submit pull request

## License

MIT License - See LICENSE file

## Contact

For questions or issues, please open a GitHub issue.

---

**Built for KaStack Labs AI/ML Internship Interview**
