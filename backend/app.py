import os
import sys
import logging
from datetime import datetime
from typing import Dict

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conversation_processor import ConversationProcessor
from persona_extractor import PersonaExtractor
from rag_system import RAGSystem


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

processor = None
persona_extractor = None
rag_system = None
personas_data = None
processed_data = None
initialization_error = None


def initialize_system(force: bool = False) -> bool:
    global processor, persona_extractor, rag_system, personas_data, processed_data, initialization_error

    if not force and processor and persona_extractor and rag_system and personas_data and processed_data:
        return True

    logger.info("Initializing RAG system...")

    try:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "conversations.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing dataset: {csv_path}")

        processor = ConversationProcessor(csv_path)

        logger.info("Processing conversations...")
        processed_data = processor.process()

        logger.info("Extracting personas...")
        persona_extractor = PersonaExtractor()
        personas_data = persona_extractor.extract_persona(processed_data["messages"])

        logger.info("Building RAG index...")
        rag_system = RAGSystem(use_embeddings=False)
        rag_system.build_index(
            processed_data["messages"],
            processed_data["topics"],
            processed_data["checkpoints"],
        )

        initialization_error = None
        logger.info("System initialized successfully")
        return True
    except Exception as exc:
        initialization_error = str(exc)
        logger.exception("Error initializing system")
        return False


def ensure_initialized() -> bool:
    return initialize_system(force=False)


@app.route("/health", methods=["GET"])
def health():
    ready = bool(processed_data)
    return (
        jsonify(
            {
                "status": "ok",
                "ready": ready,
                "error": initialization_error,
                "timestamp": datetime.now().isoformat(),
            }
        ),
        200 if ready else 503,
    )


@app.route("/api/stats", methods=["GET"])
def get_stats():
    if not ensure_initialized():
        return jsonify({"error": initialization_error or "System not initialized"}), 500

    return jsonify(
        {
            "total_messages": processed_data["total_messages"],
            "total_topics": processed_data["total_topics"],
            "total_checkpoints": processed_data["total_checkpoints"],
            "conversations_count": len(processed_data["conversations"]),
            "system_initialized": True,
        }
    )


@app.route("/api/topics", methods=["GET"])
def get_topics():
    if not ensure_initialized():
        return jsonify({"error": initialization_error or "System not initialized"}), 500

    topics_list = []
    for topic in processed_data["topics"]:
        topics_list.append(
            {
                "id": topic.get("topic_id", 0),
                "start_message": topic.get("start_idx", 0),
                "end_message": topic.get("end_idx", 0),
                "num_messages": len(topic.get("messages", [])),
                "keywords": topic.get("keywords", []),
            }
        )

    return jsonify({"topics": topics_list, "total": len(topics_list)})


@app.route("/api/checkpoints", methods=["GET"])
def get_checkpoints():
    if not ensure_initialized():
        return jsonify({"error": initialization_error or "System not initialized"}), 500

    checkpoints_list = []
    for checkpoint in processed_data["checkpoints"]:
        checkpoints_list.append(
            {
                "id": checkpoint.get("checkpoint_id", 0),
                "start_message": checkpoint.get("start_msg", 0),
                "end_message": checkpoint.get("end_msg", 0),
                "num_messages": checkpoint.get("num_messages", 0),
                "summary": checkpoint.get("summary", ""),
            }
        )

    return jsonify({"checkpoints": checkpoints_list, "total": len(checkpoints_list)})


@app.route("/api/retrieve", methods=["POST"])
def retrieve_context():
    if not ensure_initialized():
        return jsonify({"error": initialization_error or "System not initialized"}), 500

    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    top_k = data.get("top_k", 5)

    if not query:
        return jsonify({"error": "Query required"}), 400

    results = rag_system.retrieve(query, top_k=top_k)
    return jsonify(
        {
            "success": True,
            "query": query,
            "results": {
                "messages": results.get("messages", []),
                "topics": results.get("topics", []),
                "checkpoints": results.get("checkpoints", []),
                "context": results.get("context", {}),
            },
        }
    )


@app.route("/api/persona", methods=["GET"])
def get_persona():
    if not ensure_initialized():
        return jsonify({"error": initialization_error or "System not initialized"}), 500

    return jsonify({"personas": personas_data, "total_users": len(personas_data)})


@app.route("/api/persona/<user_id>", methods=["GET"])
def get_user_persona(user_id):
    if not ensure_initialized():
        return jsonify({"error": initialization_error or "System not initialized"}), 500

    key = f"user_{user_id}" if not user_id.startswith("user_") else user_id
    if key not in personas_data:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"user": user_id, "persona": personas_data[key]})


def generate_personality_answer(question: str, personas: Dict) -> str:
    answers = []

    for user_key, persona in personas.items():
        traits = persona.get("personality_traits", {})
        interests = persona.get("personal_facts", {}).get("interests", [])

        description = f"{user_key.replace('_', ' ').title()} is "
        parts = []
        if traits.get("positive", {}).get("frequency", 0) > 0:
            parts.append("generally positive")
        if traits.get("emotional", {}).get("frequency", 0) > 2:
            parts.append("emotionally expressive")
        if traits.get("humorous", {}).get("frequency", 0) > 2:
            parts.append("often uses humor")
        if interests:
            parts.append(f"interested in {', '.join(interests[:3])}")

        if parts:
            description += ", ".join(parts) + "."
        else:
            description += "a thoughtful individual."

        answers.append(description)

    return " ".join(answers) if answers else "Limited personality data available."


def generate_habit_answer(question: str, personas: Dict, context: Dict) -> str:
    answers = []

    for user_key, persona in personas.items():
        habits = persona.get("habits", {})
        mentioned_habits = [habit for habit, data in habits.items() if data.get("mentioned")]
        if mentioned_habits:
            answers.append(
                f"{user_key.replace('_', ' ').title()} enjoys: {', '.join(mentioned_habits[:5])}."
            )

    if context.get("messages"):
        msg_preview = context["messages"][0].get("text", "")[:100]
        if msg_preview:
            answers.append(f'Example: "{msg_preview}..."')

    return " ".join(answers) if answers else "Limited habit information available."


def generate_communication_answer(question: str, personas: Dict) -> str:
    answers = []

    for user_key, persona in personas.items():
        communication = persona.get("communication_style", {})
        style_desc = f"{user_key.replace('_', ' ').title()} "
        parts = []

        if communication.get("tone") == "casual":
            parts.append("uses a casual tone")
        elif communication.get("tone") == "formal":
            parts.append("uses a formal tone")

        if communication.get("expressiveness") == "high":
            parts.append("is highly expressive with frequent exclamations")

        avg_len = communication.get("average_message_length", 0)
        if avg_len < 10:
            parts.append("tends to write short messages")
        elif avg_len > 20:
            parts.append("typically writes longer, detailed messages")

        if parts:
            style_desc += ", ".join(parts) + "."
        else:
            style_desc += "maintains a balanced communication style."

        answers.append(style_desc)

    return " ".join(answers) if answers else "Limited communication data available."


def generate_general_answer(question: str, context: Dict, personas: Dict) -> str:
    answer_parts = []

    if context.get("messages"):
        messages_text = [message.get("text", "") for message in context["messages"][:2]]
        if messages_text:
            answer_parts.append(f"Based on conversations: {' '.join(messages_text[:100])}")

    context_summary = context.get("context", {}).get("summary", "")
    if context_summary:
        answer_parts.append(f"Context: {context_summary}")

    if answer_parts:
        return " ".join(answer_parts)
    return "I found limited information relevant to your question in the conversations."


def generate_answer(question: str, context: Dict, personas: Dict) -> str:
    question_lower = question.lower()

    if "person" in question_lower or "kind of" in question_lower or "like" in question_lower:
        return generate_personality_answer(question, personas)
    if "habit" in question_lower or "do" in question_lower:
        return generate_habit_answer(question, personas, context)
    if "talk" in question_lower or "communicate" in question_lower or "write" in question_lower:
        return generate_communication_answer(question, personas)
    return generate_general_answer(question, context, personas)


@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    if not ensure_initialized():
        return jsonify({"error": initialization_error or "System not initialized"}), 500

    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question required"}), 400

    context = rag_system.retrieve(question, top_k=3)
    answer = generate_answer(question, context, personas_data)

    return jsonify(
        {
            "question": question,
            "answer": answer,
            "context_used": {
                "messages_found": len(context["messages"]),
                "topics_found": len(context["topics"]),
                "checkpoints_found": len(context["checkpoints"]),
            },
        }
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    if not ensure_initialized():
        return jsonify({"error": initialization_error or "System not initialized"}), 500

    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "Message required"}), 400

    context = rag_system.retrieve(message, top_k=2)
    response = generate_answer(message, context, personas_data)

    return jsonify(
        {
            "user_message": message,
            "bot_response": response,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/ui", methods=["GET"])
def ui():
    return send_from_directory(os.path.dirname(__file__), "index.html")


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "name": "Conversation Analysis RAG System",
            "version": "1.0.0",
            "ui": "/ui",
            "endpoints": {
                "GET /ui": "Chat UI interface",
                "GET /health": "Health check",
                "GET /api/stats": "System statistics",
                "GET /api/topics": "Get detected topics",
                "GET /api/checkpoints": "Get 100-message checkpoints",
                "POST /api/retrieve": "Retrieve context for query",
                "GET /api/persona": "Get all personas",
                "GET /api/persona/<user_id>": "Get specific user persona",
                "POST /api/chatbot": "Main chatbot endpoint",
                "POST /api/chat": "Simple chat endpoint",
            },
        }
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if os.getenv("FLASK_SKIP_INIT", "").lower() not in {"1", "true"}:
    initialize_system()


if __name__ == "__main__":
    logger.info("Starting Flask app...")
    host = "0.0.0.0"
    port = int(os.getenv("PORT", "5000"))

    if initialize_system():
        logger.info("System initialized, starting server on %s:%s", host, port)
        app.run(host=host, port=port, debug=False)
    else:
        logger.error("Failed to initialize system")
        sys.exit(1)
