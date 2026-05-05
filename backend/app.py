import os
import sys
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conversation_processor import ConversationProcessor
from persona_extractor import PersonaExtractor, PersonaGenerator
from rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for loaded data
processor = None
persona_extractor = None
rag_system = None
personas_data = None
processed_data = None

def initialize_system():
    """Initialize the entire system"""
    global processor, persona_extractor, rag_system, personas_data, processed_data
    
    logger.info("Initializing RAG system...")
    
    try:
        # Initialize processor
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'conversations.csv')
        processor = ConversationProcessor(csv_path)
        
        # Process conversations
        logger.info("Processing conversations...")
        processed_data = processor.process()
        
        # Extract personas
        logger.info("Extracting personas...")
        persona_extractor = PersonaExtractor()
        personas_data = persona_extractor.extract_persona(processed_data['messages'])
        
        # Initialize RAG system
        logger.info("Building RAG index...")
        rag_system = RAGSystem(use_embeddings=False)  # Use TF-IDF to avoid model loading delays
        rag_system.build_index(
            processed_data['messages'],
            processed_data['topics'],
            processed_data['checkpoints']
        )
        
        logger.info("System initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    if not processed_data:
        return jsonify({'error': 'System not initialized'}), 500
    
    return jsonify({
        'total_messages': processed_data['total_messages'],
        'total_topics': processed_data['total_topics'],
        'total_checkpoints': processed_data['total_checkpoints'],
        'conversations_count': len(processed_data['conversations']),
        'system_initialized': True
    })

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Get all detected topics"""
    if not processed_data:
        return jsonify({'error': 'System not initialized'}), 500
    
    topics_list = []
    for topic in processed_data['topics']:
        topics_list.append({
            'id': topic.get('topic_id', 0),
            'start_message': topic.get('start_idx', 0),
            'end_message': topic.get('end_idx', 0),
            'num_messages': len(topic.get('messages', [])),
            'keywords': topic.get('keywords', [])
        })
    
    return jsonify({
        'topics': topics_list,
        'total': len(topics_list)
    })

@app.route('/api/checkpoints', methods=['GET'])
def get_checkpoints():
    """Get all 100-message checkpoints"""
    if not processed_data:
        return jsonify({'error': 'System not initialized'}), 500
    
    checkpoints_list = []
    for cp in processed_data['checkpoints']:
        checkpoints_list.append({
            'id': cp.get('checkpoint_id', 0),
            'start_message': cp.get('start_msg', 0),
            'end_message': cp.get('end_msg', 0),
            'num_messages': cp.get('num_messages', 0),
            'summary': cp.get('summary', '')
        })
    
    return jsonify({
        'checkpoints': checkpoints_list,
        'total': len(checkpoints_list)
    })

@app.route('/api/retrieve', methods=['POST'])
def retrieve_context():
    """Retrieve context for a query"""
    if not rag_system or not processed_data:
        return jsonify({'error': 'System not initialized'}), 500
    
    data = request.get_json()
    query = data.get('query', '')
    top_k = data.get('top_k', 5)
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    results = rag_system.retrieve(query, top_k=top_k)
    
    return jsonify({
        'success': True,
        'query': query,
        'results': {
            'messages': results.get('messages', []),
            'topics': results.get('topics', []),
            'checkpoints': results.get('checkpoints', []),
            'context': results.get('context', {})
        }
    })

@app.route('/api/persona', methods=['GET'])
def get_persona():
    """Get extracted persona data"""
    if not personas_data:
        return jsonify({'error': 'System not initialized'}), 500
    
    return jsonify({
        'personas': personas_data,
        'total_users': len(personas_data)
    })

@app.route('/api/persona/<user_id>', methods=['GET'])
def get_user_persona(user_id):
    """Get specific user's persona"""
    if not personas_data:
        return jsonify({'error': 'System not initialized'}), 500
    
    key = f'user_{user_id}' if not user_id.startswith('user_') else user_id
    
    if key not in personas_data:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'user': user_id,
        'persona': personas_data[key]
    })

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """Main chatbot endpoint"""
    if not rag_system or not personas_data or not processed_data:
        return jsonify({'error': 'System not initialized'}), 500
    
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'Question required'}), 400
    
    # Retrieve context
    context = rag_system.retrieve(question, top_k=3)
    
    # Generate answer
    answer = generate_answer(question, context, personas_data)
    
    return jsonify({
        'question': question,
        'answer': answer,
        'context_used': {
            'messages_found': len(context['messages']),
            'topics_found': len(context['topics']),
            'checkpoints_found': len(context['checkpoints'])
        }
    })

def generate_answer(question: str, context: Dict, personas: Dict) -> str:
    """Generate answer using RAG and persona data"""
    
    question_lower = question.lower()
    
    # Determine question type
    if 'person' in question_lower or 'kind of' in question_lower or 'like' in question_lower:
        # Personality question
        return generate_personality_answer(question, personas)
    elif 'habit' in question_lower or 'do' in question_lower:
        # Habit question
        return generate_habit_answer(question, personas, context)
    elif 'talk' in question_lower or 'communicate' in question_lower or 'write' in question_lower:
        # Communication style question
        return generate_communication_answer(question, personas)
    else:
        # General question - use RAG context
        return generate_general_answer(question, context, personas)

def generate_personality_answer(question: str, personas: Dict) -> str:
    """Generate answer about personality"""
    answers = []
    
    for user_key, persona in personas.items():
        traits = persona.get('personality_traits', {})
        interests = persona.get('personal_facts', {}).get('interests', [])
        
        trait_descriptions = []
        for trait, data in traits.items():
            if isinstance(data, dict) and data.get('frequency', 0) > 2:
                trait_descriptions.append(trait)
        
        description = f"{user_key.replace('_', ' ').title()} is "
        
        # Build description
        parts = []
        if traits.get('positive', {}).get('frequency', 0) > 0:
            parts.append("generally positive")
        if traits.get('emotional', {}).get('frequency', 0) > 2:
            parts.append("emotionally expressive")
        if traits.get('humorous', {}).get('frequency', 0) > 2:
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
    """Generate answer about habits"""
    answers = []
    
    for user_key, persona in personas.items():
        habits = persona.get('habits', {})
        mentioned_habits = [h for h, d in habits.items() if d.get('mentioned')]
        
        if mentioned_habits:
            habit_text = f"{user_key.replace('_', ' ').title()} enjoys: {', '.join(mentioned_habits[:5])}."
            answers.append(habit_text)
    
    # Add context from messages
    if context.get('messages'):
        msg_preview = context['messages'][0].get('text', '')[:100]
        if msg_preview:
            answers.append(f"Example: \"{msg_preview}...\"")
    
    return " ".join(answers) if answers else "Limited habit information available."

def generate_communication_answer(question: str, personas: Dict) -> str:
    """Generate answer about communication style"""
    answers = []
    
    for user_key, persona in personas.items():
        comm = persona.get('communication_style', {})
        
        style_desc = f"{user_key.replace('_', ' ').title()} "
        
        parts = []
        if comm.get('tone') == 'casual':
            parts.append("uses a casual tone")
        elif comm.get('tone') == 'formal':
            parts.append("uses a formal tone")
        
        if comm.get('expressiveness') == 'high':
            parts.append("is highly expressive with frequent exclamations")
        
        avg_len = comm.get('average_message_length', 0)
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
    """Generate general answer using RAG"""
    
    answer_parts = []
    
    # Use retrieved messages
    if context.get('messages'):
        messages_text = []
        for msg in context['messages'][:2]:
            messages_text.append(msg.get('text', ''))
        if messages_text:
            answer_parts.append(f"Based on conversations: {' '.join(messages_text[:100])}")
    
    # Add context summary
    context_summary = context.get('context', {}).get('summary', '')
    if context_summary:
        answer_parts.append(f"Context: {context_summary}")
    
    if answer_parts:
        return " ".join(answer_parts)
    else:
        return "I found limited information relevant to your question in the conversations."

@app.route('/api/chat', methods=['POST'])
def chat():
    """Simple chat endpoint (similar to chatbot but simpler)"""
    if not processed_data:
        return jsonify({'error': 'System not initialized'}), 500
    
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'Message required'}), 400
    
    # Use RAG to retrieve relevant context
    if rag_system:
        context = rag_system.retrieve(message, top_k=2)
        response = generate_answer(message, context, personas_data)
    else:
        response = "System not fully initialized"
    
    return jsonify({
        'user_message': message,
        'bot_response': response,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        'name': 'Conversation Analysis RAG System',
        'version': '1.0.0',
        'endpoints': {
            'GET /health': 'Health check',
            'GET /api/stats': 'System statistics',
            'GET /api/topics': 'Get detected topics',
            'GET /api/checkpoints': 'Get 100-message checkpoints',
            'POST /api/retrieve': 'Retrieve context for query',
            'GET /api/persona': 'Get all personas',
            'GET /api/persona/<user_id>': 'Get specific user persona',
            'POST /api/chatbot': 'Main chatbot endpoint',
            'POST /api/chat': 'Simple chat endpoint'
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize system on startup
    logger.info("Starting Flask app...")
    
    if initialize_system():
        logger.info("System initialized, starting server...")
        # For production, use gunicorn: gunicorn -w 4 app:app
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize system")
        sys.exit(1)
