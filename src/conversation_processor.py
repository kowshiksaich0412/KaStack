import csv
import json
import re
from typing import List, Dict, Tuple
from collections import defaultdict
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ConversationParser:
    """Parse and structure conversations from CSV"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.conversations = []
        self.all_messages = []
        self.message_count = 0
        
    def parse(self) -> List[Dict]:
        """Parse conversations from CSV file"""
        conversations = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if not row or not row[0].strip():
                    continue
                    
                conversation_text = row[0]
                messages = self._extract_messages(conversation_text)
                
                if messages:
                    conversations.append({
                        'conversation_id': row_idx,
                        'day': row_idx,
                        'messages': messages,
                        'num_messages': len(messages)
                    })
                    self.all_messages.extend(messages)
                    
        self.conversations = conversations
        self.message_count = len(self.all_messages)
        return conversations
    
    def _extract_messages(self, text: str) -> List[Dict]:
        """Extract individual messages from conversation text"""
        messages = []
        
        # Split by "User X:" pattern
        pattern = r'User\s+(\d+):\s*(.+?)(?=User\s+\d+:|$)'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            user_id = int(match.group(1))
            message_text = match.group(2).strip()
            
            if message_text:
                messages.append({
                    'user_id': user_id,
                    'text': message_text,
                    'message_idx': len(messages),
                    'tokens': message_text.lower().split()
                })
        
        return messages
    
    def get_flattened_messages(self) -> List[Dict]:
        """Get all messages in chronological order"""
        messages = []
        global_idx = 0
        
        for conv in self.conversations:
            for msg in conv['messages']:
                msg_copy = msg.copy()
                msg_copy['global_idx'] = global_idx
                messages.append(msg_copy)
                global_idx += 1
                
        return messages


class TopicDetector:
    """Detect topic changes in conversations using semantic similarity"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        
    def detect_topics(self, messages: List[Dict], similarity_threshold: float = 0.4) -> List[Dict]:
        """
        Detect topic changes using semantic similarity
        Topics change when similarity drops below threshold
        """
        if len(messages) < self.window_size:
            return [{
                'topic_id': 0,
                'start_idx': 0,
                'end_idx': len(messages) - 1,
                'messages': messages,
                'keywords': self._extract_keywords(messages)
            }]
        
        # Extract text for similarity analysis
        texts = [msg['text'] for msg in messages]
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
        except:
            # Fallback if vectorization fails
            return [{
                'topic_id': 0,
                'start_idx': 0,
                'end_idx': len(messages) - 1,
                'messages': messages,
                'keywords': self._extract_keywords(messages)
            }]
        
        # Calculate sliding window similarities
        topics = []
        current_topic_start = 0
        current_topic_messages = []
        
        for i in range(len(messages)):
            current_topic_messages.append(messages[i])
            
            # Check for topic shift every window_size messages
            if i > 0 and i % self.window_size == 0:
                # Calculate similarity between current window and next
                if i + self.window_size < len(messages):
                    try:
                        curr_window = tfidf_matrix[max(0, i-self.window_size):i]
                        next_window = tfidf_matrix[i:min(i+self.window_size, len(messages))]
                        
                        if curr_window.shape[0] > 0 and next_window.shape[0] > 0:
                            similarity = cosine_similarity(
                                curr_window.mean(axis=0),
                                next_window.mean(axis=0)
                            )[0, 0]
                            
                            # Topic shift detected
                            if similarity < similarity_threshold:
                                topics.append({
                                    'topic_id': len(topics),
                                    'start_idx': current_topic_start,
                                    'end_idx': i - 1,
                                    'messages': current_topic_messages[:-1],
                                    'keywords': self._extract_keywords(current_topic_messages[:-1]),
                                    'similarity': similarity
                                })
                                current_topic_start = i
                                current_topic_messages = [messages[i]]
                    except:
                        pass
        
        # Add final topic
        if current_topic_messages:
            topics.append({
                'topic_id': len(topics),
                'start_idx': current_topic_start,
                'end_idx': len(messages) - 1,
                'messages': current_topic_messages,
                'keywords': self._extract_keywords(current_topic_messages),
            })
        
        # Ensure we have at least one topic
        if not topics:
            topics = [{
                'topic_id': 0,
                'start_idx': 0,
                'end_idx': len(messages) - 1,
                'messages': messages,
                'keywords': self._extract_keywords(messages)
            }]
        
        return topics
    
    def _extract_keywords(self, messages: List[Dict], top_k: int = 5) -> List[str]:
        """Extract top keywords from a group of messages"""
        if not messages:
            return []
        
        # Combine all text
        combined_text = ' '.join([msg['text'] for msg in messages]).lower()
        
        # Simple keyword extraction based on frequency
        words = combined_text.split()
        stop_words = set(stopwords.words('english'))
        
        # Filter stopwords and short words
        words = [w for w in words if w not in stop_words and len(w) > 3 and w.isalpha()]
        
        # Count frequencies
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [kw[0] for kw in top_keywords]


class MessageCheckpointGenerator:
    """Generate checkpoints every 100 messages"""
    
    def __init__(self, checkpoint_interval: int = 100):
        self.checkpoint_interval = checkpoint_interval
        
    def generate_checkpoints(self, messages: List[Dict]) -> List[Dict]:
        """Generate checkpoints every N messages"""
        checkpoints = []
        
        for i in range(0, len(messages), self.checkpoint_interval):
            end_idx = min(i + self.checkpoint_interval, len(messages))
            checkpoint_messages = messages[i:end_idx]
            
            checkpoints.append({
                'checkpoint_id': len(checkpoints),
                'start_msg': i,
                'end_msg': end_idx - 1,
                'num_messages': len(checkpoint_messages),
                'messages': checkpoint_messages,
                'summary': self._generate_summary(checkpoint_messages)
            })
        
        return checkpoints
    
    def _generate_summary(self, messages: List[Dict], max_sentences: int = 3) -> str:
        """Generate summary of messages"""
        if not messages:
            return "No messages"
        
        # Extract sentences
        all_text = ' '.join([msg['text'] for msg in messages])
        sentences = sent_tokenize(all_text)
        
        # Return first N sentences as summary
        summary = ' '.join(sentences[:max_sentences])
        return summary if summary else "Conversation checkpoint"


class ConversationProcessor:
    """Main processor combining all components"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.parser = ConversationParser(csv_path)
        self.topic_detector = TopicDetector()
        self.checkpoint_generator = MessageCheckpointGenerator()
        
    def process(self) -> Dict:
        """Process entire conversation dataset"""
        # Parse conversations
        print("Parsing conversations...")
        self.parser.parse()
        messages = self.parser.get_flattened_messages()
        print(f"Total messages: {len(messages)}")
        
        # Detect topics
        print("Detecting topics...")
        topics = self.topic_detector.detect_topics(messages)
        print(f"Topics detected: {len(topics)}")
        
        # Generate checkpoints
        print("Generating 100-message checkpoints...")
        checkpoints = self.checkpoint_generator.generate_checkpoints(messages)
        print(f"Checkpoints created: {len(checkpoints)}")
        
        return {
            'messages': messages,
            'topics': topics,
            'checkpoints': checkpoints,
            'conversations': self.parser.conversations,
            'total_messages': len(messages),
            'total_topics': len(topics),
            'total_checkpoints': len(checkpoints)
        }
    
    def save_processed_data(self, output_dir: str = 'data'):
        """Save processed data to JSON files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        result = self.process()
        
        # Save messages
        with open(f'{output_dir}/messages.json', 'w') as f:
            json.dump(result['messages'], f, indent=2)
        
        # Save topics with summaries
        topics_output = []
        for topic in result['topics']:
            topics_output.append({
                'topic_id': topic['topic_id'],
                'start_msg': topic['start_idx'],
                'end_msg': topic['end_idx'],
                'num_messages': len(topic['messages']),
                'keywords': topic['keywords'],
                'summary': self._generate_topic_summary(topic['messages'])
            })
        
        with open(f'{output_dir}/topics.json', 'w') as f:
            json.dump(topics_output, f, indent=2)
        
        # Save checkpoints
        checkpoints_output = []
        for cp in result['checkpoints']:
            checkpoints_output.append({
                'checkpoint_id': cp['checkpoint_id'],
                'start_msg': cp['start_msg'],
                'end_msg': cp['end_msg'],
                'num_messages': cp['num_messages'],
                'summary': cp['summary']
            })
        
        with open(f'{output_dir}/checkpoints.json', 'w') as f:
            json.dump(checkpoints_output, f, indent=2)
        
        print(f"Data saved to {output_dir}/")
        return result
    
    def _generate_topic_summary(self, messages: List[Dict], max_length: int = 200) -> str:
        """Generate summary for a topic"""
        if not messages:
            return "Empty topic"
        
        combined_text = ' '.join([msg['text'] for msg in messages])
        
        # Extract key sentences
        sentences = sent_tokenize(combined_text)
        summary = ' '.join(sentences[:2])  # First 2 sentences
        
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary if summary else "Topic discussion"


if __name__ == "__main__":
    processor = ConversationProcessor('conversations.csv')
    result = processor.save_processed_data()
    
    print("\n" + "="*50)
    print("Processing Complete!")
    print("="*50)
    print(f"Total Messages: {result['total_messages']}")
    print(f"Total Topics: {result['total_topics']}")
    print(f"Total Checkpoints: {result['total_checkpoints']}")
