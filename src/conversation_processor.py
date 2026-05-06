import csv
import json
import re
from typing import List, Dict
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


def clean_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"http\S+|www\.\S+", " ", value)
    value = re.sub(r"[^a-z\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


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
    """Detect topics using KMeans clustering on TF-IDF vectors"""
    
    def __init__(self, n_clusters: int = 8):
        """
        Initialize topic detector
        n_clusters: number of topics to detect
        """
        self.n_clusters = n_clusters
        custom_words = [
            "like",
            "yeah",
            "okay",
            "hello",
            "hi",
            "thanks",
            "thank",
            "well",
            "user",
            "name",
            "today",
            "love",
            "good",
            "great",
            "fun",
            "cool",
            "nice",
            "awesome",
            "amazing",
            "best",
            "favorite",
            "really",
            "sounds",
        ]
        self.custom_stopwords = text.ENGLISH_STOP_WORDS.union(custom_words)
        self.stop_words = set(self.custom_stopwords).union(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            stop_words=list(self.custom_stopwords),
            max_df=0.6,
            min_df=10,
            ngram_range=(1, 3),
            max_features=3000,
        )

    def _normalize_text(self, text: str) -> str:
        return clean_text(text)

    def _noun_only_text(self, value: str) -> str:
        """Keep nouns so clusters represent subjects instead of sentiment."""
        cleaned = clean_text(value)
        tokens = re.findall(r"\b[a-z]{3,}\b", cleaned)
        tokens = [token for token in tokens if token not in self.stop_words]

        if not tokens:
            return ""

        tagged_tokens = nltk.pos_tag(tokens)
        nouns = [
            token
            for token, tag in tagged_tokens
            if tag in {"NN", "NNS", "NNP", "NNPS"} and token not in self.stop_words
        ]

        return " ".join(nouns)

    def _target_cluster_count(self, message_count: int) -> int:
        """Choose enough clusters to avoid one-topic collapse while staying stable."""
        if message_count < 2:
            return 1
        return min(self.n_clusters, message_count)
    
    def detect_topics(self, messages: List[Dict]) -> List[Dict]:
        """
        Detect topics using KMeans clustering
        """
        if len(messages) < 2:
            # Too few messages, return all as one topic
            return [{
                'topic_id': 0,
                'start_idx': 0,
                'end_idx': len(messages) - 1,
                'messages': messages,
                'keywords': self._extract_keywords(messages)
            }]
        
        # Extract text for clustering
        texts = [self._noun_only_text(msg.get('text', '')) for msg in messages]
        usable_messages = [(idx, msg, text) for idx, (msg, text) in enumerate(zip(messages, texts)) if text]

        if len(usable_messages) < 2:
            return self._fallback_topic(messages)
        
        try:
            # Build TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform([text for _, _, text in usable_messages])
            
            n_clusters = self._target_cluster_count(len(usable_messages))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tfidf_matrix)
            cluster_keywords = self._extract_cluster_keywords(kmeans, top_k=8)
            
            # Group messages by cluster
            clusters = defaultdict(list)
            for (original_idx, msg, _), label in zip(usable_messages, labels):
                clusters[int(label)].append((original_idx, msg))
            
            # Create topic entries ordered by first message index
            topics = []
            for cluster_id, message_indices in sorted(clusters.items()):
                message_indices.sort(key=lambda x: x[0])
                
                start_idx = message_indices[0][0]
                end_idx = message_indices[-1][0]
                topic_messages = [msg for _, msg in message_indices]
                
                topics.append({
                    'topic_id': len(topics),
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'messages': topic_messages,
                    'keywords': cluster_keywords.get(cluster_id, self._extract_keywords(topic_messages, top_k=8)),
                    'num_messages': len(topic_messages)
                })
            
            # Sort by start_idx
            topics.sort(key=lambda x: x['start_idx'])
            
            return topics if topics else self._fallback_topic(messages)
            
        except Exception as e:
            print(f"Topic detection error: {e}, using fallback")
            return self._fallback_topic(messages)
    
    def _fallback_topic(self, messages: List[Dict]) -> List[Dict]:
        """Fallback when clustering fails"""
        # Split into N roughly equal topics
        n = self._target_cluster_count(len(messages))
        chunk_size = len(messages) // n
        
        topics = []
        for i in range(n):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n - 1 else len(messages)
            
            if end > start:
                chunk = messages[start:end]
                topics.append({
                    'topic_id': i,
                    'start_idx': start,
                    'end_idx': end - 1,
                    'messages': chunk,
                    'keywords': self._extract_keywords(chunk),
                    'num_messages': len(chunk)
                })
        
        return topics if topics else [{
            'topic_id': 0,
            'start_idx': 0,
            'end_idx': len(messages) - 1,
            'messages': messages,
            'keywords': self._extract_keywords(messages),
            'num_messages': len(messages)
        }]
    
    def _extract_cluster_keywords(self, kmeans: KMeans, top_k: int = 8) -> Dict[int, List[str]]:
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_keywords = {}

        for cluster_id, center in enumerate(kmeans.cluster_centers_):
            top_indices = np.argsort(center)[::-1]
            keywords = []
            seen_parts = set()

            for idx in top_indices:
                term = feature_names[idx]
                term_parts = set(term.split())

                if self._is_meaningful_keyword(term) and not term_parts.issubset(seen_parts):
                    keywords.append(term)
                    seen_parts.update(term_parts)
                if len(keywords) >= top_k:
                    break

            cluster_keywords[cluster_id] = keywords

        return cluster_keywords

    def _is_meaningful_keyword(self, term: str) -> bool:
        parts = term.split()
        return all(
            len(part) > 2 and part not in self.stop_words
            for part in parts
        )

    def _extract_keywords(self, messages: List[Dict], top_k: int = 8) -> List[str]:
        """Extract top keywords from messages"""
        if not messages:
            return []
        
        combined_text = self._noun_only_text(' '.join([msg.get('text', '') for msg in messages]))
        words = re.findall(r"\b[a-z]{3,}\b", combined_text)
        
        words = [word for word in words if word not in self.stop_words]
        return [word for word, _ in Counter(words).most_common(top_k)]


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
