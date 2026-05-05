import json
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import os

class RAGSystem:
    """Retrieval-Augmented Generation System for conversations"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_embeddings: bool = False):
        """
        Initialize RAG system
        model_name: sentence transformer model
        use_embeddings: if True, use sentence transformers, else use TF-IDF
        """
        self.use_embeddings = use_embeddings
        
        if use_embeddings:
            try:
                self.model = SentenceTransformer(model_name)
            except:
                print("Warning: Could not load sentence transformer, falling back to TF-IDF")
                self.use_embeddings = False
                self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        else:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        self.messages = []
        self.topics = []
        self.checkpoints = []
        self.embeddings_cache = {}
        self.fitted = False
        
    def build_index(self, messages: List[Dict], topics: List[Dict], checkpoints: List[Dict]):
        """Build RAG index from messages, topics, and checkpoints"""
        self.messages = messages
        self.topics = topics
        self.checkpoints = checkpoints
        
        if self.use_embeddings:
            self._build_embedding_index()
        else:
            self._build_tfidf_index()
        
        self.fitted = True
        print(f"RAG index built: {len(messages)} messages, {len(topics)} topics, {len(checkpoints)} checkpoints")
    
    def _build_tfidf_index(self):
        """Build TF-IDF index"""
        message_texts = [msg.get('text', '') for msg in self.messages]
        self.tfidf_matrix = self.vectorizer.fit_transform(message_texts)
        
        # Also vectorize topic summaries and checkpoints
        topic_texts = [self._get_topic_summary(t) for t in self.topics]
        checkpoint_texts = [cp.get('summary', '') for cp in self.checkpoints]
        
        self.topic_tfidf = self.vectorizer.transform(topic_texts) if topic_texts else None
        self.checkpoint_tfidf = self.vectorizer.transform(checkpoint_texts) if checkpoint_texts else None
    
    def _build_embedding_index(self):
        """Build embedding index using sentence transformers"""
        message_texts = [msg.get('text', '') for msg in self.messages]
        
        print("Encoding message embeddings...")
        self.message_embeddings = self.model.encode(message_texts, convert_to_numpy=True)
        
        topic_texts = [self._get_topic_summary(t) for t in self.topics]
        print("Encoding topic embeddings...")
        self.topic_embeddings = self.model.encode(topic_texts, convert_to_numpy=True) if topic_texts else None
        
        checkpoint_texts = [cp.get('summary', '') for cp in self.checkpoints]
        print("Encoding checkpoint embeddings...")
        self.checkpoint_embeddings = self.model.encode(checkpoint_texts, convert_to_numpy=True) if checkpoint_texts else None
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """
        Retrieve relevant messages, topics, and checkpoints
        Returns both semantic matches and topic/checkpoint context
        """
        if not self.fitted:
            return {'error': 'RAG system not initialized'}
        
        results = {
            'query': query,
            'messages': [],
            'topics': [],
            'checkpoints': [],
            'context': {}
        }
        
        # Retrieve relevant messages
        message_results = self._retrieve_messages(query, top_k)
        results['messages'] = message_results
        
        # Retrieve relevant topics
        if self.topics:
            topic_results = self._retrieve_topics(query, top_k // 2)
            results['topics'] = topic_results
        
        # Retrieve relevant checkpoints
        if self.checkpoints:
            checkpoint_results = self._retrieve_checkpoints(query, top_k // 2)
            results['checkpoints'] = checkpoint_results
        
        # Build context from results
        results['context'] = self._build_context(results)
        
        return results
    
    def _retrieve_messages(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve most relevant messages"""
        if self.use_embeddings:
            return self._retrieve_messages_embedding(query, top_k)
        else:
            return self._retrieve_messages_tfidf(query, top_k)
    
    def _retrieve_messages_tfidf(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using TF-IDF"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Filter out very low matches
                results.append({
                    'message_idx': self.messages[idx].get('global_idx', idx),
                    'text': self.messages[idx].get('text', ''),
                    'user_id': self.messages[idx].get('user_id', 1),
                    'similarity': float(similarities[idx])
                })
        
        return results
    
    def _retrieve_messages_embedding(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using sentence embeddings"""
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.message_embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Filter threshold
                results.append({
                    'message_idx': self.messages[idx].get('global_idx', idx),
                    'text': self.messages[idx].get('text', ''),
                    'user_id': self.messages[idx].get('user_id', 1),
                    'similarity': float(similarities[idx])
                })
        
        return results
    
    def _retrieve_topics(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve relevant topics"""
        if not self.topics:
            return []
        
        if self.use_embeddings and self.topic_embeddings is not None:
            query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
            similarities = cosine_similarity([query_embedding], self.topic_embeddings)[0]
        else:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.topic_tfidf)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                topic = self.topics[idx]
                results.append({
                    'topic_id': topic.get('topic_id', idx),
                    'keywords': topic.get('keywords', []),
                    'message_range': f"{topic.get('start_idx', 0)}-{topic.get('end_idx', 0)}",
                    'similarity': float(similarities[idx]),
                    'summary': self._get_topic_summary(topic)
                })
        
        return results
    
    def _retrieve_checkpoints(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve relevant checkpoints"""
        if not self.checkpoints:
            return []
        
        if self.use_embeddings and self.checkpoint_embeddings is not None:
            query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
            similarities = cosine_similarity([query_embedding], self.checkpoint_embeddings)[0]
        else:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.checkpoint_tfidf)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                checkpoint = self.checkpoints[idx]
                results.append({
                    'checkpoint_id': checkpoint.get('checkpoint_id', idx),
                    'message_range': f"{checkpoint.get('start_msg', 0)}-{checkpoint.get('end_msg', 0)}",
                    'num_messages': checkpoint.get('num_messages', 0),
                    'summary': checkpoint.get('summary', ''),
                    'similarity': float(similarities[idx])
                })
        
        return results
    
    def _get_topic_summary(self, topic: Dict) -> str:
        """Get summary text from topic"""
        if 'summary' in topic:
            return topic['summary']
        elif 'keywords' in topic:
            return ' '.join(topic['keywords'])
        else:
            return ''
    
    def _build_context(self, results: Dict) -> Dict:
        """Build complete context from retrieval results"""
        context = {
            'num_relevant_messages': len(results['messages']),
            'num_relevant_topics': len(results['topics']),
            'num_relevant_checkpoints': len(results['checkpoints']),
            'summary': self._generate_context_summary(results)
        }
        
        return context
    
    def _generate_context_summary(self, results: Dict) -> str:
        """Generate summary of retrieval context"""
        summary = []
        
        if results['messages']:
            summary.append(f"Found {len(results['messages'])} relevant messages")
        
        if results['topics']:
            topic_keywords = []
            for topic in results['topics']:
                topic_keywords.extend(topic.get('keywords', []))
            if topic_keywords:
                summary.append(f"Related topics: {', '.join(set(topic_keywords[:5]))}")
        
        if results['checkpoints']:
            summary.append(f"Relevant checkpoints: {len(results['checkpoints'])}")
        
        return '; '.join(summary) if summary else "No relevant context found"
    
    def save_index(self, path: str = 'rag_index'):
        """Save RAG index to disk"""
        os.makedirs(path, exist_ok=True)
        
        index_data = {
            'messages': self.messages,
            'topics': self.topics,
            'checkpoints': self.checkpoints,
            'use_embeddings': self.use_embeddings
        }
        
        with open(f'{path}/index.json', 'w') as f:
            json.dump(index_data, f, indent=2)
        
        if self.use_embeddings and hasattr(self, 'message_embeddings'):
            np.save(f'{path}/message_embeddings.npy', self.message_embeddings)
            if self.topic_embeddings is not None:
                np.save(f'{path}/topic_embeddings.npy', self.topic_embeddings)
            if self.checkpoint_embeddings is not None:
                np.save(f'{path}/checkpoint_embeddings.npy', self.checkpoint_embeddings)
        else:
            pickle.dump(self.vectorizer, open(f'{path}/vectorizer.pkl', 'wb'))
        
        print(f"RAG index saved to {path}/")
    
    def load_index(self, path: str = 'rag_index'):
        """Load RAG index from disk"""
        with open(f'{path}/index.json', 'r') as f:
            index_data = json.load(f)
        
        self.messages = index_data['messages']
        self.topics = index_data['topics']
        self.checkpoints = index_data['checkpoints']
        self.use_embeddings = index_data['use_embeddings']
        
        if self.use_embeddings:
            self.message_embeddings = np.load(f'{path}/message_embeddings.npy')
            if os.path.exists(f'{path}/topic_embeddings.npy'):
                self.topic_embeddings = np.load(f'{path}/topic_embeddings.npy')
            if os.path.exists(f'{path}/checkpoint_embeddings.npy'):
                self.checkpoint_embeddings = np.load(f'{path}/checkpoint_embeddings.npy')
        else:
            self.vectorizer = pickle.load(open(f'{path}/vectorizer.pkl', 'rb'))
            self._build_tfidf_index()
        
        self.fitted = True
        print(f"RAG index loaded from {path}/")


if __name__ == "__main__":
    # Test RAG system
    print("RAG System initialized for testing")
