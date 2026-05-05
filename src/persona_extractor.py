import json
import re
from typing import List, Dict
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class PersonaExtractor:
    """Extract user persona from conversations"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
        # Patterns and keywords for extraction
        self.habit_patterns = {
            'sleep': r'(sleep|wake|alarm|morning|night|late|early)',
            'food': r'(eat|food|cook|recipe|pizza|coffee|tea|drink|meal|breakfast|lunch|dinner)',
            'exercise': r'(run|walk|gym|yoga|sport|exercise|hike|bike|swim)',
            'reading': r'(read|book|novel|author|story)',
            'music': r'(sing|music|song|concert|guitar|piano|instrument)',
            'work': r'(work|job|career|profession|business)',
            'social': r'(friend|family|parent|sibling|relationship|dating|social)',
            'hobbies': r'(hobby|hobby|gaming|game|video|watch|movie)',
            'pet': r'(dog|cat|pet|animal)',
            'travel': r'(travel|trip|visit|city|country|beach|mountain)',
        }
        
        self.personality_indicators = {
            'positive': r'(love|awesome|amazing|great|wonderful|fantastic|cool|nice)',
            'emotional': r'(feel|sad|happy|excited|nervous|worried|stressed)',
            'humorous': r'(!+|haha|lol|funny|joke)',
            'thoughtful': r'(think|believe|hope|sure|grateful|appreciate)',
            'energetic': r'(excited|enthusiastic|active|busy)',
            'calm': r'(relax|peace|peaceful|calm|stress)',
        }
        
    def extract_persona(self, messages: List[Dict]) -> Dict:
        """Extract complete persona from messages"""
        
        # Combine all messages for analysis
        user_messages = defaultdict(list)
        for msg in messages:
            user_id = msg.get('user_id', 1)
            user_messages[user_id].append(msg['text'])
        
        # Extract for each user
        personas = {}
        for user_id, texts in user_messages.items():
            personas[f'user_{user_id}'] = self._extract_user_persona(texts)
        
        return personas
    
    def _extract_user_persona(self, messages: List[str]) -> Dict:
        """Extract persona for a single user"""
        
        combined_text = ' '.join(messages).lower()
        
        return {
            'habits': self._extract_habits(combined_text, messages),
            'personal_facts': self._extract_personal_facts(messages),
            'personality_traits': self._extract_personality_traits(combined_text, messages),
            'communication_style': self._extract_communication_style(messages),
            'topics_of_interest': self._extract_topics(combined_text),
            'relationships': self._extract_relationships(messages),
            'sentiment_summary': self._analyze_sentiment(messages)
        }
    
    def _extract_habits(self, combined_text: str, messages: List[str]) -> Dict:
        """Extract habits from conversations"""
        habits = {}
        
        for habit_type, pattern in self.habit_patterns.items():
            matches = re.findall(pattern, combined_text)
            if matches:
                habits[habit_type] = {
                    'mentioned': True,
                    'frequency': len(matches),
                    'examples': self._find_example_sentences(pattern, messages, limit=2)
                }
            else:
                habits[habit_type] = {
                    'mentioned': False,
                    'frequency': 0,
                    'examples': []
                }
        
        return habits
    
    def _extract_personal_facts(self, messages: List[str]) -> Dict:
        """Extract factual personal information"""
        facts = {
            'relationships': self._extract_relationships(messages),
            'occupation': self._extract_occupation(messages),
            'locations': self._extract_locations(messages),
            'interests': self._extract_main_interests(messages),
        }
        return facts
    
    def _extract_personality_traits(self, combined_text: str, messages: List[str]) -> Dict:
        """Extract personality traits from text signals"""
        traits = {}
        
        for trait_type, pattern in self.personality_indicators.items():
            matches = re.findall(pattern, combined_text)
            if matches:
                traits[trait_type] = {
                    'score': len(matches) / max(len(messages), 1),
                    'frequency': len(matches),
                    'examples': self._find_example_sentences(pattern, messages, limit=2)
                }
        
        # Analyze emoji usage
        emoji_count = len(re.findall(r'[😀-🙏🌀-🗿]|[!?]{2,}', ' '.join(messages)))
        traits['expressive'] = {
            'score': emoji_count / max(len(messages), 1),
            'frequency': emoji_count
        }
        
        return traits
    
    def _extract_communication_style(self, messages: List[str]) -> Dict:
        """Analyze communication style"""
        avg_length = sum(len(m.split()) for m in messages) / max(len(messages), 1)
        
        # Count exclamations and questions
        exclamations = sum(m.count('!') for m in messages)
        questions = sum(m.count('?') for m in messages)
        
        # Check for short messages
        short_messages = sum(1 for m in messages if len(m.split()) < 5)
        
        # Check for formal vs casual
        formal_words = len(re.findall(r'\b(indeed|therefore|moreover|furthermore)\b', ' '.join(messages).lower()))
        casual_words = len(re.findall(r"\b(lol|haha|gonna|wanna|kinda)\b", ' '.join(messages).lower()))
        
        return {
            'average_message_length': round(avg_length, 2),
            'message_count': len(messages),
            'exclamations': exclamations,
            'questions': questions,
            'short_message_percentage': round((short_messages / max(len(messages), 1)) * 100, 1),
            'tone': 'casual' if casual_words > formal_words else 'formal' if formal_words > 0 else 'neutral',
            'expressiveness': 'high' if (exclamations + questions) > len(messages) * 0.3 else 'low',
        }
    
    def _extract_relationships(self, messages: List[str]) -> List[str]:
        """Extract relationship mentions"""
        relationships = []
        rel_pattern = r'\b(mom|mother|dad|father|brother|sister|wife|husband|boyfriend|girlfriend|son|daughter|friend|friend|colleague|boss|family)\b'
        
        for msg in messages:
            matches = re.findall(rel_pattern, msg.lower())
            relationships.extend(matches)
        
        return list(set(relationships))
    
    def _extract_occupation(self, messages: List[str]) -> List[str]:
        """Extract job/occupation mentions"""
        occupations = []
        job_pattern = r'\b(teacher|engineer|doctor|nurse|artist|programmer|student|manager|developer|designer|chef|nurse|firefighter|writer|librarian|murallist|muralist|barista|trainer|emc|tutor|juggler)\b'
        
        for msg in messages:
            matches = re.findall(job_pattern, msg.lower())
            occupations.extend(matches)
        
        return list(set(occupations))
    
    def _extract_locations(self, messages: List[str]) -> List[str]:
        """Extract location mentions"""
        locations = []
        # Common locations mentioned in conversations
        location_pattern = r'\b(Portland|Oregon|Everglades|Georgia|Maldives|Florida Keys|city|town|country|beach|mountain|park|home|house)\b'
        
        for msg in messages:
            matches = re.findall(location_pattern, msg)
            locations.extend(matches)
        
        return list(set(locations))
    
    def _extract_topics(self, combined_text: str) -> List[str]:
        """Extract main topics of interest"""
        topics = []
        topic_keywords = {
            'sports': ['soccer', 'football', 'basketball', 'tennis', 'golf', 'sports'],
            'arts': ['art', 'music', 'painting', 'drawing', 'photography'],
            'outdoor': ['hiking', 'camping', 'fishing', 'beach', 'mountain', 'trail'],
            'tech': ['coding', 'programming', 'computer', 'software', 'tech'],
            'food': ['cooking', 'recipe', 'food', 'restaurant', 'meal'],
            'reading': ['book', 'reading', 'novel', 'author'],
            'social': ['friends', 'family', 'social', 'party'],
            'fitness': ['exercise', 'gym', 'yoga', 'run', 'walk'],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in combined_text for kw in keywords):
                topics.append(topic)
        
        return topics
    
    def _extract_main_interests(self, messages: List[str]) -> List[str]:
        """Extract main interests/hobbies"""
        interests = []
        combined_text = ' '.join(messages).lower()
        
        interest_phrases = {
            'cooking': r'(cook|recipe|meal|food|chef)',
            'reading': r'(read|book|novel|author|story)',
            'music': r'(sing|music|song|guitar|piano|band)',
            'sports': r'(play|soccer|football|basketball)',
            'outdoor_activities': r'(hike|camp|fish|climb|trail)',
            'creative': r'(art|paint|draw|create|design)',
            'technology': r'(code|program|computer|software)',
            'social': r'(friends|family|social|party)',
        }
        
        for interest, pattern in interest_phrases.items():
            if re.search(pattern, combined_text):
                interests.append(interest)
        
        return interests
    
    def _find_example_sentences(self, pattern: str, messages: List[str], limit: int = 2) -> List[str]:
        """Find example sentences containing pattern"""
        examples = []
        
        for msg in messages:
            if re.search(pattern, msg.lower()):
                # Truncate to reasonable length
                if len(msg) > 100:
                    msg = msg[:100] + "..."
                examples.append(msg)
                if len(examples) >= limit:
                    break
        
        return examples
    
    def _analyze_sentiment(self, messages: List[str]) -> Dict:
        """Analyze overall sentiment"""
        sentiments = []
        
        for msg in messages:
            scores = self.sia.polarity_scores(msg)
            sentiments.append(scores['compound'])
        
        avg_sentiment = sum(sentiments) / max(len(sentiments), 1)
        
        if avg_sentiment > 0.1:
            sentiment_type = 'positive'
        elif avg_sentiment < -0.1:
            sentiment_type = 'negative'
        else:
            sentiment_type = 'neutral'
        
        return {
            'overall': sentiment_type,
            'score': round(avg_sentiment, 2),
            'positivity': round((sum(1 for s in sentiments if s > 0.1) / max(len(sentiments), 1)) * 100, 1),
        }


class PersonaGenerator:
    """Generate persona reports"""
    
    @staticmethod
    def generate_report(persona_data: Dict) -> str:
        """Generate human-readable persona report"""
        report = []
        
        for user, persona in persona_data.items():
            report.append(f"\n{'='*60}")
            report.append(f"PERSONA: {user.upper()}")
            report.append(f"{'='*60}\n")
            
            # Communication Style
            comm = persona.get('communication_style', {})
            report.append("Communication Style:")
            report.append(f"  - Tone: {comm.get('tone', 'unknown')}")
            report.append(f"  - Expressiveness: {comm.get('expressiveness', 'unknown')}")
            report.append(f"  - Average Message Length: {comm.get('average_message_length', 0)} words")
            
            # Personality Traits
            report.append("\nPersonality Traits:")
            traits = persona.get('personality_traits', {})
            for trait, data in traits.items():
                if isinstance(data, dict) and 'frequency' in data:
                    report.append(f"  - {trait.title()}: {data.get('frequency', 0)} mentions")
            
            # Habits
            report.append("\nHabits & Interests:")
            habits = persona.get('habits', {})
            for habit, data in habits.items():
                if data.get('mentioned'):
                    report.append(f"  - {habit.title()}: {data.get('frequency', 0)} mentions")
            
            # Personal Facts
            report.append("\nPersonal Facts:")
            facts = persona.get('personal_facts', {})
            
            if facts.get('occupation'):
                report.append(f"  - Occupations: {', '.join(facts.get('occupation', []))}")
            
            if facts.get('relationships'):
                report.append(f"  - Relationships: {', '.join(facts.get('relationships', []))}")
            
            if facts.get('locations'):
                report.append(f"  - Locations Mentioned: {', '.join(facts.get('locations', []))}")
            
            # Sentiment
            sentiment = persona.get('sentiment_summary', {})
            report.append(f"\nOverall Sentiment: {sentiment.get('overall', 'unknown')} ({sentiment.get('score', 0)})")
        
        return '\n'.join(report)


if __name__ == "__main__":
    # Test with sample messages
    sample_messages = [
        {"user_id": 1, "text": "I love cooking and reading books! Best way to relax."},
        {"user_id": 1, "text": "Just finished a great hike with my dog. Amazing!"},
        {"user_id": 1, "text": "I work as a software engineer and enjoy playing soccer."},
        {"user_id": 2, "text": "I'm a student studying radiology."},
        {"user_id": 2, "text": "I play in a band that my parents don't know about!"},
    ]
    
    extractor = PersonaExtractor()
    personas = extractor.extract_persona(sample_messages)
    
    generator = PersonaGenerator()
    report = generator.generate_report(personas)
    print(report)
    
    # Save as JSON
    with open('persona_data.json', 'w') as f:
        json.dump(personas, f, indent=2)
