import spacy
import nltk
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        self.setup_nltk()
        self.setup_spacy()
    
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logger.warning(f"NLTK setup warning: {e}")
    
    def setup_spacy(self):
        """Load spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            # Fallback to basic processing
            self.nlp = None
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback using NLTK
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
    
    def extract_svo_triplets(self, text: str) -> List[Dict]:
        """Extract Subject-Verb-Object triplets from text"""
        triplets = []
        
        if not self.nlp:
            logger.warning("spaCy not available, using simplified triplet extraction")
            return self._extract_svo_simple(text)
        
        doc = self.nlp(text)
        
        for sent in doc.sents:
            sent_triplets = self._extract_triplets_from_sentence(sent)
            for triplet in sent_triplets:
                triplet['sentence'] = sent.text.strip()
                triplet['sentence_id'] = len([s for s in doc.sents if s.start < sent.start])
            triplets.extend(sent_triplets)
        
        return triplets
    
    def _extract_triplets_from_sentence(self, sent) -> List[Dict]:
        """Extract triplets from a single sentence"""
        triplets = []
        
        # Find the root verb
        root = None
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                root = token
                break
        
        if not root:
            return triplets
        
        # Find subject
        subject = None
        for child in root.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                subject = self._get_full_phrase(child)
                break
        
        # Find object
        obj = None
        for child in root.children:
            if child.dep_ in ["dobj", "pobj", "attr"]:
                obj = self._get_full_phrase(child)
                break
        
        # Create triplet if we have subject and verb
        if subject:
            verb = self._get_full_phrase(root)
            triplet = {
                "subject": subject,
                "verb": verb,
                "object": obj if obj else "NONE",
                "confidence": self._calculate_triplet_confidence(subject, verb, obj)
            }
            triplets.append(triplet)
        
        return triplets
    
    def _get_full_phrase(self, token):
        """Get the full phrase for a token including its children"""
        def get_phrase_tokens(tok):
            tokens = [tok]
            for child in tok.children:
                if child.dep_ in ["det", "amod", "compound", "prep", "pobj"]:
                    tokens.extend(get_phrase_tokens(child))
            return sorted(tokens, key=lambda x: x.i)
        
        phrase_tokens = get_phrase_tokens(token)
        return " ".join([t.text for t in phrase_tokens])
    
    def _calculate_triplet_confidence(self, subject: str, verb: str, obj: str) -> float:
        """Calculate confidence score for a triplet"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on phrase length and complexity
        if len(subject.split()) > 1:
            confidence += 0.1
        if obj and obj != "NONE" and len(obj.split()) > 1:
            confidence += 0.1
        if len(verb.split()) > 1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_svo_simple(self, text: str) -> List[Dict]:
        """Simplified SVO extraction without spaCy"""
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk import pos_tag
        
        triplets = []
        sentences = sent_tokenize(text)
        
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence)
            pos_tags = pos_tag(words)
            
            # Simple pattern matching for SVO
            subject = None
            verb = None
            obj = None
            
            for j, (word, tag) in enumerate(pos_tags):
                if tag in ['NN', 'NNP', 'NNS', 'NNPS'] and not subject:
                    subject = word
                elif tag.startswith('VB') and subject and not verb:
                    verb = word
                elif tag in ['NN', 'NNP', 'NNS', 'NNPS'] and verb and not obj:
                    obj = word
                    break
            
            if subject and verb:
                triplets.append({
                    "subject": subject,
                    "verb": verb,
                    "object": obj if obj else "NONE",
                    "sentence": sentence,
                    "sentence_id": i,
                    "confidence": 0.6
                })
        
        return triplets
    
    def load_gutenberg_text(self, book_id: str = "2600") -> str:
        """Load text from Project Gutenberg"""
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            text = response.text
            
            # Clean the text - remove Project Gutenberg header/footer
            start_markers = [
                "*** START OF THE PROJECT GUTENBERG EBOOK",
                "*** START OF THIS PROJECT GUTENBERG EBOOK"
            ]
            end_markers = [
                "*** END OF THE PROJECT GUTENBERG EBOOK",
                "*** END OF THIS PROJECT GUTENBERG EBOOK"
            ]
            
            for marker in start_markers:
                if marker in text:
                    text = text.split(marker)[1]
                    break
            
            for marker in end_markers:
                if marker in text:
                    text = text.split(marker)[0]
                    break
            
            # Clean up formatting
            text = re.sub(r'\r\n', '\n', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()
            
            # Find the actual story beginning - skip table of contents
            # Look for the first substantial paragraph of narrative text
            lines = text.split('\n')
            story_start_idx = 0
            
            for i, line in enumerate(lines):
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                    
                # Look for the beginning of actual narrative content
                # Usually starts with "Well, Prince" or similar opening lines
                if (len(line) > 50 and 
                    not line.isupper() and 
                    not line.startswith(('CHAPTER', 'BOOK', 'PART', 'Contents')) and
                    not re.match(r'^[A-Z\s]{10,}$', line) and
                    ('Prince' in line or 'Well' in line or 'Genoa' in line or
                     line.count('.') > 0 or line.count(',') > 2)):
                    story_start_idx = i
                    break
            
            # Extract the narrative content
            if story_start_idx > 0:
                narrative_lines = lines[story_start_idx:]
                text = '\n'.join(narrative_lines)
            
            # Take a substantial portion of the book for analysis
            # Using 75,000 characters to get multiple chapters with rich content
            if len(text) > 75000:
                # Find a good breaking point (end of sentence)
                break_point = text.rfind('.', 50000, 75000)
                if break_point > 50000:
                    text = text[:break_point + 1]
                else:
                    text = text[:75000] + "..."
            
            return text
            
        except Exception as e:
            logger.error(f"Error loading Gutenberg text: {e}")
            # Return a more substantial sample text as fallback
            return """War and Peace is a novel by Leo Tolstoy that chronicles the French invasion of Russia and the impact of the Napoleonic era on Tsarist society. 

Pierre Bezukhov inherited a vast fortune from his father. The young man struggled with questions of meaning and purpose in life. Natasha Rostova danced gracefully at the ball. Prince Andrei Bolkonsky fought bravely in the battle of Austerlitz. Napoleon Bonaparte led his army across Europe. The French emperor believed he could conquer all of Russia. 

Count Rostov worried about his family's finances. The old man loved his children dearly. Countess Rostova organized magnificent social gatherings. She invited the nobility to her lavish parties. Dolokhov challenged Pierre to a duel. The reckless officer enjoyed gambling and fighting. Princess Marya Bolkonskaya cared for her aging father. The devout woman prayed for peace in troubled times.

Moscow burned during the French occupation. The citizens fled their beloved city. General Kutuzov planned strategic retreats. The wise commander understood Russian winter would defeat the invaders. Soldiers marched through deep snow and bitter cold. Many troops died from hunger and freezing temperatures. The Russian people fought for their homeland's freedom. Patriots resisted the foreign occupation with fierce determination.

Natasha fell in love with Prince Andrei during their engagement. The couple planned a beautiful wedding ceremony. Pierre discovered his wife's infidelity and felt devastated. Helene Kuragina betrayed her husband's trust repeatedly. Count Bezukhov questioned his marriage's true meaning. The philosopher sought spiritual enlightenment through suffering.

Captain Tushin commanded his artillery battery with great skill. The brave officer defended his position against overwhelming odds. Prince Bagration led cavalry charges across bloody battlefields. The general inspired his men through personal courage. Russian soldiers sang folk songs around campfires. The troops shared stories of home and family.

War changed everyone's perspective on life's priorities. Families separated by conflict yearned for reunion. Children grew up without knowing their fathers. Mothers wept for sons who never returned home. The nobility learned that titles meant nothing during crisis. Common people displayed extraordinary heroism under pressure."""