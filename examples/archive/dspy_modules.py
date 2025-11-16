"""DSPY Module Classes and Signatures

This module contains the enhanced DSPY module classes and custom signatures
for question answering and sentiment analysis tasks.
"""

try:
    import dspy
except ImportError:
    dspy = None

from typing import Dict, Any


# Set up mock classes for when DSPY is not available

if dspy is None:
    class MockField:
        def __init__(self, desc="", default=None):
            self.desc = desc
            self.default = default
    
    class MockSignature:
        pass
    
    class MockModule:
        def __init__(self):
            pass
        
        def forward(self, **kwargs):
            return MockPrediction(
                answer="Mock forward response",
                reasoning="Mock reasoning"
            )
        
        def __call__(self, **kwargs):
            return self.forward(**kwargs)
    
    class MockPrediction:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Create mock dspy object
    class MockDSPY:
        InputField = MockField
        OutputField = MockField
        Signature = MockSignature
        Module = MockModule
        Prediction = MockPrediction
        
        @staticmethod
        def ChainOfThought(signature_class):
            """Mock ChainOfThought that returns simple predictions."""
            def mock_generator(**kwargs):
                if 'question' in kwargs:
                    return MockPrediction(
                        answer=f"Mock answer to: {kwargs.get('question', 'question')}",
                        reasoning="Mock reasoning (ChainOfThought)")
                elif 'text' in kwargs:
                    return MockPrediction(
                        sentiment="neutral",
                        confidence=0.5,
                        reasoning="Mock sentiment analysis (ChainOfThought)")
                else:
                    return MockPrediction(
                        answer="Mock response",
                        reasoning="Mock reasoning"
                    )
            return mock_generator
    
    dspy = MockDSPY()

# Now define the classes using the available (real or mock) dspy


class QuestionAnswering(dspy.Signature):
    """Answer questions accurately and concisely."""
    question = dspy.InputField(desc="The question to answer")
    context = dspy.InputField(desc="Relevant context information (optional)", default="")
    answer = dspy.OutputField(desc="A clear, accurate answer to the question")


class SentimentClassification(dspy.Signature):
    """Classify text sentiment as positive, negative, or neutral."""
    text = dspy.InputField(desc="The text to analyze")
    sentiment = dspy.OutputField(desc="The sentiment: positive, negative, or neutral")
    confidence = dspy.OutputField(desc="Confidence score from 0 to 1")
    reasoning = dspy.OutputField(desc="Brief reasoning for the sentiment classification")


class SimpleQA(dspy.Module):
    """Enhanced question answering module with proper DSPY signatures."""
    
    def __init__(self):
        super().__init__()
        if dspy and hasattr(dspy, 'ChainOfThought'):
            self.generate_answer = dspy.ChainOfThought(QuestionAnswering)
        else:
            self.generate_answer = lambda **kwargs: dspy.Prediction(
                answer="Mock answer (DSPY not installed)",
                reasoning="Mock reasoning"
            )
    
    def forward(self, question: str, context: str = "") -> dspy.Prediction:
        """Forward pass for question answering with enhanced reasoning.
        
        Args:
            question: The question to answer
            context: Optional context information
            
        Returns:
            Prediction containing the generated answer
        """
        prediction = self.generate_answer(question=question, context=context)
        return dspy.Prediction(
            answer=getattr(prediction, 'answer', f'Mock answer to: {question}'),
            reasoning=getattr(prediction, 'reasoning', 'Mock reasoning (DSPY not installed)'),
            question=question
        )


class SentimentAnalysis(dspy.Module):
    """Enhanced sentiment analysis with confidence scoring and reasoning."""
    
    def __init__(self):
        super().__init__()
        if dspy and hasattr(dspy, 'ChainOfThought'):
            self.analyze_sentiment = dspy.ChainOfThought(SentimentClassification)
        else:
            self.analyze_sentiment = lambda **kwargs: dspy.Prediction(
                sentiment="neutral",
                confidence=0.5,
                reasoning="Mock sentiment analysis (DSPY not installed)"
            )
    
    def forward(self, text: str) -> dspy.Prediction:
        """Forward pass for sentiment analysis with detailed output.
        
        Args:
            text: The text to analyze for sentiment
            
        Returns:
            Prediction containing sentiment, confidence, and reasoning
        """
        prediction = self.analyze_sentiment(text=text)
        
        # Parse and validate confidence score
        confidence = getattr(prediction, 'confidence', '0.5')
        try:
            confidence_val = float(str(confidence).replace('%', '').strip())
            if confidence_val > 1:  # If it's a percentage
                confidence_val = confidence_val / 100
            confidence_val = max(0.0, min(1.0, confidence_val))  # Clamp to [0,1]
        except (ValueError, AttributeError):
            confidence_val = 0.5
        
        return dspy.Prediction(
            sentiment=getattr(prediction, 'sentiment', 'neutral'),
            confidence=confidence_val,
            reasoning=getattr(prediction, 'reasoning', 'Mock reasoning (DSPY not installed)'),
            text=text
        )