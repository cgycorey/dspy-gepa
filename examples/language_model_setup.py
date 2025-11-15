"""Language Model Setup and Configuration

This module contains the robust language model setup functions
that support multiple providers and fallback mechanisms.
"""

import os
import random
from typing import Optional, Any
from dataclasses import dataclass

try:
    import dspy
except ImportError:
    dspy = None


@dataclass
class LMConfig:
    """Configuration for language model setup."""
    provider: str  # "openai", "anthropic", "local", or "mock"
    model_name: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    api_base: Optional[str] = None
    timeout: int = 30


def setup_language_model() -> LMConfig:
    """Set up a real language model with robust fallback support.
    
    Priority order:
    1. OpenAI (if OPENAI_API_KEY is set)
    2. Anthropic (if ANTHROPIC_API_KEY is set)
    3. Local models (if LOCAL_MODEL_PATH is set)
    4. Enhanced mock model (for demonstration)
    
    Returns:
        LMConfig: Configuration object with model details
    """
    print("🔧 Setting up language model...")
    
    if dspy is None:
        print("⚠️ DSPY not available, using mock configuration")
        config = LMConfig(
            provider="mock",
            model_name="mock-no-dspy",
            temperature=0.7
        )
        print("✅ Mock LM configured (DSPY not installed)")
        return config
    
    # Check for OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
            max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
            
            lm = dspy.OpenAI(
                api_key=openai_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            dspy.settings.configure(lm=lm)
            
            config = LMConfig(
                provider="openai",
                model_name=model_name,
                api_key=openai_key[:10] + "...",  # Store partial key for logging
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"✅ OpenAI LM configured: {model_name}")
            return config
            
        except Exception as e:
            print(f"⚠️ OpenAI setup failed: {e}")
    
    # Check for Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
            temperature = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7"))
            max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024"))
            
            lm = dspy.Anthropic(
                api_key=anthropic_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            dspy.settings.configure(lm=lm)
            
            config = LMConfig(
                provider="anthropic",
                model_name=model_name,
                api_key=anthropic_key[:10] + "...",
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"✅ Anthropic LM configured: {model_name}")
            return config
            
        except Exception as e:
            print(f"⚠️ Anthropic setup failed: {e}")
    
    # Check for local models
    local_model_path = os.getenv("LOCAL_MODEL_PATH")
    if local_model_path:
        try:
            # For local models, we'll use a generic setup
            # In practice, this could be Ollama, llama.cpp, etc.
            model_name = os.getenv("LOCAL_MODEL_NAME", "local-model")
            api_base = os.getenv("LOCAL_API_BASE", "http://localhost:11434")
            
            # This is a placeholder - actual local model setup depends on the backend
            print(f"🔗 Attempting local model setup at {api_base}")
            
            config = LMConfig(
                provider="local",
                model_name=model_name,
                api_base=api_base,
                temperature=0.7
            )
            
            # Create a basic mock for local models (would be real in production)
            lm = create_enhanced_mock_lm("local")
            dspy.settings.configure(lm=lm)
            
            print(f"✅ Local LM configured: {model_name} (mocked for demo)")
            return config
            
        except Exception as e:
            print(f"⚠️ Local model setup failed: {e}")
    
    # Fallback to enhanced mock model
    print("🪙 No API keys found, using enhanced mock model for demonstration")
    print("💡 Set OPENAI_API_KEY or ANTHROPIC_API_KEY to use real models")
    
    mock_lm = create_enhanced_mock_lm()
    dspy.settings.configure(lm=mock_lm)
    
    config = LMConfig(
        provider="mock",
        model_name="enhanced-mock",
        temperature=0.7
    )
    
    print("✅ Enhanced mock LM configured")
    return config


def create_enhanced_mock_lm(provider: str = "mock") -> Any:
    """Create an enhanced mock language model with realistic responses.
    
    Args:
        provider: Mock provider type for customizing responses
        
    Returns:
        Mock language model instance
    """
    if dspy is None:
        # Return a simple mock when DSPY is not available
        class SimpleMockLM:
            def __init__(self, model: str = "mock-no-dspy"):
                self.model = model
            def __call__(self, prompt=None, **kwargs):
                return "Mock response (DSPY not installed)"
        return SimpleMockLM(f"mock-{provider}")
    
    class EnhancedMockLM:
        def __init__(self, model: str = "enhanced-mock-v1"):
            self.model = model
            self.provider = provider
            
            # Predefined responses for common question categories
            self.qa_responses = {
                "machine learning": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions with minimal human intervention.",
                "python": "Python is a high-level, interpreted programming language known for its simplicity, readability, and versatility. It was created by Guido van Rossum and first released in 1991. Python is widely used for web development, data science, artificial intelligence, and automation.",
                "genetic programming": "Genetic programming is an evolutionary computation technique that automatically solves problems without requiring users to specify the form or structure of the solution in advance. It evolves computer programs using principles inspired by biological evolution, including selection, mutation, and crossover.",
                "artificial intelligence": "Artificial Intelligence is a broad field of computer science focused on creating systems that can perform tasks that typically require human intelligence. This includes learning, reasoning, problem-solving, perception, and language understanding.",
                "default": "This is a complex topic that requires careful consideration. Based on current understanding and available information, we can approach this question from multiple perspectives."
            }
            
            # Sentiment analysis responses with confidence
            self.sentiment_responses = {
                "positive": ["positive", "very positive", "somewhat positive"],
                "negative": ["negative", "very negative", "somewhat negative"],
                "neutral": ["neutral", "somewhat neutral", "mixed"]
            }
            
        def __call__(self, prompt=None, **kwargs):
            """Mock LM call that returns realistic responses based on the input."""
            
            # Extract the core task from the prompt
            prompt_str = str(prompt) if prompt else ""
            prompt_lower = prompt_str.lower()
            
            # Handle question answering
            if "question" in prompt_lower or "answer" in prompt_lower:
                # Check for known topics
                for topic, response in self.qa_responses.items():
                    if topic.lower() in prompt_lower:
                        # Add some variety to the response
                        variations = [
                            response,
                            f"{response} This topic has gained significant attention in recent years.",
                            f"Let me explain: {response} It's important to understand the broader context here."
                        ]
                        return random.choice(variations)
                
                # Default QA response
                return self.qa_responses["default"]
            
            # Handle sentiment analysis
            elif "sentiment" in prompt_lower or "analyze" in prompt_lower:
                # Extract sentiment from sample texts
                if "love" in prompt_lower or "great" in prompt_lower or "excellent" in prompt_lower:
                    sentiment = "positive"
                    confidence = random.uniform(0.8, 0.95)
                elif "terrible" in prompt_lower or "hate" in prompt_lower or "awful" in prompt_lower:
                    sentiment = "negative"
                    confidence = random.uniform(0.8, 0.95)
                elif "okay" in prompt_lower or "average" in prompt_lower or "fine" in prompt_lower:
                    sentiment = "neutral"
                    confidence = random.uniform(0.6, 0.8)
                else:
                    # Random sentiment for unknown text
                    sentiment = random.choice(list(self.sentiment_responses.keys()))
                    confidence = random.uniform(0.5, 0.8)
                
                # Generate reasoning
                reasoning_templates = {
                    "positive": [
                        f"The text contains positive language and enthusiastic tone, indicating {sentiment} sentiment with {confidence:.1%} confidence.",
                        f"Strong positive emotional language detected. Confidence: {confidence:.1%}."
                    ],
                    "negative": [
                        f"The text expresses dissatisfaction or negative emotions, suggesting {sentiment} sentiment with {confidence:.1%} confidence.",
                        f"Clear negative indicators found in the language. Confidence: {confidence:.1%}."
                    ],
                    "neutral": [
                        f"The text maintains a balanced or objective tone, indicating {sentiment} sentiment with {confidence:.1%} confidence.",
                        f"Language appears factual or emotionally neutral. Confidence: {confidence:.1%}."
                    ]
                }
                
                reasoning = random.choice(reasoning_templates[sentiment])
                
                # Format the response to match DSPY expectations
                return f"Sentiment: {sentiment}\nConfidence: {confidence:.1%}\nReasoning: {reasoning}"
            
            # Default response
            else:
                default_responses = [
                    "Based on the provided input and context, I'll provide a thoughtful response that considers multiple perspectives.",
                    "This requires careful analysis. Let me consider the key aspects and provide a comprehensive answer.",
                    "I need to analyze this systematically. The response should address the core components while maintaining clarity and accuracy."
                ]
                return random.choice(default_responses)
        
        def get_token_count(self, text: str) -> int:
            """Simple token count for mock LM."""
            return len(text.split())
    
    return EnhancedMockLM(f"enhanced-{provider}-mock-v1")