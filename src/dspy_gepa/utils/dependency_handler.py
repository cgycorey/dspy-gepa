"""Graceful dependency handling utilities.

This module provides utilities for handling optional dependencies gracefully,
with informative error messages and fallback behavior.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Type, Union


class DependencyError(Exception):
    """Raised when an optional dependency is not available."""
    
    def __init__(self, dependency: str, install_command: str, purpose: str = ""):
        self.dependency = dependency
        self.install_command = install_command
        self.purpose = purpose
        
        message = f"Optional dependency '{dependency}' is not installed."
        if purpose:
            message += f" Required for: {purpose}."
        message += f" Install with: {install_command}"
        
        super().__init__(message)


class DependencyManager:
    """Manages optional dependencies with graceful fallbacks."""
    
    def __init__(self):
        self._checked_modules: Dict[str, bool] = {}
        self._module_cache: Dict[str, Any] = {}
    
    def check_availability(self, module_name: str) -> bool:
        """Check if a module is available.
        
        Args:
            module_name: Name of the module to check
            
        Returns:
            True if module is available, False otherwise
        """
        if module_name not in self._checked_modules:
            try:
                __import__(module_name)
                self._checked_modules[module_name] = True
            except ImportError:
                self._checked_modules[module_name] = False
        
        return self._checked_modules[module_name]
    
    def import_optional(
        self, 
        module_name: str, 
        install_hint: str,
        purpose: str = "",
        fallback: Any = None
    ) -> Any:
        """Import an optional module with graceful error handling.
        
        Args:
            module_name: Name of the module to import
            install_hint: Installation hint for the user
            purpose: What the module is used for
            fallback: Value to return if module is not available
            
        Returns:
            The imported module or fallback value
            
        Raises:
            DependencyError: If module is not available and no fallback provided
        """
        if module_name in self._module_cache:
            return self._module_cache[module_name]
        
        if not self.check_availability(module_name):
            if fallback is not None:
                self._module_cache[module_name] = fallback
                return fallback
            
            raise DependencyError(
                dependency=module_name,
                install_command=install_hint,
                purpose=purpose
            )
        
        try:
            module = __import__(module_name)
            self._module_cache[module_name] = module
            return module
        except Exception as e:
            if fallback is not None:
                self._module_cache[module_name] = fallback
                return fallback
            
            raise DependencyError(
                dependency=module_name,
                install_command=install_hint,
                purpose=f"Import failed: {e}"
            ) from e
    
    def get_dependency_status(self) -> Dict[str, bool]:
        """Get status of all checked dependencies.
        
        Returns:
            Dictionary mapping module names to availability status
        """
        return self._checked_modules.copy()
    
    def print_dependency_report(self) -> None:
        """Print a human-readable dependency status report."""
        print("\n📦 Dependency Status Report")
        print("=" * 30)
        
        # Check common optional dependencies
        optional_deps = {
            "dspy": "dspy>=2.4.0",
            "openai": "pip install 'dspy-gepa[openai]'",
            "anthropic": "pip install 'dspy-gepa[anthropic]'",
            "transformers": "pip install transformers",
            "torch": "pip install torch",
            "numpy": "pip install numpy",
            "pandas": "pip install pandas",
        }
        
        for dep, install_cmd in optional_deps.items():
            available = self.check_availability(dep)
            status = "✅ Available" if available else "❌ Not available"
            print(f"{dep:<12} {status:<15} {install_cmd}")
        
        print()


# Global dependency manager instance
dependency_manager = DependencyManager()


# Convenience functions
def require_dspy():
    """Import DSPy with graceful error handling."""
    return dependency_manager.import_optional(
        "dspy",
        "pip install 'dspy-gepa[dspy-full]' or pip install dspy>=2.4.0",
        "DSPy integration and advanced prompt optimization"
    )


def require_openai():
    """Import OpenAI with graceful error handling."""
    return dependency_manager.import_optional(
        "openai",
        "pip install 'dspy-gepa[openai]' or pip install openai",
        "OpenAI LLM provider"
    )


def require_anthropic():
    """Import Anthropic with graceful error handling."""
    return dependency_manager.import_optional(
        "anthropic",
        "pip install 'dspy-gepa[anthropic]' or pip install anthropic",
        "Anthropic LLM provider"
    )


def require_transformers():
    """Import Transformers with graceful error handling."""
    return dependency_manager.import_optional(
        "transformers",
        "pip install transformers",
        "Hugging Face Transformers models"
    )


def is_dspy_available() -> bool:
    """Check if DSPy is available."""
    return dependency_manager.check_availability("dspy")


def is_openai_available() -> bool:
    """Check if OpenAI is available."""
    return dependency_manager.check_availability("openai")


def is_anthropic_available() -> bool:
    """Check if Anthropic is available."""
    return dependency_manager.check_availability("anthropic")


def create_mock_provider(provider_name: str):
    """Create a mock provider for testing when real providers are not available.
    
    Args:
        provider_name: Name of the provider to mock
        
    Returns:
        A mock provider instance
    """
    class MockProvider:
        def __init__(self, name: str):
            self.name = name
        
        def get_provider_name(self) -> str:
            return self.name
        
        async def generate_reflection(self, prompt: str, config) -> str:
            return f"Mock reflection from {self.name} provider for prompt: {prompt[:50]}..."
    
    return MockProvider(provider_name)


if __name__ == "__main__":
    # Demo dependency handling
    dependency_manager.print_dependency_report()
    
    # Example usage
    try:
        dspy = require_dspy()
        print("✅ DSPy is available")
    except DependencyError as e:
        print(f"⚠️ {e}")
    
    try:
        openai = require_openai()
        print("✅ OpenAI is available")
    except DependencyError as e:
        print(f"⚠️ {e}")
        # Create mock provider as fallback
        mock_provider = create_mock_provider("openai-mock")
        print(f"🔧 Created mock provider: {mock_provider.get_provider_name()}")
