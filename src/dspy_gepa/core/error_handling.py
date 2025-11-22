"""Comprehensive error handling and logging utilities.

This module provides centralized error handling, logging, and recovery
mechanisms for the multi-objective GEPA framework.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import traceback
import time
from typing import Any, Dict, List, Optional, Callable, Union, Type
from dataclasses import dataclass
from enum import Enum
import functools

from ..utils.logging import get_logger


_logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better organization."""
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    EXECUTION = "execution"
    RESOURCE = "resource"
    COMMUNICATION = "communication"
    ALGORITHM = "algorithm"
    DATA = "data"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception: Optional[Exception]
    traceback_str: Optional[str]
    context_data: Dict[str, Any]
    recovery_attempts: int = 0
    resolved: bool = False
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class MultiObjectiveGEPAError(Exception):
    """Base exception for multi-objective GEPA framework."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.__cause__ = cause


class ValidationError(MultiObjectiveGEPAError):
    """Error raised for validation failures."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None, **kwargs):
        context = kwargs.get('context', {})
        if field:
            context['field'] = field
        if value is not None:
            context['value'] = str(value)
        
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs
        )


class ConfigurationError(MultiObjectiveGEPAError):
    """Error raised for configuration issues."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


class ExecutionError(MultiObjectiveGEPAError):
    """Error raised during execution of optimization tasks."""
    
    def __init__(self, message: str, task_id: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if task_id:
            context['task_id'] = task_id
        
        super().__init__(
            message,
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs
        )


class ResourceError(MultiObjectiveGEPAError):
    """Error raised for resource-related issues."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if resource_type:
            context['resource_type'] = resource_type
        
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


class OptimizationError(MultiObjectiveGEPAError):
    """Error raised during optimization process."""
    
    def __init__(self, message: str, generation: Optional[int] = None, **kwargs):
        context = kwargs.get('context', {})
        if generation is not None:
            context['generation'] = generation
        
        super().__init__(
            message,
            category=ErrorCategory.ALGORITHM,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs
        )


class LLMError(MultiObjectiveGEPAError):
    """Error raised for LLM-related issues."""
    
    def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if provider:
            context['provider'] = provider
        
        # Add helpful suggestions
        suggestions = [
            "Check your API key configuration",
            "Verify your internet connection",
            "Check API rate limits and quota",
            "Try a different model or provider"
        ]
        
        context['suggestions'] = suggestions
        
        super().__init__(
            message,
            category=ErrorCategory.COMMUNICATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


class ErrorHandler:
    """Centralized error handler for the framework."""
    
    def __init__(self, max_error_history: int = 1000):
        """Initialize error handler.
        
        Args:
            max_error_history: Maximum number of errors to keep in history
        """
        self.max_error_history = max_error_history
        self.error_history: List[ErrorContext] = []
        self.error_callbacks: Dict[ErrorCategory, List[Callable]] = {
            category: [] for category in ErrorCategory
        }
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.error_counts: Dict[str, int] = {}
        
        _logger.info("Error handler initialized")
    
    def handle_error(
        self,
        error: Exception,
        context_data: Optional[Dict[str, Any]] = None,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None
    ) -> ErrorContext:
        """Handle an error with logging and optional recovery.
        
        Args:
            error: The exception that occurred
            context_data: Additional context information
            category: Error category (inferred if not provided)
            severity: Error severity (inferred if not provided)
            
        Returns:
            ErrorContext with details about the error
        """
        # Determine category and severity if not provided
        if category is None:
            category = self._infer_error_category(error)
        if severity is None:
            severity = self._infer_error_severity(error)
        
        # Create error context
        error_id = f"error_{int(time.time() * 1000)}_{id(error)}"
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=str(error),
            exception=error,
            traceback_str=traceback.format_exc(),
            context_data=context_data or {}
        )
        
        # Log the error
        self._log_error(error_context)
        
        # Update statistics
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to history
        self.error_history.append(error_context)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
        
        # Trigger callbacks
        self._trigger_error_callbacks(error_context)
        
        # Attempt recovery
        if not error_context.resolved:
            self._attempt_recovery(error_context)
        
        return error_context
    
    def _infer_error_category(self, error: Exception) -> ErrorCategory:
        """Infer error category from exception type."""
        if isinstance(error, (ValueError, TypeError)):
            return ErrorCategory.VALIDATION
        elif isinstance(error, (ConfigurationError, AttributeError, KeyError)):
            return ErrorCategory.CONFIGURATION
        elif isinstance(error, (ExecutionError, RuntimeError)):
            return ErrorCategory.EXECUTION
        elif isinstance(error, (MemoryError, TimeoutError)):
            return ErrorCategory.RESOURCE
        elif isinstance(error, (ConnectionError, OSError)):
            return ErrorCategory.COMMUNICATION
        else:
            return ErrorCategory.SYSTEM
    
    def _infer_error_severity(self, error: Exception) -> ErrorSeverity:
        """Infer error severity from exception type and content."""
        error_msg = str(error).lower()
        
        # Critical indicators
        critical_indicators = ['critical', 'fatal', 'system', 'corruption']
        if any(indicator in error_msg for indicator in critical_indicators):
            return ErrorSeverity.CRITICAL
        
        # High severity indicators
        if isinstance(error, (MemoryError, TimeoutError)):
            return ErrorSeverity.HIGH
        
        # Medium severity
        if isinstance(error, (ValueError, RuntimeError)):
            return ErrorSeverity.MEDIUM
        
        # Default to low
        return ErrorSeverity.LOW
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log the error with appropriate level."""
        log_message = f"[{error_context.category.value.upper()}] {error_context.message}"
        
        if error_context.context_data:
            log_message += f" | Context: {error_context.context_data}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            _logger.critical(log_message, exc_info=error_context.exception)
        elif error_context.severity == ErrorSeverity.HIGH:
            _logger.error(log_message, exc_info=error_context.exception)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            _logger.warning(log_message, exc_info=error_context.exception)
        else:
            _logger.info(log_message, exc_info=error_context.exception)
    
    def _trigger_error_callbacks(self, error_context: ErrorContext) -> None:
        """Trigger registered error callbacks."""
        for callback in self.error_callbacks[error_context.category]:
            try:
                callback(error_context)
            except Exception as e:
                _logger.warning(f"Error callback failed: {e}")
    
    def _attempt_recovery(self, error_context: ErrorContext) -> None:
        """Attempt to recover from the error."""
        if error_context.recovery_attempts >= 3:
            _logger.warning(f"Max recovery attempts reached for {error_context.error_id}")
            return
        
        error_context.recovery_attempts += 1
        
        # Try type-specific recovery strategies
        error_type = type(error_context.exception)
        if error_type in self.recovery_strategies:
            try:
                recovery_success = self.recovery_strategies[error_type](error_context)
                if recovery_success:
                    error_context.resolved = True
                    _logger.info(f"Error {error_context.error_id} recovered successfully")
                else:
                    _logger.warning(f"Recovery failed for {error_context.error_id}")
            except Exception as e:
                _logger.warning(f"Recovery strategy failed for {error_context.error_id}: {e}")
    
    def register_error_callback(self, category: ErrorCategory, callback: Callable[[ErrorContext], None]) -> None:
        """Register a callback for specific error categories."""
        self.error_callbacks[category].append(callback)
        _logger.debug(f"Registered error callback for {category.value}")
    
    def register_recovery_strategy(self, error_type: Type[Exception], strategy: Callable[[ErrorContext], bool]) -> None:
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = strategy
        _logger.debug(f"Registered recovery strategy for {error_type.__name__}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        total_errors = sum(self.error_counts.values())
        
        # Recent errors (last hour)
        recent_time = time.time() - 3600
        recent_errors = sum(
            1 for error in self.error_history
            if error.timestamp > recent_time
        )
        
        # Errors by severity
        severity_counts = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Errors by category
        category_counts = {}
        for error in self.error_history:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_errors": total_errors,
            "recent_errors_last_hour": recent_errors,
            "error_types": dict(self.error_counts),
            "by_severity": severity_counts,
            "by_category": category_counts,
            "error_history_size": len(self.error_history)
        }
    
    def clear_error_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()
        self.error_counts.clear()


# User-friendly error handling functions
def handle_common_errors(func: Callable) -> Callable:
    """Decorator to handle common errors with helpful messages."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConfigurationError as e:
            print(f"❌ Configuration Error: {e}")
            print("💡 Suggestions:")
            print("   • Check your API key is set correctly")
            print("   • Verify your LLM provider configuration")
            print("   • Run print_llm_status() for detailed help")
            raise
        except ValidationError as e:
            print(f"❌ Validation Error: {e}")
            print("💡 Suggestions:")
            print("   • Check your evaluation data format")
            print("   • Ensure all required fields are present")
            print("   • Verify data types match expected format")
            raise
        except LLMError as e:
            print(f"❌ LLM Error: {e}")
            print("💡 Suggestions:")
            for suggestion in e.context.get('suggestions', []):
                print(f"   • {suggestion}")
            raise
        except OptimizationError as e:
            print(f"❌ Optimization Error: {e}")
            print("💡 Suggestions:")
            print("   • Try the 'quick' preset for testing")
            print("   • Reduce population size or generations")
            print("   • Check your evaluation data quality")
            raise
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            print("💡 Suggestions:")
            print("   • Check the traceback above for details")
            print("   • Try running with debug mode enabled")
            print("   • Report the issue if it persists")
            raise
    return wrapper


def set_debug_mode(enabled: bool = True):
    """Enable or disable debug mode for detailed error messages."""
    if enabled:
        import logging
        logging.getLogger('dspy_gepa').setLevel(logging.DEBUG)
        print("🐛 Debug mode enabled - detailed error messages will be shown")
    else:
        import logging
        logging.getLogger('dspy_gepa').setLevel(logging.INFO)
        print("✅ Debug mode disabled")


def format_error_with_help(error: Exception) -> str:
    """Format error with helpful suggestions."""
    if isinstance(error, ConfigurationError):
        return f"\n❌ Configuration Error: {error}\n\n💡 To fix this:\n   1. Check your API key is set\n   2. Verify LLM provider configuration\n   3. Run print_llm_status() for help"
    
    elif isinstance(error, ValidationError):
        return f"\n❌ Validation Error: {error}\n\n💡 To fix this:\n   1. Check your evaluation data format\n   2. Ensure all required fields are present\n   3. Verify data types are correct"
    
    elif isinstance(error, LLMError):
        return f"\n❌ LLM Error: {error}\n\n💡 To fix this:\n   1. Check your API key and quota\n   2. Verify internet connection\n   3. Try a different model or provider"
    
    else:
        return f"\n❌ Error: {error}\n\n💡 For help, run show_help('troubleshooting')"
        _logger.info("Error history cleared")


def handle_errors(category: ErrorCategory = ErrorCategory.SYSTEM, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for automatic error handling.
    
    Args:
        category: Error category for handled exceptions
        severity: Error severity for handled exceptions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get error handler (try from module or create new)
                error_handler = getattr(wrapper, '_error_handler', ErrorHandler())
                
                # Handle the error
                error_context = error_handler.handle_error(
                    e,
                    context_data={
                        'function': func.__name__,
                        'module': func.__module__,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    },
                    category=category,
                    severity=severity
                )
                
                # Re-raise if not resolved and severity is high or critical
                if not error_context.resolved and error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                    raise
                
                # Return None for resolved errors or low severity
                return None
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    default_value: Any = None,
    category: ErrorCategory = ErrorCategory.EXECUTION,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[Dict[str, Any]] = None
) -> Any:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        default_value: Value to return on error
        category: Error category
        severity: Error severity
        context: Additional context
        
    Returns:
        Function result or default value on error
    """
    try:
        return func()
    except Exception as e:
        # Get or create error handler
        error_handler = ErrorHandler()
        
        # Handle the error
        error_context = error_handler.handle_error(
            e,
            context_data=context or {},
            category=category,
            severity=severity
        )
        
        # Return default value if not resolved
        if not error_context.resolved:
            return default_value
        
        # If error was resolved, try again (once)
        if error_context.recovery_attempts == 1:
            try:
                return func()
            except Exception:
                return default_value
        
        return default_value


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_global_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_global_error(
    error: Exception,
    context_data: Optional[Dict[str, Any]] = None,
    category: Optional[ErrorCategory] = None,
    severity: Optional[ErrorSeverity] = None
) -> ErrorContext:
    """Handle an error using the global error handler."""
    return get_global_error_handler().handle_error(error, context_data, category, severity)


# Common recovery strategies
def memory_error_recovery(error_context: ErrorContext) -> bool:
    """Recovery strategy for memory errors."""
    _logger.warning("Attempting memory error recovery: triggering garbage collection")
    import gc
    gc.collect()
    return True


def timeout_error_recovery(error_context: ErrorContext) -> bool:
    """Recovery strategy for timeout errors."""
    _logger.warning("Timeout error recovery: increasing timeout tolerance")
    # In practice, this might adjust timeout settings
    return False  # Not actually resolved, but attempted


def validation_error_recovery(error_context: ErrorContext) -> bool:
    """Recovery strategy for validation errors."""
    # Try to provide default values for missing/invalid data
    context = error_context.context_data
    if 'field' in context and 'value' in context:
        _logger.info(f"Validation error recovery: using default value for {context['field']}")
        return True
    return False


# Register common recovery strategies
def _register_default_recovery_strategies():
    """Register default recovery strategies with the global error handler."""
    handler = get_global_error_handler()
    
    handler.register_recovery_strategy(MemoryError, memory_error_recovery)
    handler.register_recovery_strategy(TimeoutError, timeout_error_recovery)
    handler.register_recovery_strategy(ValueError, validation_error_recovery)
    handler.register_recovery_strategy(TypeError, validation_error_recovery)


# Initialize default recovery strategies when module is imported
_register_default_recovery_strategies()