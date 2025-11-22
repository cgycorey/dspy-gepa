"""Patch to configure LiteLLM globally before DSPY uses it."""

import os
import sys

def patch_litellm():
    """Configure LiteLLM to drop unsupported parameters globally."""
    try:
        import litellm
        litellm.drop_params = True
        print("✅ LiteLLM patched: drop_params=True")
        
        # Also set environment variable for LiteLLM
        os.environ['LITELLM_DROP_PARAMS'] = 'true'
        
        return True
    except ImportError:
        print("⚠️  LiteLLM not available, skipping patch")
        return False
    except Exception as e:
        print(f"⚠️  LiteLLM patch failed: {e}")
        return False

def patch_dspy_settings():
    """Patch DSPY to use compatible models by default."""
    try:
        import dspy
        
        # Check if we can access settings
        if hasattr(dspy, 'settings'):
            # Try to set a compatible model if none is configured
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                # Check for OpenAI API key
                if os.getenv('OPENAI_API_KEY'):
                    try:
                        # Use gpt-4o-mini which supports structured outputs
                        lm = dspy.LM(model='openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
                        dspy.settings.configure(lm=lm)
                        print("✅ DSPY patched: Using gpt-4o-mini by default")
                        return True
                    except Exception as e:
                        print(f"⚠️  Failed to set default DSPY LM: {e}")
                        return False
        return True
        
    except ImportError:
        print("⚠️  DSPY not available, skipping DSPY patch")
        return False
    except Exception as e:
        print(f"⚠️  DSPY patch failed: {e}")
        return False

def apply_all_patches():
    """Apply all patches to fix GEPA compatibility issues."""
    print("🔧 Applying GEPA compatibility patches...")
    
    litellm_success = patch_litellm()
    dspy_success = patch_dspy_settings()
    
    if litellm_success and dspy_success:
        print("✅ All patches applied successfully")
        return True
    else:
        print("⚠️  Some patches failed, but continuing...")
        return False

# Auto-apply patches when imported
if __name__ != "__main__":
    # Only auto-apply when imported as a module
    apply_all_patches()