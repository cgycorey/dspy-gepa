#!/usr/bin/env python3
"""
LLM Setup and Configuration Demo
=================================

Demonstrates LLM provider detection, configuration, and status monitoring.
Perfect for understanding how to set up and manage different LLM providers.

Run with:
  uv run python examples/02_llm_setup.py
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dspy_gepa import GEPAAgent

def main():
    """Demonstrate LLM setup and configuration options."""
    print("🔧 LLM Setup and Configuration Demo")
    print("=" * 50)
    
    # Demo 1: Auto-detection from config/environment
    print("\n📋 Auto-detection from config.yaml")
    print("-" * 35)
    
    agent = GEPAAgent(objectives={"accuracy": 0.7, "efficiency": 0.3})
    status = agent.get_llm_status()
    
    print(f"✅ Status: {status['status']}")
    if status['status'] == 'available':
        print(f"✅ Provider: {status.get('provider', 'N/A')}")
        print(f"✅ Model: {status.get('model', 'N/A')}")
        print(f"✅ Config Source: {status.get('configuration_source', 'N/A')}")
    else:
        print("ℹ️  LLM not available - will use handcrafted mutations")
        print(f"ℹ️  Message: {status.get('message', 'No details available')}")
    
    # Demo 2: Manual provider configuration
    print("\n📋 Manual Provider Configuration")
    print("-" * 35)
    
    try:
        agent_manual = GEPAAgent(
            objectives={"accuracy": 1.0},
            llm_config={
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.5,
                "enabled": True
            },
            auto_detect_llm=False
        )
        manual_status = agent_manual.get_llm_status()
        print(f"✅ Provider: {manual_status.get('provider')}")
        print(f"✅ Model: {manual_status.get('model')}")
        print(f"✅ Temperature: {manual_status.get('temperature')}")
        print(f"✅ Available: {manual_status.get('available')}")
    except Exception as e:
        print(f"ℹ️  Manual config demo (needs valid API key): {e}")
    
    # Demo 3: Dynamic reconfiguration
    print("\n📋 Dynamic LLM Reconfiguration")
    print("-" * 35)
    
    # Switch to a different provider
    agent.configure_llm("anthropic", model="claude-3-sonnet", temperature=0.3)
    dynamic_status = agent.get_llm_status()
    print(f"✅ New Provider: {dynamic_status.get('provider')}")
    print(f"✅ New Model: {dynamic_status.get('model')}")
    print(f"✅ Mutation Type: {dynamic_status.get('mutation_type')}")
    
    # Demo 4: Fallback mode
    print("\n📋 Fallback Mode (No LLM)")
    print("-" * 30)
    
    fallback_agent = GEPAAgent(
        objectives={"accuracy": 0.8, "efficiency": 0.2},
        llm_config={"enabled": False}
    )
    
    fallback_status = fallback_agent.get_llm_status()
    print(f"✅ Will use LLM: {fallback_status['will_use_llm']}")
    print(f"✅ Mutation Type: {fallback_status['mutation_type']}")
    print(f"✅ Status: {fallback_status['message']}")
    
    # Demo 5: Environment variable setup
    print("\n📋 Environment Variable Support")
    print("-" * 35)
    
    print("✅ Set these environment variables for LLM access:")
    print("   export OPENAI_API_KEY='your-openai-key'")
    print("   export ANTHROPIC_API_KEY='your-anthropic-key'")
    print("   export DSpyGEPA_DEFAULT_PROVIDER='openai'")
    
    print("\n💡 Configuration Tips:")
    print("1. Use config.yaml for persistent settings")
    print("2. Environment variables override config files")
    print("3. Manual config takes precedence over auto-detection")
    print("4. Fallback mode works when LLM is unavailable")
    print("5. Use get_llm_status() to check current configuration")
    
    print("\n🎉 LLM setup demo completed!")

if __name__ == "__main__":
    main()