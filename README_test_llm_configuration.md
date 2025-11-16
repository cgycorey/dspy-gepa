# LLM Configuration Detection Test Script

This document describes the comprehensive test script for LLM configuration detection in the dspy-gepa system.

## Overview

The `test_llm_configuration.py` script provides comprehensive testing of all aspects of LLM configuration detection, ensuring the system works correctly in various scenarios.

## Test Scenarios

### 1. Auto-detection from config.yaml
- Tests automatic loading of LLM configuration from the default config.yaml file
- Verifies default provider detection
- Validates provider configuration loading

### 2. Environment Variable Loading
- Tests loading of API keys from environment variables
- Validates OPENAI_API_KEY and ANTHROPIC_API_KEY integration
- Ensures proper configuration when environment variables are set

### 3. Manual LLM Configuration
- Tests loading custom configuration files
- Validates manual configuration overrides
- Ensures custom settings are properly applied

### 4. Configuration with No API Keys
- Tests system behavior when no API keys are available
- Validates fallback to mock/local configurations
- Ensures graceful degradation

### 5. Invalid Configuration Scenarios
- Tests handling of invalid provider names
- Validates malformed YAML file handling
- Tests missing file scenarios
- Ensures robust error handling

### 6. Comprehensive Status Reporting
- Tests complete system status gathering
- Validates detailed provider status reporting
- Ensures comprehensive configuration overview

## Usage

### Prerequisites
- Ensure you're in the dspy-gepa root directory
- The `config.yaml` file should be present
- Python dependencies should be installed

### Running the Tests

```bash
# Make the script executable (if not already)
chmod +x test_llm_configuration.py

# Run all tests
python test_llm_configuration.py

# Or execute directly
./test_llm_configuration.py
```

### Output Format

The script provides:
- **Individual test results** with pass/fail status
- **Detailed information** for each test scenario
- **Comprehensive summary** with success rates
- **System component status** overview
- **Detailed configuration status** for all providers

## Example Output

```
🧪 LLM Configuration Detection Test Suite
============================================================
Testing dspy-gepa LLM configuration system...

🔍 SCENARIO 1: Testing auto-detection from config.yaml
--------------------------------------------------
✅ PASS: Config File Detection
     Default provider: openai
     openai_model: gpt-4
     anthropic_model: claude-3-opus-20240229

📊 DETAILED CONFIGURATION STATUS:
========================================
Default Provider: openai
Total Providers: 2

Provider: openai
  Configured: ✅ Yes
  Has API Key: ✅ Yes
  Model: gpt-4
  Temperature: 0.7
  Max Tokens: 2048

============================================================
🏁 TEST SUMMARY
============================================================
Total Tests: 9
✅ Passed: 9
❌ Failed: 0
Success Rate: 100.0%

🎉 ALL TESTS PASSED! LLM configuration system is working correctly.
```

## Test Details

### Mock API Keys

The script uses mock API keys for testing:
- `OPENAI_API_KEY`: `sk-test1234567890abcdef1234567890abcdef12345678`
- `ANTHROPIC_API_KEY`: `sk-ant-test03abcdefghijklmnopqrstuvwxyz123456`

These are fake keys used only for testing and will not work with real APIs.

### Environment Setup

The script automatically:
1. Backs up original environment variables
2. Sets up test environment with mock values
3. Runs all test scenarios
4. Restores original environment
5. Cleans up temporary files

### Error Handling

The script includes comprehensive error handling:
- Graceful handling of missing dependencies
- Fallback mechanisms for unavailable modules
- Detailed error reporting
- Clean cleanup on failure

## Integration with CI/CD

This test script can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Test LLM Configuration
  run: |
    python test_llm_configuration.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the dspy-gepa root directory
2. **Missing Config**: Verify `config.yaml` exists in the current directory
3. **Permission Issues**: Ensure the script is executable (`chmod +x`)

### Debug Mode

For detailed debugging, modify the script to enable verbose logging:

```python
# Add at the top of the script
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When adding new test scenarios:
1. Follow the existing naming convention (`test_scenario_X_name`)
2. Use the `log_test_result` method for consistent reporting
3. Include comprehensive test details
4. Ensure proper cleanup
5. Update this documentation

## Security Notes

- The script uses mock API keys - no real API calls are made
- Temporary files are automatically cleaned up
- Original environment variables are safely restored
- No sensitive information is logged

## License

This test script is part of the dspy-gepa project and follows the same license terms.