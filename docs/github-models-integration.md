# GitHub Models Integration Guide

This guide explains how to integrate and use GitHub Models with the MTC Extraction Benchmark project.

## What is GitHub Models?

GitHub Models provides access to state-of-the-art AI models through a simple API, directly integrated with GitHub. It's particularly useful for:
- Experimenting with different LLMs for document extraction
- Rapid prototyping without infrastructure setup
- Testing various models before committing to deployment

## Available Models

GitHub Models provides access to several model families relevant to this project:

### Language Models (LLMs)
- **GPT-4o**: Advanced reasoning, multimodal capabilities
- **GPT-4o mini**: Faster, cost-effective variant
- **GPT-3.5 Turbo**: Good balance of performance and speed
- **Phi-3**: Efficient small language models
- **Llama 3.1**: Open-source LLM with strong performance
- **Mistral**: Efficient and powerful open-source models
- **Cohere Command**: Strong for structured extraction tasks

### Embedding Models
- **text-embedding-3-small**: Efficient semantic embeddings
- **text-embedding-3-large**: Higher quality embeddings
- **Cohere Embed**: Alternative embedding solution

## Getting Started

### 1. Prerequisites

- GitHub account (personal or organization)
- Access to GitHub Models (currently in preview)
- Python 3.8 or higher

### 2. Authentication

GitHub Models uses your GitHub token for authentication:

```bash
# Create a GitHub Personal Access Token (PAT)
# Go to: https://github.com/settings/tokens
# Scopes needed: 'read:user' and 'user:email'

# Set the token as an environment variable
export GITHUB_TOKEN="your_github_pat_here"
```

**In Codespaces:**
The `GITHUB_TOKEN` is automatically available as an environment variable.

### 3. Installation

The required packages are already included in the devcontainer:

```bash
pip install openai requests python-dotenv
```

## Using GitHub Models in This Project

### Option 1: OpenAI-Compatible API

GitHub Models provides an OpenAI-compatible endpoint:

```python
import os
from openai import OpenAI

# Configure the client to use GitHub Models
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)

# Use for text extraction
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are an expert at extracting structured information from Mill Test Certificates."
        },
        {
            "role": "user",
            "content": f"Extract the following fields from this certificate text: {ocr_text}\n\nFields: heat_number, material_grade, chemical_composition, mechanical_properties"
        }
    ],
    temperature=0.1,  # Low temperature for consistent extraction
    response_format={"type": "json_object"}  # Get structured JSON output
)

extracted_data = response.choices[0].message.content
```

### Option 2: Direct REST API

```python
import os
import requests
import json

def extract_with_github_models(text: str, model: str = "gpt-4o-mini") -> dict:
    """
    Extract structured information using GitHub Models.
    
    Args:
        text: OCR text from the certificate
        model: Model to use (gpt-4o, gpt-4o-mini, etc.)
    
    Returns:
        Extracted data as dictionary
    """
    endpoint = "https://models.inference.ai.azure.com/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": """Extract information from Mill Test Certificates into JSON format.
                
                Output schema:
                {
                    "heat_number": "string",
                    "material_grade": "string",
                    "chemical_composition": {"element": "percentage"},
                    "mechanical_properties": {
                        "yield_strength": "value with unit",
                        "tensile_strength": "value with unit",
                        "elongation": "value with unit"
                    },
                    "dimensions": {"dimension": "value"},
                    "manufacturer": "string",
                    "certificate_date": "YYYY-MM-DD",
                    "standards": ["standard1", "standard2"]
                }"""
            },
            {
                "role": "user",
                "content": f"Certificate text:\n\n{text}"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2000
    }
    
    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    return json.loads(result["choices"][0]["message"]["content"])
```

## Integration Patterns for MTC Extraction

### Pattern 1: LLM-Based Extraction Pipeline

Use GitHub Models as the primary extraction engine:

```python
from typing import List, Dict
import json

class GitHubModelsExtractor:
    """Extract structured data from MTCs using GitHub Models."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ["GITHUB_TOKEN"],
        )
        self.model = model
    
    def extract(self, ocr_text: str) -> Dict:
        """Extract structured data from OCR text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._get_extraction_prompt()
                },
                {
                    "role": "user",
                    "content": ocr_text
                }
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _get_extraction_prompt(self) -> str:
        """Get the system prompt for extraction."""
        return """You are an expert at extracting structured information from Mill Test Certificates.
        
        Extract all available information and return it in valid JSON format.
        Use null for missing values. Be precise with numbers and units.
        
        Expected schema:
        {
            "heat_number": "string or null",
            "material_grade": "string or null",
            "chemical_composition": {
                "C": "percentage",
                "Si": "percentage",
                "Mn": "percentage",
                "P": "percentage",
                "S": "percentage",
                // ... other elements
            },
            "mechanical_properties": {
                "yield_strength": "value with unit",
                "tensile_strength": "value with unit",
                "elongation": "percentage",
                "hardness": "value with unit"
            },
            "dimensions": {},
            "manufacturer": "string or null",
            "certificate_date": "YYYY-MM-DD or null",
            "standards": []
        }"""
    
    def batch_extract(self, documents: List[str]) -> List[Dict]:
        """Extract from multiple documents."""
        return [self.extract(doc) for doc in documents]
```

### Pattern 2: Hybrid Approach

Combine traditional OCR with LLM-based refinement:

```python
class HybridExtractor:
    """Combine OCR with GitHub Models for best results."""
    
    def __init__(self, ocr_engine, llm_model="gpt-4o-mini"):
        self.ocr_engine = ocr_engine
        self.llm_extractor = GitHubModelsExtractor(llm_model)
    
    def extract(self, document_path: str) -> Dict:
        """Two-stage extraction: OCR then LLM refinement."""
        # Stage 1: OCR
        ocr_text = self.ocr_engine.extract_text(document_path)
        
        # Stage 2: LLM-based structured extraction
        structured_data = self.llm_extractor.extract(ocr_text)
        
        return {
            "raw_ocr": ocr_text,
            "extracted_data": structured_data
        }
```

### Pattern 3: Few-Shot Learning

Improve extraction with examples:

```python
def extract_with_examples(text: str, examples: List[Dict]) -> Dict:
    """Use few-shot learning for better extraction."""
    
    # Build prompt with examples
    example_messages = []
    for ex in examples:
        example_messages.extend([
            {"role": "user", "content": ex["input_text"]},
            {"role": "assistant", "content": json.dumps(ex["expected_output"])}
        ])
    
    messages = [
        {"role": "system", "content": "Extract structured data from certificates."}
    ] + example_messages + [
        {"role": "user", "content": text}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.1
    )
    
    return json.loads(response.choices[0].message.content)
```

## Evaluation and Benchmarking

Compare GitHub Models against other solutions:

```python
import time
from typing import Dict, Any

class ModelBenchmark:
    """Benchmark different extraction approaches."""
    
    def __init__(self):
        self.results = []
    
    def benchmark_model(self, model_name: str, extractor, test_set: List[Dict]) -> Dict[str, Any]:
        """Benchmark a specific model."""
        start_time = time.time()
        predictions = []
        
        for test_case in test_set:
            try:
                pred = extractor.extract(test_case["text"])
                predictions.append(pred)
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                predictions.append({})
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        accuracy = self._calculate_accuracy(predictions, test_set)
        
        return {
            "model": model_name,
            "accuracy": accuracy,
            "avg_time_per_doc": elapsed / len(test_set),
            "total_time": elapsed
        }
    
    def _calculate_accuracy(self, predictions, ground_truth):
        """Calculate field-level accuracy."""
        # Implement your accuracy metrics here
        pass
```

## Cost Optimization

### 1. Model Selection

Choose the right model for your use case:

| Model | Use Case | Speed | Cost | Accuracy |
|-------|----------|-------|------|----------|
| gpt-4o | Complex documents, high accuracy needed | Slow | High | Highest |
| gpt-4o-mini | Most MTC extraction tasks | Fast | Low | High |
| gpt-3.5-turbo | Simple, consistent formats | Fastest | Lowest | Good |

### 2. Prompt Optimization

Efficient prompts reduce tokens:

```python
# Good: Concise, specific
prompt = "Extract heat_number, grade, and composition from: {text}"

# Bad: Verbose, unnecessary
prompt = """Please carefully read the following certificate text and 
extract the heat number, material grade, and chemical composition. 
Make sure to be accurate and thorough in your extraction..."""
```

### 3. Caching and Batching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def extract_cached(text_hash: str) -> Dict:
    """Cache extraction results to avoid redundant API calls.
    
    Args:
        text_hash: MD5 hash of the text to extract from
        
    Note: The actual text must be retrieved separately.
    This is a simplified example - in production, you'd want
    a more robust caching mechanism.
    """
    # In a real implementation, you'd retrieve the text from a store
    # using the hash, or pass both hash and text but cache by hash
    return extract_with_github_models(text)

# Use it
text_hash = hashlib.md5(ocr_text.encode()).hexdigest()
# Store text separately if needed
result = extract_cached(text_hash)

# Better approach: Use a proper cache with both key and value
from functools import wraps

def cache_by_text_hash(func):
    """Decorator to cache function results by text hash."""
    cache = {}
    
    @wraps(func)
    def wrapper(text: str) -> Dict:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash not in cache:
            cache[text_hash] = func(text)
        return cache[text_hash]
    
    return wrapper

@cache_by_text_hash
def extract_with_cache(text: str) -> Dict:
    """Extract with automatic caching by text hash."""
    return extract_with_github_models(text)
```

## Example: Complete Integration

```python
# src/extractors/github_models_extractor.py

import os
import json
from openai import OpenAI
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class GitHubModelsExtractor:
    """MTC information extractor using GitHub Models."""
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        """Initialize the extractor.
        
        Args:
            model: Model to use (gpt-4o, gpt-4o-mini, etc.)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize client
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ.get("GITHUB_TOKEN"),
        )
        
        logger.info(f"Initialized GitHub Models extractor with {model}")
    
    def extract(self, text: str, schema: Optional[Dict] = None) -> Dict:
        """Extract structured information from certificate text.
        
        Args:
            text: OCR text from certificate
            schema: Optional custom schema to use
            
        Returns:
            Extracted data as dictionary
        """
        system_prompt = self._build_system_prompt(schema)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            extracted = json.loads(response.choices[0].message.content)
            logger.info("Successfully extracted data")
            return extracted
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise
    
    def _build_system_prompt(self, schema: Optional[Dict] = None) -> str:
        """Build the system prompt for extraction."""
        if schema:
            schema_str = json.dumps(schema, indent=2)
        else:
            schema_str = self._get_default_schema()
        
        return f"""You are an expert at extracting structured information from Mill Test Certificates (MTCs).

Extract all available information and return valid JSON matching this schema:

{schema_str}

Rules:
- Use null for missing or unclear values
- Include units with numerical values
- Be precise with numbers (don't round unnecessarily)
- Extract dates in YYYY-MM-DD format
- List all chemical elements found
- Return valid JSON only, no markdown or comments"""
    
    def _get_default_schema(self) -> str:
        """Get the default MTC schema."""
        return """{
  "heat_number": "string or null",
  "material_grade": "string or null",
  "specification": "string or null",
  "chemical_composition": {
    "element_symbol": "percentage as string with unit"
  },
  "mechanical_properties": {
    "yield_strength": "value with unit",
    "tensile_strength": "value with unit",
    "elongation": "percentage",
    "hardness": "value with unit"
  },
  "dimensions": {
    "dimension_name": "value with unit"
  },
  "manufacturer": "string or null",
  "certificate_date": "YYYY-MM-DD or null",
  "standards": ["standard1", "standard2"],
  "lot_number": "string or null",
  "notes": "string or null"
}"""
```

## Environment Configuration

Create a `.env` file for local development:

```bash
# .env
GITHUB_TOKEN=your_github_personal_access_token_here

# Optional: Override default model
DEFAULT_MODEL=gpt-4o-mini

# Optional: API configuration
API_TIMEOUT=30
MAX_RETRIES=3
```

Load in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Testing

```python
# tests/test_github_models_extractor.py

import pytest
from src.extractors.github_models_extractor import GitHubModelsExtractor

@pytest.fixture
def extractor():
    return GitHubModelsExtractor(model="gpt-4o-mini")

def test_basic_extraction(extractor):
    sample_text = """
    MATERIAL TEST CERTIFICATE
    Heat Number: 12345-ABC
    Material Grade: ASTM A516 Grade 70
    Carbon: 0.18%
    Manganese: 1.05%
    """
    
    result = extractor.extract(sample_text)
    
    assert result["heat_number"] == "12345-ABC"
    assert "A516" in result["material_grade"]
    assert "C" in result["chemical_composition"] or "Carbon" in result["chemical_composition"]
```

## Best Practices

1. **Use structured output**: Always request JSON format
2. **Set low temperature**: Use 0.1-0.2 for consistent extraction
3. **Handle errors gracefully**: Wrap API calls in try-except
4. **Log API usage**: Track calls for debugging and cost management
5. **Version your prompts**: Keep prompts in version control
6. **Validate outputs**: Use JSON schema validation
7. **Monitor performance**: Track accuracy, latency, and costs

## Limitations and Considerations

1. **Rate limits**: GitHub Models has usage limits (check current quotas)
2. **Token limits**: Maximum input/output tokens per request
3. **No fine-tuning**: Currently cannot fine-tune models
4. **Internet required**: Cannot work offline
5. **Preview feature**: API may change

## Next Steps

1. Experiment with different models using the provided examples
2. Compare GitHub Models against other solutions in your evaluation
3. Fine-tune prompts for your specific certificate formats
4. Set up automated evaluation pipelines

## Resources

- [GitHub Models Documentation](https://github.com/marketplace/models)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Azure AI Inference API](https://learn.microsoft.com/en-us/azure/ai-studio/reference/reference-model-inference-api)

## Support

For issues with GitHub Models:
- Check [GitHub Models status](https://www.githubstatus.com/)
- Review [GitHub Models discussions](https://github.com/orgs/community/discussions/categories/models)
- Contact GitHub Support for access issues
