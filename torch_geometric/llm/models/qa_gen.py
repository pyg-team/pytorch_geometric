import asyncio
import hashlib
import json
import logging
import os
import pickle
import random
import re
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import faiss
import numpy as np
import pandas as pd
from jinja2 import Template
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging - default to WARNING, --verbose enables INFO/DEBUG
logger = logging.getLogger(__name__)

artifact_extraction_prompt = """
Analyze the following transcript and extract semantic artifacts that would be valuable for generating high-quality question-answer pairs.

TRANSCRIPT:
{{text}}

ARTIFACT TYPES TO EXTRACT:
{{artifact_descriptions}}

INSTRUCTIONS:
1. Extract up to {{max_artifacts}} artifacts for each relevant type
2. Focus on the most significant and informative elements
3. Provide clear, concise descriptions for each artifact
4. Include context about why each artifact is important
5. Ensure artifacts are specific and actionable for Q&A generation

Output your evaluation as a valid JSON object with the following structure:
```json
{
  "key_concepts": [
    {"text": "concept name", "description": "detailed description", "importance": "why it's important"},
    ...
  ],
  "relationships": [
    {"text": "relationship description", "description": "detailed explanation", "importance": "relevance"},
    ...
  ],
  "themes": [...],
  "entities": [...],
  "processes": [...],
  "insights": [...],
  "technical_terms": [...],
  "contextual_factors": [...]
}

Please ensure your output is ONLY the JSON object with no preamble or additional text.
"""
artifact_extraction_prompt_template = Template(artifact_extraction_prompt)

llm_rank_artifacts_prompt = """You are an expert at evaluating semantic artifacts for question-answer generation.

Given the following extracted artifacts from a transcript, select the TOP {{top_k}} artifacts that would be MOST VALUABLE for generating insightful and complex question-answer pairs.

AVAILABLE ARTIFACTS:
{{artifacts_text}}

SELECTION CRITERIA:
1. Prioritize artifacts that enable multi-hop reasoning questions
2. Choose artifacts with rich relationships and connections
3. Select concepts that can generate "why" and "how" questions
4. Prefer artifacts that reveal cause-effect relationships
5. Include artifacts that support synthesis and analysis questions
6. Consider diversity - avoid selecting only one type

RESPOND WITH ONLY THE NUMBERS of your top {{top_k}} choices in order of importance.
Format: [1, 5, 3, 2, 7]

Your selection (top {{top_k}} numbers only):"""
llm_rank_artifacts_prompt_template = Template(llm_rank_artifacts_prompt)

generate_timeline_prompt = """
You are an expert timeline curator creating a rich event log from transcript segments.

Objectives:
- Surface at most {{max_events}} pivotal moments that show progress, decisions, or issues in the session.
- Each event must cite one of the provided segment IDs (1-based numbering) with exact timestamps taken from that segment.
- Provide enough detail (key actions, quotes, implications) so downstream systems can build cross-part reasoning.

Input Segments (use only these, never invent content):
{% for seg in segments %}
- Segment {{seg.segment_id}} ({{seg.start_time}} - {{seg.end_time}}): {{seg.text}}
{% endfor %}

Required JSON output:
{
  "timeline": [
    {
      "segment_id": <int>,
      "start_time": "HH:MM:SS",
      "end_time": "HH:MM:SS",
      "headline": "<= 18 words capturing the moment",
      "key_details": "2 sentences summarising the action/outcome (<= 220 chars)",
      "key_quote": "Exact quote or concise paraphrase from the segment (<= 200 chars)",
      "impact": "Why this matters for securing/operating AI agents (<= 140 chars)",
      "entities": ["Speaker or system names involved" (0-4 items)]
    }
  ]
}

Strict rules:
1. Use only segment IDs provided; keep start/end timestamps within that segment's window.
2. Do not output more than {{max_events}} events; ensure they are ordered chronologically.
3. If you cannot find a quote, use a faithful paraphrase and label it clearly.
4. All strings must be plain text (no markdown) and JSON must be valid.

Return only the JSON object.
Please ensure your output is ONLY the JSON object with no preamble or additional text.
"""
generate_timeline_prompt_template = Template(generate_timeline_prompt)


class Backend(Enum):
    NIM = 'nim'
    VLLM = 'vllm'


class TaskStatus(Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    RETRYING = 'retrying'
    VALIDATING = 'validating'


class LLMClient:
    """LLM Client supporting generation, evaluation, and embedding models.

    Manages three separate models with sleep/wake functionality for efficient
    GPU memory utilization. Only one model is active at a time.
    """
    def __init__(
        self,
        generation_model: str,
        evaluation_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        backend: str = 'nim',
        api_key: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        enable_sleep_mode: bool = True,
    ):
        """Initialize LLMClient with generation, evaluation, and embedding models.

        Args:
            generation_model: Model name/path for text generation
            evaluation_model: Model name/path for evaluation (defaults to generation_model)
            embedding_model: Model name/path for embeddings (defaults to generation_model)
            backend: Backend to use (NIM or VLLM)
            api_key: API key (for NIM backend)
            tensor_parallel_size: Number of GPUs for tensor parallelism (VLLM)
            gpu_memory_utilization: GPU memory utilization 0-1 (VLLM)
            max_model_len: Maximum context length (VLLM)
            enable_sleep_mode: Whether to enable sleep mode for VLLM models
        """
        self.backend = backend
        self.generation_model_name = generation_model
        self.evaluation_model_name = evaluation_model or generation_model
        self.embedding_model_name = embedding_model or generation_model
        self.api_key = api_key
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enable_sleep_mode = enable_sleep_mode

        # Track which model is currently active (for VLLM sleep/wake)
        self._active_model: Optional[str] = None

        # Initialize clients based on backend
        if backend == 'vllm':
            self._init_vllm_clients()
        elif backend == 'nim':
            self._init_nim_clients()
        else:
            raise ValueError(f'Invalid backend: {backend}')

    def _init_vllm_clients(self) -> None:
        """Initialize all three vLLM clients with sleep mode enabled."""
        # Generation model
        self.generation_client = self._create_vllm_client(
            self.generation_model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            enable_sleep_mode=self.enable_sleep_mode,
        )

        # Evaluation model (may be same as generation)
        if self.evaluation_model_name == self.generation_model_name:
            self.evaluation_client = self.generation_client
        else:
            self.evaluation_client = self._create_vllm_client(
                self.evaluation_model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                enable_sleep_mode=self.enable_sleep_mode,
            )

        # Embedding model
        if self.embedding_model_name == self.generation_model_name:
            self.embedding_client = self.generation_client
        elif self.embedding_model_name == self.evaluation_model_name:
            self.embedding_client = self.evaluation_client
        else:
            self.embedding_client = self._create_vllm_client(
                self.embedding_model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                enable_sleep_mode=self.enable_sleep_mode,
                task='embed',
            )

        # Put all models to sleep initially
        if self.enable_sleep_mode:
            self._sleep_all()

    def update_generation_model(self, model: str) -> None:
        if model == self.generation_model_name:
            return

        self.generation_model_name = model
        if self.backend == Backend.VLLM:
            del self.generation_client
            self.generation_client = self._create_vllm_client(
                self.generation_model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                enable_sleep_mode=self.enable_sleep_mode,
            )

    def _init_nim_clients(self) -> None:
        """Initialize NIM client (shared for all operations)."""
        self.nim_client = self._create_nim_client()

    def _create_vllm_client(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        enable_sleep_mode: bool = True,
        task: str = 'generate',
    ) -> 'LLM':
        """Create local vLLM client with OpenAI-compatible API.

        Args:
            model: Model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0-1)
            max_model_len: Maximum context length
            enable_sleep_mode: Whether to enable sleep mode
            task: Task type ('generate' or 'embed')

        Returns:
            vLLM client with OpenAI-compatible interface
        """
        try:
            from vllm import LLM
        except ImportError as err:
            raise ImportError(
                'vLLM is not installed. Please install it with: pip install vllm'
            ) from err

        logger.info('Initializing local vLLM with model: %s (task=%s)', model,
                    task)
        llm_kwargs = {
            'model': model,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'enable_sleep_mode': enable_sleep_mode,
        }
        if max_model_len is not None:
            llm_kwargs['max_model_len'] = max_model_len
        if task == 'embed':
            llm_kwargs['task'] = 'embed'

        vllm_model = LLM(**llm_kwargs)
        logger.info('Local vLLM initialized successfully: %s', model)
        return vllm_model

    def _create_nim_client(self) -> OpenAI:
        """Create NIM (NVIDIA API) client."""
        api_key = os.getenv('NVIDIA_API_KEY')

        if not api_key:
            raise ValueError('NVIDIA_API_KEY environment variable is not set')

        return OpenAI(base_url='https://integrate.api.nvidia.com/v1',
                      api_key=api_key)

    def _sleep_all(self) -> None:
        """Put all vLLM models to sleep."""
        if self.backend != Backend.VLLM or not self.enable_sleep_mode:
            return

        seen = set()
        for client in [
                self.generation_client,
                self.evaluation_client,
                self.embedding_client,
        ]:
            if id(client) not in seen:
                seen.add(id(client))
                client.sleep()
        self._active_model = None

    def _activate_model(self, model_type: str) -> None:
        """Activate a specific model, putting the previous one to sleep.

        Args:
            model_type: One of 'generation', 'evaluation', or 'embedding'
        """
        if self.backend != Backend.VLLM or not self.enable_sleep_mode:
            return

        if self._active_model == model_type:
            return  # Already active

        # Get the client for the requested model type
        client_map = {
            'generation': self.generation_client,
            'evaluation': self.evaluation_client,
            'embedding': self.embedding_client,
        }

        new_client = client_map[model_type]

        # Put previous model to sleep if different
        if self._active_model is not None:
            old_client = client_map[self._active_model]
            if id(old_client) != id(new_client):
                old_client.sleep()

        # Wake up new model
        new_client.wake_up()
        self._active_model = model_type

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        model_type: str,
        temperature: float = 0,
        max_tokens: int = 100000,
    ) -> str:
        """Internal chat method for generation and evaluation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model_type: One of 'generation' or 'evaluation'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated/evaluated text string
        """
        self._activate_model(model_type)

        # Get the appropriate client and model name
        if model_type == 'generation':
            vllm_client = (self.generation_client
                           if self.backend == Backend.VLLM else None)
            model_name = self.generation_model_name
        else:  # evaluation
            vllm_client = (self.evaluation_client
                           if self.backend == Backend.VLLM else None)
            model_name = self.evaluation_model_name

        if self.backend == Backend.VLLM:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=1,
                max_tokens=max_tokens,
                seed=33,
            )
            outputs = vllm_client.chat(messages, sampling_params)
            return outputs[0].outputs[0].text
        else:
            completion = self.nim_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=1,
                max_tokens=max_tokens,
                stream=True,
                seed=33,
            )
            return ''.join(chunk.choices[0].delta.content
                           for chunk in completion
                           if chunk.choices[0].delta.content)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0,
        max_tokens: int = 100000,
    ) -> str:
        """Generate text using the generation model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text string
        """
        return self._chat(messages, 'generation', temperature, max_tokens)

    def evaluate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0,
        max_tokens: int = 100000,
    ) -> str:
        """Evaluate using the evaluation model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Evaluation result text string
        """
        return self._chat(messages, 'evaluation', temperature, max_tokens)

    def embed(
        self,
        prompts: List[str],
        input_type: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate embeddings using the embedding model.

        Args:
            prompts: List of text strings to embed
            input_type: Text input type

        Returns:
            List of embedding vectors (list of floats)
        """
        self._activate_model('embedding')

        if self.backend == Backend.VLLM:
            outputs = self.embedding_client.embed(prompts)
            return [output.outputs.embedding for output in outputs]
        else:
            # NIM embedding API
            embeddings = []
            for prompt in prompts:
                extra_body = {'truncate': 'NONE'}
                if input_type is not None:
                    extra_body['input_type'] = input_type
                response = self.nim_client.embeddings.create(
                    model=self.embedding_model_name,
                    input=prompt,
                    encoding_format='float',
                    extra_body=extra_body,
                )
                embeddings.append(response.data[0].embedding)
            return embeddings


class QAGenerationState(TypedDict):
    """State for the QA generation workflow."""

    task_id: str
    file_path: str
    input_dir: Optional[str]
    output_dir: str
    qa_pairs: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    negative_answers: List[List[str]]
    model_selection: str
    retry_count: int
    error_messages: List[str]
    status: str
    metrics: Dict[str, Any]
    feedback: Dict[str, Any]
    client: Optional[LLMClient]  # OpenAI client
    num_pairs: int  # Number of QA pairs to generate
    num_negatives: int  # Number of negative answers per question
    dedup_threshold: float  # Deduplication threshold
    parts: int
    evaluation: Dict[str, Any]
    evaluation_metrics: Dict[str, Any]  # Metrics from LLM evaluation
    quality_threshold: float  # Minimum acceptable quality score
    hard: bool
    use_artifact: bool
    min_complexity: int
    query_type_distribution: str
    reasoning_type_distribution: Optional[str]
    min_hops: int
    max_hops: int
    models: Dict[str, str]
    segments: List[Dict[str, Any]]
    summary: List[Dict[str, Any]]
    top_artifacts: List[str]
    validated_pairs: List[Dict[str, Any]]
    high_quality_qa_pairs: List[Dict[str, Any]]
    not_evaluated_pairs: List[Dict[str, Any]]
    low_quality_qa_pairs: List[Dict[str, Any]]
    cross_part_contexts: List[Dict[str, Any]]
    self_contained_question: (
        bool  # Generate questions without referencing context/transcript
    )
    question_only: bool
    hard_negatives_top_k: int  # Number of hard negatives to mine per question
    hard_negatives_min_sim: (
        float  # Minimum similarity threshold (too dissimilar = not useful)
    )
    hard_negatives_max_sim: float  # Maximum similarity threshold (too similar = likely false negative)
    hard_negatives_stats: Dict[str, Any]  # Statistics about hard negatives
    enable_hard_negatives: bool
    text_artifacts: List[Any]
    qa_data_to_write: List[Dict[str, Any]]
    model_progression: List[str]


class ArtifactExtractor:
    """LLM-driven artifact extractor that identifies key semantic elements
    from transcripts to enhance Q&A generation quality.
    """
    def __init__(
        self,
        client: Optional['LLMClient'],
    ):
        self.client = client
        self.model = client.generation_model_name
        self.artifact_storage = {}  # In-memory storage for artifacts
        self.artifact_types = {
            'key_concepts': 'Important concepts, terms, or ideas mentioned',
            'relationships': 'Connections and relationships between concepts',
            'themes': 'Overarching themes and topics',
            'entities': 'People, organizations, locations, and specific items',
            'processes': 'Procedures, methods, or sequences described',
            'insights': 'Key insights, conclusions, or findings',
            'technical_terms': 'Domain-specific terminology and definitions',
            'contextual_factors': 'Background information and context',
        }

    def extract_artifacts(
            self, text: str,
            max_artifacts_per_type: int = 8) -> List[Dict[str, Any]]:
        """Extract semantic artifacts from transcript text using LLM.

        Args:
            text: Input transcript text
            max_artifacts_per_type: Maximum artifacts to extract per type

        Returns:
            List of artifacts with relevance scores
        """
        if not self.client:
            logger.warning(
                'No OpenAI client available, returning empty artifacts')
            return []

        try:
            # Create extraction prompt
            prompt = self._create_extraction_prompt(text,
                                                    max_artifacts_per_type)

            logger.debug('extract_artifacts prompt: %s', prompt)
            # Call LLM for artifact extraction
            messages = [
                {
                    'role':
                    'system',
                    'content':
                    'You are an expert at extracting semantic artifacts from text for enhanced Q&A generation.',
                },
                {
                    'role': 'user',
                    'content': prompt
                },
            ]

            response_text = self.client.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=10000,
            )

            # Parse and structure artifacts
            artifacts = self._parse_artifacts_response(response_text)

            # Calculate relevance scores
            artifacts = self._calculate_relevance_scores(artifacts, text)

            # Store artifacts with hash key
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            self.artifact_storage[text_hash] = artifacts

            # Rank and filter top artifacts
            top_artifacts = self._rank_and_filter_artifacts(artifacts)

            logger.info('Extracted %d high-quality artifacts',
                        len(top_artifacts))
            return top_artifacts

        except Exception as e:
            logger.error('Error extracting artifacts: %s', e)
            return []

    def _create_extraction_prompt(self, text: str, max_artifacts: int) -> str:
        """Create prompt for artifact extraction."""
        artifact_descriptions = '\n'.join([
            f'- **{name}**: {desc}'
            for name, desc in self.artifact_types.items()
        ])

        text = text[:3000] + '...' if len(text) > 3000 else text
        return artifact_extraction_prompt_template.render(
            text=text,
            artifact_descriptions=artifact_descriptions,
            max_artifacts=max_artifacts,
        )

    def _parse_artifacts_response(self,
                                  response_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response and structure artifacts."""
        artifacts = []

        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text,
                                   re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'```\s*(.*?)\s*```', response_text,
                                       re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON without code blocks
                    json_str = response_text

            # Parse JSON
            try:
                artifact_data = json.loads(json_str)
            except Exception as e:
                logger.error(
                    'Error in parse_artifacts_response: %s. json_str: %s',
                    e,
                    json_str,
                )
                logger.info('Try again')
                try:
                    # Remove only from the very beginning and very end
                    cleaned = re.sub(r'^```json\s*', '',
                                     json_str)  # Remove from start
                    cleaned = re.sub(r'\s*```$', '',
                                     cleaned)  # Remove from end
                    return json.loads(cleaned)
                except Exception as e1:
                    logger.error('Error again in parse_artifacts_response: %s',
                                 e1)
                    raise
            # Structure artifacts with type information
            for artifact_type, items in artifact_data.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            artifact = {
                                'type': artifact_type,
                                'text': item.get('text', ''),
                                'description': item.get('description', ''),
                                'importance': item.get('importance', ''),
                                'relevance_score': 0.0,  # Will be calculated
                            }
                            artifacts.append(artifact)

        except json.JSONDecodeError as e2:
            logger.error(
                'Error parsing JSON response: %s. response_text: %s',
                e2,
                json_str,
            )

            # Fallback: try to extract key information from text
            artifacts = self._extract_fallback_artifacts(response_text)

        return artifacts

    def _extract_fallback_artifacts(self, text: str) -> List[Dict[str, Any]]:
        """Fallback method to extract artifacts from non-JSON response."""
        artifacts = []

        # Simple pattern matching for different artifact types
        patterns = {
            'key_concepts': r'(?:concept|idea|principle)s?:\s*([^\n]+)',
            'entities': r'(?:person|organization|location)s?:\s*([^\n]+)',
            'themes': r'(?:theme|topic)s?:\s*([^\n]+)',
        }

        for artifact_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:3]:  # Limit to 3 per type
                artifacts.append({
                    'type': artifact_type,
                    'text': match.strip(),
                    'description': match.strip(),
                    'importance': 'Extracted from transcript',
                    'relevance_score': 0.5,
                })

        return artifacts

    def _calculate_relevance_scores(self, artifacts: List[Dict[str, Any]],
                                    text: str) -> List[Dict[str, Any]]:
        """Calculate relevance scores for artifacts based on context."""
        text_lower = text.lower()
        text_words = set(text_lower.split())

        for artifact in artifacts:
            # Calculate relevance based on multiple factors
            score = 0.0

            # Factor 1: Frequency of artifact text in original
            artifact_text = artifact.get('text', '').lower()
            if artifact_text:
                frequency = text_lower.count(artifact_text)
                score += min(frequency / 10, 0.3)  # Max 0.3 from frequency

            # Factor 2: Word overlap
            artifact_words = set(artifact_text.split())
            overlap = len(artifact_words & text_words)
            if artifact_words:
                score += (overlap /
                          len(artifact_words)) * 0.3  # Max 0.3 from overlap

            # Factor 3: Type importance (some types are inherently more important)
            type_weights = {
                'key_concepts': 0.25,
                'relationships': 0.20,
                'insights': 0.20,
                'themes': 0.15,
                'processes': 0.15,
                'entities': 0.10,
                'technical_terms': 0.10,
                'contextual_factors': 0.05,
            }
            score += type_weights.get(artifact.get('type', ''), 0.1)

            # Factor 4: Length and detail of description
            description_length = len(artifact.get('description', ''))
            if description_length > 50:
                score += 0.1

            # Normalize score to 0-1 range
            artifact['relevance_score'] = min(score, 1.0)

        return artifacts

    def _rank_and_filter_artifacts(
            self, artifacts: List[Dict[str, Any]],
            min_score: float = 0.3) -> List[Dict[str, Any]]:
        """Rank artifacts by relevance and filter low-quality ones."""
        # Filter artifacts with minimum score
        filtered = [
            a for a in artifacts if a.get('relevance_score', 0) >= min_score
        ]

        # Sort by relevance score
        filtered.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        # Ensure diversity - get top artifacts from each type
        diverse_artifacts = []
        type_counts = {}

        for artifact in filtered:
            artifact_type = artifact.get('type', 'unknown')
            if type_counts.get(artifact_type, 0) < 3:  # Max 3 per type
                diverse_artifacts.append(artifact)
                type_counts[artifact_type] = (
                    type_counts.get(artifact_type, 0) + 1)

        return diverse_artifacts

    def get_top_artifacts_for_qa(
        self,
        artifacts: List[Dict[str, Any]],
        top_k: int = 5,
        use_llm_ranking: bool = True,
    ) -> List[Dict[str, Any]]:
        """Select top K artifacts most suitable for Q&A generation.

        Args:
            artifacts: List of all artifacts
            top_k: Number of top artifacts to select
            use_llm_ranking: Whether to use LLM-based ranking (True) or rule-based (False)

        Returns:
            List of top artifacts for Q&A enhancement
        """
        if not artifacts:
            return []

        if use_llm_ranking and self.client:
            # Use LLM-based intelligent ranking
            return self._llm_rank_artifacts_for_qa(artifacts, top_k)
        else:
            # Fallback to rule-based ranking
            return self._rule_based_ranking(artifacts, top_k)

    def _rule_based_ranking(self, artifacts: List[Dict[str, Any]],
                            top_k: int) -> List[Dict[str, Any]]:
        """Original rule-based ranking method."""
        # Prioritize certain types for Q&A generation
        qa_type_weights = {
            'key_concepts': 1.5,
            'relationships': 1.3,
            'insights': 1.2,
            'processes': 1.1,
            'themes': 1.0,
            'technical_terms': 0.9,
            'entities': 0.8,
            'contextual_factors': 0.7,
        }

        # Adjust scores based on Q&A suitability
        for artifact in artifacts:
            artifact_type = artifact.get('type', 'unknown')
            weight = qa_type_weights.get(artifact_type, 1.0)
            artifact['qa_score'] = artifact.get('relevance_score', 0) * weight

        # Sort by Q&A score and select top K
        artifacts.sort(key=lambda x: x.get('qa_score', 0), reverse=True)

        return artifacts[:top_k]

    def _llm_rank_artifacts_for_qa(self, artifacts: List[Dict[str, Any]],
                                   top_k: int) -> List[Dict[str, Any]]:
        """Use LLM to intelligently rank artifacts for Q&A generation.

        Args:
            artifacts: List of all artifacts
            top_k: Number of top artifacts to select

        Returns:
            List of top artifacts selected by LLM
        """
        if not self.client or not artifacts:
            return self._rule_based_ranking(artifacts, top_k)

        try:
            # Format artifacts for LLM evaluation
            artifacts_text = ''
            for i, artifact in enumerate(artifacts[:15],
                                         1):  # Limit to 15 for prompt size
                artifacts_text += f'\n{i}. [{artifact["type"]}] {artifact["text"]}: {artifact["description"]}'

            prompt = llm_rank_artifacts_prompt_template.render(
                top_k=top_k, artifacts_text=artifacts_text)

            messages = [
                {
                    'role':
                    'system',
                    'content':
                    'You are an expert at evaluating content for educational Q&A generation.',
                },
                {
                    'role': 'user',
                    'content': prompt
                },
            ]

            response_text = self.client.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=100,
            )

            numbers = re.findall(r'\d+', response_text)
            selected_indices = [int(n) - 1 for n in numbers[:top_k]
                                ]  # Convert to 0-based index

            # Get the selected artifacts
            selected_artifacts = []
            for idx in selected_indices:
                if 0 <= idx < len(artifacts):
                    # Add LLM ranking score
                    artifact = artifacts[idx].copy()
                    artifact['llm_rank'] = len(
                        selected_indices) - selected_indices.index(idx)
                    artifact['qa_score'] = 1.0 - (
                        selected_indices.index(idx) * 0.1
                    )  # Higher score for earlier selections
                    selected_artifacts.append(artifact)

            # If we didn't get enough, fill with rule-based selection
            if len(selected_artifacts) < top_k:
                remaining = self._rule_based_ranking(artifacts, top_k * 2)
                for artifact in remaining:
                    if (artifact not in selected_artifacts
                            and len(selected_artifacts) < top_k):
                        selected_artifacts.append(artifact)

            logger.info(
                'LLM selected %d artifacts for Q&A generation',
                len(selected_artifacts),
            )
            return selected_artifacts[:top_k]

        except Exception as e:
            logger.warning(
                'LLM ranking failed, falling back to rule-based: %s', e)
            return self._rule_based_ranking(artifacts, top_k)

    def format_artifacts_for_prompt(self, artifacts: List[Dict[str,
                                                               Any]]) -> str:
        """Format artifacts into a string suitable for inclusion in Q&A generation prompt.

        Args:
            artifacts: List of artifacts to format

        Returns:
            Formatted string of artifacts for prompt enhancement
        """
        if not artifacts:
            return ''

        formatted_sections = []

        # Group artifacts by type
        artifacts_by_type = {}
        for artifact in artifacts:
            artifact_type = artifact.get('type', 'unknown')
            if artifact_type not in artifacts_by_type:
                artifacts_by_type[artifact_type] = []
            artifacts_by_type[artifact_type].append(artifact)

        # Format each type section
        for artifact_type, items in artifacts_by_type.items():
            type_label = artifact_type.replace('_', ' ').title()
            section = f'\n{type_label}:'
            for item in items:
                section += f'\n- {item["text"]}: {item["description"]}'
            formatted_sections.append(section)

        return '\n'.join(formatted_sections)

    def save_artifacts_to_file(self, filepath: str):
        """Save extracted artifacts to file for persistence."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.artifact_storage, f)
            logger.info('Saved artifacts to %s', filepath)
        except Exception as e:
            logger.error('Error saving artifacts: %s', e)

    def load_artifacts_from_file(self, filepath: str):
        """Load previously extracted artifacts from file."""
        try:
            with open(filepath, 'rb') as f:
                self.artifact_storage = pickle.load(f)
            logger.info('Loaded artifacts from %s', filepath)
        except Exception as e:
            logger.error('Error loading artifacts: %s', e)


def split_segments_text_structured(segments: List[dict],
                                   parts: int = 3) -> List[str]:
    """Most structured approach: Clear section headers and organized timestamp info."""
    total = len(segments)
    if total == 0:
        return []

    section_size = max(1, total // parts)
    splits = []

    for i in range(parts):
        start_idx = i * section_size
        end_idx = (i + 1) * section_size if i < parts - 1 else total

        section_segments = []
        for j, seg in enumerate(segments[start_idx:end_idx]):
            text = seg.get('text', '').strip()
            if not text:
                continue

            start_time = seg.get('start', 0)
            end_time = seg.get('end', 0)

            start_formatted = format_timestamp(start_time)
            end_formatted = format_timestamp(end_time)

            # Structured format with 1-based segment numbering
            segment_info = f'Segment {start_idx + j + 1} ({start_formatted} - {end_formatted}): {text}'
            section_segments.append(segment_info)

        if section_segments:
            # Add section header
            section_text = f'=== Section {i + 1} ===\n' + '\n'.join(
                section_segments)
            splits.append(section_text)

    return splits


def format_timestamp(seconds: float) -> str:
    """Converts seconds to H+:MM:SS format, allowing arbitrary hour lengths."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'


def text_to_sentence_chunks(text: str,
                            sentences_per_chunk: int = 5) -> List[dict]:
    """Alternative: Chunk by sentences for more natural boundaries."""
    logger.debug('text length: %d', len(text))
    import re

    # Simple sentence splitting (you could use nltk for better results)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    logger.debug('num sentences: %d', len(sentences))

    chunks = []
    word_position = 0

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk_text = '. '.join(chunk_sentences)
        if chunk_text and not chunk_text.endswith('.'):
            chunk_text += '.'

        chunk_words = chunk_text.split()
        start_word_pos = word_position
        end_word_pos = word_position + len(chunk_words)
        word_position = end_word_pos

        chunks.append({
            'text': chunk_text,
            'start': start_word_pos,
            'end': end_word_pos,
            'sentence_count': len(chunk_sentences),
            'word_count': len(chunk_words),
            'chunk_id': len(chunks) + 1,
        })

    logger.debug('num chunks: %d', len(chunks))
    return chunks


def get_text_artifacts(segments, client):
    text_extractor = ArtifactExtractor(client)
    text = ' '.join([seg['text'] for seg in segments])
    return text_extractor.extract_artifacts(text)


def split_segments(segments: List[dict], parts: int = 3) -> List:
    """Split transcript segments into `parts` sections for diverse sampling."""
    total = len(segments)
    if total == 0:
        return []
    section_size = max(1, total // parts)
    splits = []
    for i in range(parts):
        start, end = (
            i * section_size,
            (i + 1) * section_size if i < parts - 1 else total,
        )
        splits.append([seg for seg in segments[start:end]])
    return splits


# Node functions for LangGraph
async def load_input(state: QAGenerationState) -> Dict[str, Any]:
    """Load data from file."""
    logger.info('Loading file for task %s', state['task_id'])
    file_path = state['file_path']
    sentences_per_chunk = 5

    # PLAIN TEXT FILE
    logger.info('[%s] Processing as plain text file', file_path)
    with open(file_path, encoding='utf-8') as f:
        text = f.read()

    if not text.strip():
        logger.warning('[%s] No text found; skipping.', file_path)
        return {
            'segments': [],
            'summary': [],
            'status': TaskStatus.PROCESSING.value,
        }

    # Convert text to chunks (this creates the 'segments' structure)
    processing_segments = text_to_sentence_chunks(text, sentences_per_chunk)
    summary = []  # No summary for plain text
    logger.info('num segments: %d', len(processing_segments))
    return {
        'segments': processing_segments,
        'summary': summary,
        'status': TaskStatus.PROCESSING.value,
    }


async def create_artifacts(state: QAGenerationState) -> Dict[str, Any]:
    logger.info('Create artifacts for task %s', state['task_id'])
    client = state.get('client')
    use_artifact = state.get('use_artifact')
    logger.debug('use_artifact? %s', use_artifact)
    if not use_artifact:
        return {'top_artifacts': None}

    segments = state.get('segments', [])
    try:
        text_artifacts = state.get('text_artifacts')
        if not text_artifacts:
            text_artifacts = get_text_artifacts(segments, client)

        logger.info('text artifacts: %d', len(text_artifacts))

        parts = state.get('parts', 3)
        artifacts_str_splits = []
        splits = split_segments(segments, parts=parts)
        for _split in splits:
            all_artifacts = text_artifacts

            # Select top artifacts for this split
            text_extractor = ArtifactExtractor(client)
            top_artifacts = text_extractor.get_top_artifacts_for_qa(
                all_artifacts, top_k=5, use_llm_ranking=True)

            logger.debug('all_artifacts: %d', len(all_artifacts))
            logger.debug('top_artifacts: %d', len(top_artifacts))
            top_artifacts_str = text_extractor.format_artifacts_for_prompt(
                top_artifacts)
            logger.debug('top_artifacts_str: %d', len(top_artifacts_str))
            artifacts_str_splits.append(top_artifacts_str)

        return {'top_artifacts': artifacts_str_splits}
    except Exception as e:
        logger.warning('create_artifacts failed %s', e)
        return {'top_artifacts': None}


def _normalize_segment_metadata(
    segments: List[Dict[str, Any]], ) -> List[Dict[str, Any]]:
    metadata = []
    for idx, seg in enumerate(segments):
        seg_id = (seg.get('segment_id') or seg.get('id')
                  or seg.get('segmentIndex') or idx + 1)
        start_seconds = seg.get('start')
        end_seconds = seg.get('end')
        if isinstance(start_seconds, str):
            try:
                start_seconds = float(start_seconds)
            except ValueError:
                start_seconds = None
        if isinstance(end_seconds, str):
            try:
                end_seconds = float(end_seconds)
            except ValueError:
                end_seconds = None
        metadata.append({
            'segment_id':
            int(seg_id),
            'start_seconds':
            start_seconds,
            'end_seconds':
            end_seconds,
            'start_time':
            format_timestamp(start_seconds)
            if start_seconds is not None else '00:00:00',
            'end_time':
            format_timestamp(end_seconds)
            if end_seconds is not None else '00:00:00',
            'text':
            seg.get('text', '').strip(),
        })
    return metadata


def extract_json_from_response(response_text: str) -> List[dict]:
    match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    if not match:
        raise ValueError('No JSON array found in response text.')
    try:
        return json.loads(match.group(0))
    except Exception as e:
        logger.error(
            'Error in extract_json_from_response : %s. Response_text: %s',
            e,
            response_text,
        )
        logger.info('Try again')
        try:
            # Remove only from the very beginning and very end
            cleaned = re.sub(r'^```json\s*', '',
                             response_text)  # Remove from start
            cleaned = re.sub(r'^```\s*', '', cleaned)  # Remove from start
            cleaned = re.sub(r'\s*```$', '', cleaned)  # Remove from end
            return json.loads(cleaned)
        except Exception as e1:
            logger.error('Error again in extract_json_from_response: %s', e1)
            raise


def extract_timeline_from_response(response_text: str) -> Dict[str, Any]:
    try:
        data = json.loads(response_text)
        if isinstance(data, list):
            return {'timeline': data}
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    try:
        obj = extract_json_from_response(response_text)
        if isinstance(obj, list):
            return {'timeline': obj}
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # attempt enhanced extraction by looking for first JSON object/array
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        snippet = match.group(0)
        try:
            data = json.loads(snippet)
            if isinstance(data, list):
                return {'timeline': data}
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    raise ValueError(
        'Failed to extract structured timeline JSON from response')


def parse_timestamp(ts: str) -> int:
    """Parses a timestamp string H+:MM:SS into total seconds."""
    parts = ts.split(':')
    if len(parts) != 3:
        raise ValueError(f'Invalid timestamp format: {ts}')
    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s


def validate_timeline_events(
        events: List[Dict[str, Any]],
        segment_lookup: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    validated: List[Dict[str, Any]] = []
    seen_ids = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        seg_id = event.get('segment_id')
        try:
            seg_id = int(seg_id)
        except (TypeError, ValueError):
            continue
        segment = segment_lookup.get(seg_id)
        if not segment:
            continue

        start_time = event.get('start_time') or segment['start_time']
        end_time = event.get('end_time') or segment['end_time']
        try:
            start_seconds = parse_timestamp(start_time)
            end_seconds = parse_timestamp(end_time)
        except Exception:
            start_seconds = segment.get('start_seconds')
            end_seconds = segment.get('end_seconds')

        seg_start = segment.get('start_seconds')
        seg_end = segment.get('end_seconds')
        if seg_start is not None and (start_seconds is None
                                      or start_seconds < seg_start):
            start_seconds = seg_start
            start_time = segment['start_time']
        if seg_end is not None and (end_seconds is None
                                    or end_seconds > seg_end):
            end_seconds = seg_end
            end_time = segment['end_time']
        if (start_seconds is None or end_seconds is None
                or start_seconds > end_seconds):
            continue

        headline = (event.get('headline') or '').strip()
        if not headline:
            headline = segment['text'][:90]
        headline = headline[:140]

        key_details = (event.get('key_details') or '').strip()
        if not key_details:
            key_details = segment['text'][:220]
        key_details = key_details[:260]

        key_quote = (event.get('key_quote') or '').strip()
        if not key_quote:
            key_quote = segment['text'][:200]
        key_quote = key_quote[:220]

        impact = (event.get('impact') or '').strip()
        if not impact:
            impact = ''
        impact = impact[:180]

        entities = event.get('entities')
        if isinstance(entities, list):
            entities = [
                str(e).strip()[:60] for e in entities if str(e).strip()
            ]
            entities = entities[:4]
        else:
            entities = []

        unique_key = (seg_id, headline)
        if unique_key in seen_ids:
            continue
        seen_ids.add(unique_key)

        validated.append({
            'segment_id': seg_id,
            'start_time': format_timestamp(start_seconds),
            'end_time': format_timestamp(end_seconds),
            'start_seconds': start_seconds,
            'end_seconds': end_seconds,
            'headline': headline,
            'key_details': key_details,
            'key_quote': key_quote,
            'impact': impact,
            'entities': entities,
            'source': 'timeline',
            'segment_text': segment['text'],
        })
    validated.sort(key=lambda ev: ev.get('start_seconds', 0))
    return validated


def generate_structured_timeline(
    segments: List[Dict[str, Any]],
    client: LLMClient,
    max_events: int = 18,
    max_segments: int = 400,
) -> List[Dict[str, Any]]:
    if not segments:
        return []
    segment_metadata = _normalize_segment_metadata(segments)
    prompt_segments = []
    for seg in segment_metadata[:max_segments]:
        text = seg['text'].replace('\n', ' ')
        prompt_segments.append({
            'segment_id': seg['segment_id'],
            'start_time': seg['start_time'],
            'end_time': seg['end_time'],
            'text': text[:220],
        })
    prompt = generate_timeline_prompt_template.render(segments=prompt_segments,
                                                      max_events=max_events)
    messages = [
        {
            'role': 'system',
            'content': 'Expert timeline summarizer.'
        },
        {
            'role': 'user',
            'content': prompt
        },
    ]
    response = client.generate(messages, temperature=0.2)
    timeline_data = extract_timeline_from_response(response)
    events = (timeline_data.get('timeline', []) if isinstance(
        timeline_data, dict) else [])
    lookup = {item['segment_id']: item for item in segment_metadata}
    validated = validate_timeline_events(events, lookup)
    return validated[:max_events]


def format_timeline_highlights(events: List[Dict[str, Any]],
                               limit: int = 8) -> List[str]:
    lines = []
    for event in events[:limit]:
        headline = event.get('headline', '').strip()
        details = (event.get('key_details') or '').strip()
        impact = (event.get('impact') or '').strip()
        snippet = f'- [{event["start_time"]} - {event["end_time"]}] (Segment {event["segment_id"]}) {headline}'
        if details:
            snippet += f' — {details}'
        if impact:
            snippet += f' (Impact: {impact})'
        lines.append(snippet[:400])
    return lines


_SEGMENT_LINE_PATTERN = re.compile(
    r'^Segment\s+(?P<id>\d+)\s*\((?P<start>\d{2}:\d{2}:\d{2})\s*-\s*(?P<end>\d{2}:\d{2}:\d{2})\):\s*(?P<text>.*)$'
)


def _parse_structured_lines(
        raw: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Extract clean lines and timestamp metadata from a structured split."""
    lines: List[str] = []
    entries: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('==='):
            continue
        lines.append(stripped)
        match = _SEGMENT_LINE_PATTERN.match(stripped)
        if match:
            segment_id = match.group('id')
            start_ts = match.group('start')
            end_ts = match.group('end')
            text = match.group('text')
            try:
                start_seconds = parse_timestamp(start_ts)
                end_seconds = parse_timestamp(end_ts)
            except Exception:
                start_seconds = end_seconds = None
            entries.append({
                'segment_id': int(segment_id),
                'start': start_ts,
                'end': end_ts,
                'start_seconds': start_seconds,
                'end_seconds': end_seconds,
                'text': text.strip(),
            })
    return lines, entries


def build_cross_part_contexts_from_segments(
        splits: List[str], max_sections: int = 3,
        max_chars: int = 1200) -> List[Dict[str, Any]]:
    """Construct per-part summaries of other transcript sections to encourage
    cross-part multi-hop reasoning while retaining timestamp metadata for
    validation.
    """
    if not splits:
        return []

    parsed_sections: List[List[str]] = []
    section_metadata: List[List[Dict[str, Any]]] = []
    for raw in splits:
        lines, entries = _parse_structured_lines(raw)
        parsed_sections.append(lines)
        section_metadata.append(entries)

    per_section_limit = max(200, max_chars // max(1, max_sections))
    total_splits = len(splits)
    contexts: List[Dict[str, Any]] = []

    for idx in range(total_splits):
        other_sections_text: List[str] = []
        allowed_ranges: List[Dict[str, Any]] = []
        sections_truncated: List[int] = []
        sections_added = 0
        total_chars = 0

        for jdx in range(total_splits):
            if jdx == idx:
                continue
            if sections_added >= max_sections or total_chars >= max_chars:
                break

            lines = parsed_sections[jdx]
            entries = section_metadata[jdx]
            if not lines:
                continue

            snippet_lines: List[str] = []
            char_budget = per_section_limit
            consumed = 0
            entry_idx = 0
            entries_added = 0

            for line in lines:
                line_len = len(line)
                if consumed + line_len + 1 > char_budget:
                    # we have consumed the budget for this section
                    sections_truncated.append(jdx + 1)
                    break
                snippet_lines.append(line)
                consumed += line_len + 1
                total_chars += line_len + 1

                if entry_idx < len(entries):
                    entry = entries[entry_idx]
                    allowed_ranges.append({
                        'start':
                        entry['start'],
                        'end':
                        entry['end'],
                        'start_seconds':
                        entry['start_seconds'],
                        'end_seconds':
                        entry['end_seconds'],
                        'source':
                        f'section_{jdx + 1}',
                    })
                    entry_idx += 1
                    entries_added += 1

                if total_chars >= max_chars:
                    sections_truncated.append(jdx + 1)
                    break

            if snippet_lines:
                formatted = f'Section {jdx + 1}:\n' + '\n'.join(snippet_lines)
                other_sections_text.append(formatted)
                sections_added += 1

        contexts.append({
            'text':
            '\n\n'.join(other_sections_text).strip(),
            'ranges': [
                rng for rng in allowed_ranges
                if rng.get('start') and rng.get('end')
            ],
            'metadata': {
                'total_other_sections': max(0, total_splits - 1),
                'sections_included': len(other_sections_text),
                'sections_truncated': sorted(set(sections_truncated)),
            },
        })

        meta = contexts[-1]['metadata']
        if meta.get('total_other_sections',
                    0) and not meta.get('sections_included'):
            logger.debug(
                f'Cross-part context for section {idx + 1} did not include any additional segments within the current character budget.'
            )
        elif meta.get('sections_truncated'):
            logger.debug(
                f'Cross-part context for section {idx + 1} truncated sections {meta["sections_truncated"]} due to character limits.'
            )

    return contexts


def build_cross_part_contexts_with_timeline(
    splits: List[str],
    timeline_events: List[Dict[str, Any]],
    highlight_limit: int = 8,
    max_external_events: int = 5,
) -> List[Dict[str, Any]]:
    global_highlights = format_timeline_highlights(timeline_events,
                                                   limit=highlight_limit)
    contexts: List[Dict[str, Any]] = []
    parsed_sections: List[Tuple[List[str], List[Dict[str, Any]]]] = []
    for raw in splits:
        parsed_sections.append(_parse_structured_lines(raw))

    for _idx, (_, entries) in enumerate(parsed_sections):
        segment_ids = {
            entry.get('segment_id')
            for entry in entries if entry.get('segment_id') is not None
        }
        external_events = [
            event for event in timeline_events
            if event.get('segment_id') not in segment_ids
        ]
        external_lines = []
        cross_ranges: List[Dict[str, Any]] = []
        for event in external_events[:max_external_events]:
            detail = (event.get('key_details') or '').strip()
            impact = (event.get('impact') or '').strip()
            quote = (event.get('key_quote') or '').strip()
            entry = f'- [{event["start_time"]} - {event["end_time"]}] (Segment {event["segment_id"]}) {event["headline"]}'
            if detail:
                entry += f' — {detail}'
            if impact:
                entry += f' (Impact: {impact})'
            if quote:
                entry += f' | Quote: {quote}'
            external_lines.append(entry[:500])
            cross_ranges.append({
                'start': event['start_time'],
                'end': event['end_time'],
                'start_seconds': event.get('start_seconds'),
                'end_seconds': event.get('end_seconds'),
                'source': f'segment_{event["segment_id"]}',
                'segment_id': event['segment_id'],
            })

        text_lines = []
        if global_highlights:
            text_lines.append('Timeline Highlights:')
            text_lines.extend(global_highlights)
        if external_lines:
            text_lines.append('\nExternal Events to Connect:')
            text_lines.extend(external_lines)
        cross_context_text = '\n'.join(text_lines).strip()

        contexts.append({
            'text': cross_context_text,
            'ranges': cross_ranges,
            'metadata': {
                'source': 'timeline',
                'external_events': len(external_lines),
            },
        })

    return contexts


def transcription_segment_text(segments: List[dict]) -> str:
    """Concatenate segment['text'] into one transcript string."""
    return ' '.join(seg.get('text', '') for seg in segments)


infer_video_style_prompt = """
Based on this transcript excerpt, describe the video's style.
Transcript:\n{{segment_text}}

Please ensure your output is ONLY the style keyword, and no other text.

STYLE:"""
infer_video_style_prompt_template = Template(infer_video_style_prompt)


def infer_video_style(segment_text: str, client: LLMClient) -> str:
    prompt = infer_video_style_prompt_template.render(
        segment_text=segment_text)
    messages = [
        {
            'role': 'system',
            'content': 'Expert video classifier.'
        },
        {
            'role': 'user',
            'content': prompt
        },
    ]
    resp = client.generate(messages, temperature=0.4)
    return resp.splitlines()[0].strip()


def get_summary_block(summary: List[Dict[str, str]]) -> str:
    """Turn a list of segment summaries into a markdown bullet list."""
    if not summary:
        return ''
    lines = [
        f'- **{seg["segment_title"]} ({seg["timestamp"]}):** {seg["description"]}'
        for seg in summary
    ]
    return 'Video summary:\n' + '\n'.join(lines) + '\n\n'


def get_cross_part_context_block(cross_part_context: str) -> str:
    """Format cross-part context snippets for prompt injection."""
    if not cross_part_context:
        return ''
    return 'Cross-part context:\n' + cross_part_context.strip() + '\n\n'


def get_artifact_block_for_hard_question(artifacts_context):
    # Add artifact context if provided
    artifact_block = ''
    if artifacts_context:
        artifact_block = """
EXTRACTED SEMANTIC ARTIFACTS (Use these to create deeper multi-hop questions):
{artifacts_context}
ARTIFACT-ENHANCED HARD QUESTION INSTRUCTIONS:
- Use identified relationships to create questions that connect multiple concepts across the transcript
- Leverage processes and workflows to generate questions about cause-and-effect chains
- Use themes and insights to formulate questions requiring synthesis of multiple viewpoints
- Create questions that explore how technical terms relate to the broader concepts
- Design multi-hop questions that traverse the artifact graph structure
- Focus on questions that require understanding the connections between different artifact types
"""
    return artifact_block


def distribute_counts(total_count: int, distribution: Dict[str, float],
                      name: str = '') -> Dict[str, int]:
    """Distribute a total count across categories based on percentages using smart rounding.

    This algorithm solves the problem of fairly distributing small integers according to
    percentage distributions. The naive approach of using int(count * percentage) often
    results in all items going to the first category due to truncation.

    Algorithm steps:
    1. Calculate exact (floating-point) count for each category
    2. Sort categories by their fractional parts in descending order
    3. Assign floor values to all categories
    4. Distribute remaining items to categories with largest fractional parts
    5. If still items left (rare edge case), assign to the category with highest percentage

    Example:
        With 3 items and distribution {"A": 0.4, "B": 0.3, "C": 0.3}:
        - Exact counts: A=1.2, B=0.9, C=0.9
        - Floor values: A=1, B=0, C=0 (total=1, remaining=2)
        - Sorted by fractional part: B(0.9), C(0.9), A(0.2)
        - Distribute remaining 2: B gets 1, C gets 1
        - Final result: {"A": 1, "B": 1, "C": 1}

        Compare with naive approach resulting in: {"A": 3, "B": 0, "C": 0}

    Args:
        total_count: Total number of items to distribute
        distribution: Dict mapping category names to their target percentage (0-1)
                     Sum of percentages should be 1.0
        name: Optional name for logging purposes (e.g., "Query type")

    Returns:
        Dict mapping category names to their assigned integer counts.
        Sum of all counts will equal total_count.
    """
    counts = {}
    total_assigned = 0

    # Step 1: Calculate exact (floating-point) values for each category
    # Example: 3 * 0.4 = 1.2, 3 * 0.3 = 0.9, 3 * 0.3 = 0.9
    exact_counts = {
        cat: total_count * pct
        for cat, pct in distribution.items()
    }

    # Step 2: Sort by fractional part descending (0.9, 0.9, 0.2)
    # This ensures categories closest to rounding up get priority
    sorted_cats = sorted(exact_counts.items(), key=lambda x: x[1] % 1,
                         reverse=True)

    # Step 3: Assign floor values first (everyone gets their integer part)
    # Example: A gets 1, B gets 0, C gets 0
    for category, exact_count in sorted_cats:
        counts[category] = int(exact_count)
        total_assigned += int(exact_count)

    # Step 4: Distribute remaining items based on fractional parts
    # Example: 3 - 1 = 2 remaining, give to B and C (largest fractions)
    remaining = total_count - total_assigned
    for _i, (category, exact_count) in enumerate(sorted_cats):
        if remaining > 0 and exact_count % 1 > 0:  # Has fractional part
            counts[category] += 1
            remaining -= 1

    # Step 5: Edge case - if still items left (e.g., rounding errors)
    # Give to the category with the highest original percentage
    if remaining > 0:
        largest_cat = max(distribution.items(), key=lambda x: x[1])[0]
        counts[largest_cat] += remaining

    # Log the distribution for debugging
    if name:
        logger.info('%s distribution for %d items: %s', name, total_count,
                    counts)

    return counts


generate_hard_question_answer_with_timestamps_prompt = """
You are an expert at extracting complex question and answer pairs from transcripts.

{{artifact_block}}

Guidelines:
1. Generate ONLY Complex Questions (Complexity 4-5):
   - All questions MUST require understanding connections between different parts of the transcript
   - Questions should test deep understanding, not simple facts
{%- if self_contained_question %}
   - Do not mention the existence of a context/transcript in the generated question like "in the transcript", "from the given context", or "in Segment 148". Produce a natural, standalone question.
   - Only use facts present in the provided context/transcript; if missing, say you cannot generate a question.
{%- endif %}
   - Example: "How does the speaker's initial explanation of X relate to the later implementation of Y?"

2. Question Types to Generate:
   - Multi-hop questions ({{query_counts.get("multi_hop", 0)}} questions): Connect {{min_hops}}-{{max_hops}} separated segments
   - Structural questions ({{query_counts.get("structural", 0)}} questions): Focus on relationships between concepts
   - Contextual questions ({{query_counts.get("contextual", 0)}} questions): Require surrounding context to understand
   - Use the cross-part context snippets to connect evidence that lives outside the current transcript section

{%- if reasoning_counts %}
3. Reasoning Types to Include:
   - Factual questions ({{reasoning_counts.get("factual", 0)}} questions): Ask for complex facts that require synthesizing multiple pieces of information (NOT simple lookups)
   - Relational questions ({{reasoning_counts.get("relational", 0)}} questions): Ask how data points compare or correlate across different segments
   - Inferential questions ({{reasoning_counts.get("inferential", 0)}} questions): Ask about conclusions or implications requiring synthesis
   - Temporal questions ({{reasoning_counts.get("temporal", 0)}} questions): Ask about changes or events over time across segments
   - Procedural questions ({{reasoning_counts.get("procedural", 0)}} questions): Ask about complex multi-step processes or guidelines
   - Visual questions ({{reasoning_counts.get("visual", 0)}} questions): Ask about visual details requiring cross-reference
   - Causal questions ({{reasoning_counts.get("causal", 0)}} questions): Ask about cause-effect chains spanning segments

   Example COMPLEX questions by reasoning type (all must be complexity 4-5):
   - Factual: "What is the total combined budget allocation across all departmental initiatives mentioned, and how does it relate to the overall fiscal year target?"
   - Relational: "How does the performance metric achieved in Q2 compare to both the initial baseline and the revised targets that were set?"
   - Inferential: "Based on the challenges outlined and the proposed solutions, what unstated assumptions underlie the strategic pivot?"
   - Temporal: "How did the implementation timeline evolve from the initial proposal through the mid-year review to the final execution phase?"
   - Procedural: "What is the complete approval workflow including standard requirements, exceptions, and escalation processes?"
   - Visual: "How do the visual elements presented relate to the verbal descriptions provided, and what discrepancies exist between them?"
   - Causal: "What chain of events, starting from the initial decision, led through various complications to the final outcome?"

4. IMPORTANT - Orthogonal Distributions:
   - Each question must have BOTH a query type (multi_hop/structural/contextual) AND a reasoning type
   - For example: A multi_hop factual question or a structural causal question
   - Ensure the final distribution matches both specified percentages
{%- endif %}

Meta-Reasoning Playbook (follow before drafting output):
   1. Scan the current section for key statements and log their timestamps.
   2. Compare against the cross-part context to spot related events elsewhere and capture the matching timestamps.
   3. Design a reasoning chain that depends on evidence from both the local section and external context, outlining each hop.
   4. Confirm each hop's start/end time aligns with the chosen segments, then compose the question and answer around that chain.
   5. Ensure the final answer references all hops and explains why the timestamps substantiate the reasoning.

5. **IMPORTANT - Segment Identification**:
   - The transcript below contains segments formatted as "Segment N (HH:MM:SS - HH:MM:SS): text" where N starts from 1
   - For each question-answer pair you generate, identify ALL segment numbers FROM which the question is derived
   - These segments are the source material that should be retrieved when someone asks this question
   - Record these segment numbers in the "segment_ids" field as a list of integers (e.g., [1, 4, 8])
   - For multi-hop questions:
     * The top-level "segment_ids" should be the UNION of all segment IDs across all hops
     * Each hop in "hop_contexts" should specify its own "segment_ids" list
     * Example: If hop 1 uses [1, 3] and hop 2 uses [6, 8], then top-level segment_ids should be [1, 3, 6, 8]
   - The timestamps will be automatically calculated from the segments you identify


4. For Each Question:
   - Must have complexity level {{min_complexity}} or higher
   - Generate the question FROM the identified segments (these segments are the source material)
   - Multi-hop questions must specify hop_count ({{min_hops}}-{{max_hops}})
   - Provide hop_contexts: a list where each hop includes "hop", "segment_ids" (the source segments for this hop), "start_time", "end_time", and a concise "context" summary describing the supporting segment(s)

5. Generate {{num_pairs}} distinct question and answer pairs.

The output should be a JSON array of objects (with the number of objects equal to {{num_pairs}}), where each object contains:
  - "question": the complex question requiring structural understanding of the contexts/transcripts/segments without explicitly referencing the context/transcript/segments in the question
  - "answer": comprehensive answer from the contexts/transcripts/segments without explicitly referencing the context/transcript/segments in the answer
  - "question_complexity": numeric score {{min_complexity}}-5
  - "query_type": one of ["multi_hop", "structural", "contextual"]
{%- if reasoning_counts %}
  - "reasoning_type": one of ["factual", "relational", "inferential", "temporal", "procedural", "visual", "causal"]
{%- endif %}
  - "segment_ids": list of segment numbers (e.g., [1, 4, 8]) that are the source material for this question (these should be retrieved when the question is asked)
  - "start_time": HH:MM:SS timestamp (will be calculated from earliest segment)
  - "end_time": HH:MM:SS timestamp (will be calculated from latest segment)
  - "hop_count": number of hops ({{min_hops}}-{{max_hops}}) for multi_hop questions only
  - "hop_contexts": array of hop detail objects with "hop", "segment_ids" (source segments for this hop), "start_time", "end_time", "context" (required for every question)

Additional Constraints:
- **Output only the JSON. Do not include any introductory or concluding text.**
- **Please ensure that your entire response is strictly valid JSON with no additional text or formatting.**
- **The output should be a JSON array containing exactly {{num_pairs}} objects.**
- **Pay careful attention to the timestamp markers in the transcript to ensure accurate start_time and end_time values.**

Video Style Guidance: {{video_style}}

{{summary_block}}

{{cross_part_context_block}}

Transcript:
---
{{segment_text}}

Output a JSON array where each element is an object with the required keys.
Please ensure your output is ONLY the JSON object with no preamble or additional text.
"""

generate_hard_question_answer_with_timestamps_template = Template(
    generate_hard_question_answer_with_timestamps_prompt)


def _normalize_time_value(value: Any, fallback: str = '') -> str:
    """Normalize hop context timestamps to HH:MM:SS, falling back when missing."""
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        try:
            return format_timestamp(float(value))
        except Exception:
            return fallback
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or fallback
    return fallback


def normalize_hop_contexts(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize hop_context entries to a consistent structure with exactly 5 fields:
    - hop: hop number (int)
    - segment_ids: list of segment IDs
    - start_time: timestamp string
    - end_time: timestamp string
    - context: description text.

    Args:
        pair: QA pair dictionary containing hop_contexts

    Returns:
        List of normalized hop context dictionaries with 5 fields each
    """
    hop_contexts = pair.get('hop_contexts')
    if hop_contexts is None:
        return []

    # Ensure we have a list
    if isinstance(hop_contexts, dict):
        hop_contexts = [hop_contexts]
    elif not isinstance(hop_contexts, list):
        return []

    normalized: List[Dict[str, Any]] = []
    default_start = pair.get('start_time', '')
    default_end = pair.get('end_time', '')

    for idx, hop_entry in enumerate(hop_contexts, start=1):
        # Skip non-dict entries
        if not isinstance(hop_entry, dict):
            continue

        # Extract and normalize hop number
        hop_num = hop_entry.get('hop') or hop_entry.get('hop_number') or idx
        try:
            hop_num = int(hop_num)
        except (TypeError, ValueError):
            hop_num = idx

        # Extract and normalize segment_ids
        segment_ids = hop_entry.get('segment_ids', [])
        if not isinstance(segment_ids, list):
            if isinstance(segment_ids, (int, str)):
                try:
                    segment_ids = [int(segment_ids)]
                except (TypeError, ValueError):
                    segment_ids = []
            else:
                segment_ids = []

        # Extract and normalize timestamps
        start_time = _normalize_time_value(
            hop_entry.get('start_time') or hop_entry.get('start'),
            default_start)
        end_time = _normalize_time_value(
            hop_entry.get('end_time') or hop_entry.get('end'), default_end)

        # Extract context (required field)
        context = (hop_entry.get('context') or hop_entry.get('summary')
                   or hop_entry.get('text') or '')
        context = context.strip()
        if not context:
            # Skip entries without context
            continue

        # Build normalized entry with exactly 5 fields
        normalized_entry = {
            'hop': hop_num,
            'segment_ids': segment_ids,
            'start_time': start_time,
            'end_time': end_time,
            'context': context,
        }

        normalized.append(normalized_entry)

    # Sort by hop number for consistency
    normalized.sort(key=lambda entry: entry.get('hop', 0))
    pair['hop_contexts'] = normalized
    return normalized


def _range_contains(start: int, end: int, candidate: Dict[str, Any]) -> bool:
    if start is None or end is None:
        return False
    c_start = candidate.get('start_seconds')
    c_end = candidate.get('end_seconds')
    if c_start is None or c_end is None:
        return False
    return c_start <= start and end <= c_end


def _find_best_matching_range(
        start: int, candidates: List[Dict[str,
                                          Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_delta: Optional[int] = None
    for candidate in candidates:
        c_start = candidate.get('start_seconds')
        if c_start is None:
            continue
        delta = abs(c_start - start)
        if best_delta is None or delta < best_delta:
            best = candidate
            best_delta = delta
    return best


def validate_hop_context_time_ranges(
    pair: Dict[str, Any],
    local_ranges: List[Dict[str, Any]],
    cross_ranges: List[Dict[str, Any]],
) -> None:
    """Ensure hop contexts fall within known timestamp ranges. Adjust to the
    closest available match when possible and record validation warnings.
    """
    hop_contexts = pair.get('hop_contexts') or []
    if not hop_contexts:
        return

    warnings: List[str] = []
    for hop_entry in hop_contexts:
        start_ts = hop_entry.get('start_time')
        end_ts = hop_entry.get('end_time')
        try:
            start_seconds = parse_timestamp(start_ts) if start_ts else None
            end_seconds = parse_timestamp(end_ts) if end_ts else None
        except Exception:
            start_seconds = end_seconds = None

        target_range: Optional[Dict[str, Any]] = None
        if start_seconds is not None and end_seconds is not None:
            for candidate in local_ranges:
                if _range_contains(start_seconds, end_seconds, candidate):
                    target_range = candidate.copy()
                    target_range['source'] = candidate.get('source', 'local')
                    break
            if not target_range:
                for candidate in cross_ranges:
                    if _range_contains(start_seconds, end_seconds, candidate):
                        target_range = candidate.copy()
                        target_range['source'] = candidate.get(
                            'source', 'cross')
                        break

        if target_range:
            hop_entry['source'] = target_range.get('source')
            continue

        # Attempt to align with closest known range
        candidates = local_ranges + cross_ranges
        if start_seconds is not None and candidates:
            best_match = _find_best_matching_range(start_seconds, candidates)
        else:
            best_match = None

        if best_match and best_match.get('start') and best_match.get('end'):
            hop_entry['start_time'] = best_match['start']
            hop_entry['end_time'] = best_match['end']
            hop_entry['source'] = best_match.get('source')
            warnings.append(
                f'Hop {hop_entry.get("hop")} timestamps adjusted to nearest available range '
                f'({best_match.get("start")} - {best_match.get("end")}).')
        else:
            warnings.append(
                f'Hop {hop_entry.get("hop")} timestamps ({start_ts} - {end_ts}) do not map to known segments.'
            )

    if warnings:
        pair.setdefault('validation_warnings', []).extend(warnings)


def generate_hard_question_answer_with_timestamps(
    segment_text: str,
    video_style: str,
    num_pairs: int,
    client: LLMClient,
    summary: List[Dict[str, str]] = None,
    query_type_distribution: Dict[str, float] = None,
    reasoning_type_distribution: Dict[str, float] = None,
    min_complexity: int = 4,
    min_hops: int = 2,
    max_hops: int = 4,
    top_artifacts: List[Dict[str, Any]] = None,
    artifacts_context: str = None,
    cross_part_context: str = None,
    allowed_ranges_local: Optional[List[Dict[str, Any]]] = None,
    allowed_ranges_cross: Optional[List[Dict[str, Any]]] = None,
    self_contained_question: bool = False,
) -> List[dict]:
    """Generates EXCLUSIVELY hard question-answer pairs that require graph structure understanding."""
    summary_block = get_summary_block(summary or [])
    cross_part_context_block = get_cross_part_context_block(cross_part_context)
    # Add artifact context if provided
    artifact_block = get_artifact_block_for_hard_question(artifacts_context)

    logger.debug('artifact_block: %s', artifact_block)

    # Default distribution if not provided
    if not query_type_distribution:
        query_type_distribution = {
            'multi_hop': 0.4,
            'structural': 0.3,
            'contextual': 0.3,
        }

    # Use utility function for better distribution
    query_counts = distribute_counts(num_pairs, query_type_distribution,
                                     'Query type')

    # Calculate reasoning type distribution
    reasoning_counts = {}
    if reasoning_type_distribution:
        reasoning_counts = distribute_counts(num_pairs,
                                             reasoning_type_distribution,
                                             'Reasoning type')

    prompt = generate_hard_question_answer_with_timestamps_template.render(
        artifact_block=artifact_block,
        num_pairs=num_pairs,
        query_counts=query_counts,
        reasoning_counts=reasoning_counts,
        min_hops=min_hops,
        max_hops=max_hops,
        min_complexity=min_complexity,
        video_style=video_style,
        summary_block=summary_block,
        cross_part_context_block=cross_part_context_block,
        segment_text=segment_text,
        self_contained_question=self_contained_question,
    )

    logger.debug('final_prompt: %s', prompt)

    messages = [
        {
            'role': 'system',
            'content': 'Complex QA generation expert.'
        },
        {
            'role': 'user',
            'content': prompt
        },
    ]

    max_prompt_attempts = 3
    for attempt in range(1, max_prompt_attempts + 1):
        resp = client.generate(messages, temperature=0.3)
        qa_pairs = extract_json_from_response(resp)

        all_valid = True
        for pair in qa_pairs:
            if pair.get('question_complexity', 0) < min_complexity:
                pair['question_complexity'] = min_complexity

            if 'query_type' not in pair:
                pair['query_type'] = 'multi_hop'

            hop_contexts = normalize_hop_contexts(pair)
            if allowed_ranges_local or allowed_ranges_cross:
                validate_hop_context_time_ranges(pair, allowed_ranges_local
                                                 or [], allowed_ranges_cross
                                                 or [])
            if not hop_contexts:
                all_valid = False
                logger.warning(
                    f'Retrying hard QA (with timestamps) due to missing hop_contexts (attempt {attempt}/{max_prompt_attempts})'
                )
                break

            if pair['query_type'] == 'multi_hop':
                raw_hop_count = pair.get('hop_count')
                try:
                    hop_count = int(raw_hop_count)
                except (TypeError, ValueError):
                    hop_count = None

                if hop_count is None or hop_count < len(hop_contexts):
                    hop_count = max(len(hop_contexts), min_hops)
                if max_hops is not None:
                    hop_count = min(hop_count, max_hops)

                pair['hop_count'] = hop_count
            else:
                for idx, ctx in enumerate(hop_contexts, start=1):
                    ctx['hop'] = idx

        if all_valid:
            return qa_pairs

    raise ValueError(
        'Failed to generate hop_contexts for hard question answers after retries'
    )


def get_artifact_block(artifacts_context):
    # Add artifact context if provided
    artifact_block = ''
    if artifacts_context:
        artifact_block = """
EXTRACTED SEMANTIC ARTIFACTS (Use these to enhance your questions):
{artifacts_context}

ARTIFACT-ENHANCED INSTRUCTIONS:
- Use the semantic artifacts to create deeper, more insightful questions
- Reference relationships and connections identified in the artifacts
- Create questions that explore the themes and insights
- Include questions about the processes and technical terms mentioned
- Ensure answers are comprehensive and reference the artifact context
"""
    return artifact_block


generate_question_answer_with_timestamps_prompt = """
You are an expert at extracting question and answer pairs from transcripts.

{{artifact_block}}

Guidelines:
1. Frame Questions as If You Don't Know the Answer:
   - Write questions that are general and applicable to a wide range of scenarios.
   - Do not directly reference the transcript content or assume the viewer already knows the transcript details.
{%- if self_contained_question %}
   - Do not mention the existence of a context/transcript in the generated question like "in the transcript", "from the given context", or "in Segment 148". Produce a natural, standalone question.
   - Only use facts present in the provided context/transcript; if missing, say you cannot generate a question.
{%- endif %}
   - Example: "What are some emerging trends in the field of artificial intelligence?"

2. Ensure Questions are Contextually Relevant:
   - Questions should be about broad topics that could be answered by a wide range of transcripts.

3. Diversity in Question Types:
   - Include a mix of factual questions (e.g., "What are the primary benefits of cloud computing?"),
     why/how questions (e.g., "Why has AI adoption grown in recent years?"),
     and procedural questions (e.g., "How did the IT manager explain the process for requesting new software?").

{%- if reasoning_counts %}
4. Reasoning Types to Include:
   - Factual questions ({{reasoning_counts.get("factual", 0)}} questions): Ask for specific, verifiable facts unique to this document
   - Relational questions ({{reasoning_counts.get("relational", 0)}} questions): Ask how data points compare or correlate
   - Inferential questions ({{reasoning_counts.get("inferential", 0)}} questions): Ask about conclusions or implications
   - Temporal questions ({{reasoning_counts.get("temporal", 0)}} questions): Ask about changes or events over time
   - Procedural questions ({{reasoning_counts.get("procedural", 0)}} questions): Ask about steps or guidelines
   - Visual questions ({{reasoning_counts.get("visual", 0)}} questions): Ask about unique visual details
   - Causal questions ({{reasoning_counts.get("causal", 0)}} questions): Ask about causes, reasons, or contributing factors

   Example questions by reasoning type:
   - Factual: "How many paid time off days are employees eligible for per year?"
   - Relational: "How did sales revenue compare to expenses in Q3 2023?"
   - Inferential: "What does the report suggest about future AI investment trends?"
   - Temporal: "What was the change in machinery value between 2018 and 2019?"
   - Procedural: "What are the steps for submitting a travel reimbursement?"
   - Visual: "Which report includes the Department of Agriculture logo?"
   - Causal: "Why did patient recovery times improve after the new protocol?"
{%- endif %}

5. Generate {{num_pairs}} distinct question and answer pairs:
   - Each pair must be unique and have varying complexity levels between 1 (simple) and 5 (complex).
   - For each pair, identify the start and end time (in HH:MM:SS format) that best supports the answer.

6. **IMPORTANT - Segment Identification**:
   - The transcript below contains segments formatted as "Segment N (HH:MM:SS - HH:MM:SS): text" where N starts from 1
   - For each question-answer pair you generate, identify ALL segment numbers FROM which the question is derived
   - These segments contain the source material that inspired the question and provide the answer
   - Record these segment numbers in the "segment_ids" field as a list of integers (e.g., [1, 3, 6])
   - Think of it as: "If someone asks this question in a retrieval system, these are the segments that should be retrieved"
   - The timestamps (start_time and end_time) will be automatically calculated from the segments you identify
   - If a question requires information from multiple segments, include all relevant segment numbers

The output should be a JSON array of objects (with the number of objects equal to {{num_pairs}}), where each object contains:
  - "question": the generated question without explicitly referencing the context/transcript/segments in the question.
  - "answer": the answer without explicitly referencing the context/transcript/segments in the answer.
  - "question_complexity": a numeric score from 1 (simple) to 5 (complex).
  - "segment_ids": list of segment numbers (e.g., [1, 3, 6]) that are the source material for this question-answer pair.
  - "start_time": the HH:MM:SS timestamp (will be calculated from earliest segment).
  - "end_time": the HH:MM:SS timestamp (will be calculated from latest segment).

Additional Constraints:
- **Output only the JSON. Do not include any introductory or concluding text.**
- **Please ensure that your entire response is strictly valid JSON with no additional text or formatting.**
- **The output should be a JSON array containing exactly {{num_pairs}} objects.**
- **Pay careful attention to the timestamp markers in the transcript to ensure accurate start_time and end_time values.**

Video Style Guidance: {{video_style}}

{{summary_block}}

Transcript with Timestamps:
---
{{segment_text}}

Output a JSON array where each element is an object with the following keys:
  - "question"
  - "answer"
  - "question_complexity"
  - "segment_ids"
  - "start_time"
  - "end_time"

Please ensure your output is ONLY the JSON object with no preamble or additional text.
"""
generate_question_answer_with_timestamps_template = Template(
    generate_question_answer_with_timestamps_prompt)


def generate_question_answer_with_timestamps(
    segment_text: str,
    video_style: str,
    num_pairs: int,
    client: LLMClient,
    summary: List[Dict[str, str]] = None,
    reasoning_type_distribution: Dict[str, float] = None,
    top_artifacts: List[Dict[str, Any]] = None,
    artifacts_context: str = None,
    self_contained_question: bool = False,
) -> List[dict]:
    summary_block = get_summary_block(summary or [])
    # Add artifact context if provided
    artifact_block = get_artifact_block(artifacts_context)

    # Calculate reasoning type distribution
    reasoning_counts = {}
    if reasoning_type_distribution:
        remaining = num_pairs
        for reasoning_type, percentage in reasoning_type_distribution.items():
            count = int(num_pairs * percentage)
            reasoning_counts[reasoning_type] = count
            remaining -= count

        if remaining > 0:
            # Add remainder to most common type
            reasoning_counts['factual'] = (reasoning_counts.get('factual', 0) +
                                           remaining)

    prompt = generate_question_answer_with_timestamps_template.render(
        artifact_block=artifact_block,
        num_pairs=num_pairs,
        reasoning_counts=reasoning_counts,
        video_style=video_style,
        summary_block=summary_block,
        segment_text=segment_text,
        self_contained_question=self_contained_question,
    )
    logger.debug('final_prompt: %s', prompt)
    messages = [
        {
            'role': 'system',
            'content': 'QA generation expert.'
        },
        {
            'role': 'user',
            'content': prompt
        },
    ]
    resp = client.generate(messages, temperature=0.3)
    return extract_json_from_response(resp)


def extract_time_ranges_from_structured_text(
    structured_text: str, ) -> List[Dict[str, Any]]:
    """Parse structured segment text to retrieve timestamp ranges."""
    ranges: List[Dict[str, Any]] = []
    for line in structured_text.splitlines():
        match = _SEGMENT_LINE_PATTERN.match(line.strip())
        if not match:
            continue
        _, start_ts, end_ts, _ = match.groups()
        try:
            start_seconds = parse_timestamp(start_ts)
            end_seconds = parse_timestamp(end_ts)
        except Exception:
            start_seconds = end_seconds = None
        ranges.append({
            'start': start_ts,
            'end': end_ts,
            'start_seconds': start_seconds,
            'end_seconds': end_seconds,
            'source': 'local',
        })
    return ranges


def process_segments_info(qa_pair: Dict[str, Any],
                          segments: List[Dict[str, Any]]) -> None:
    """Process segment_ids in a QA pair and calculate start_time and end_time.
    Also processes hop_contexts if present, and ensures top-level segment_ids
    is the union of all hop-level segment_ids.

    Note: segment_ids are 1-based (as shown to LLM), but converted to 0-based for array indexing.

    Args:
        qa_pair: The QA pair dict that will be modified in-place
        segments: List of all segments with 'start', 'end', and 'text' fields
    """
    # First, process hop_contexts to collect hop-level segment_ids
    hop_segment_ids_union = set()
    if 'hop_contexts' in qa_pair and qa_pair['hop_contexts']:
        for hop in qa_pair['hop_contexts']:
            if 'segment_ids' in hop and hop['segment_ids']:
                hop_segment_ids = hop['segment_ids']
                hop_segment_ids_union.update(hop_segment_ids)

                # Get segments for this hop (convert 1-based to 0-based index)
                hop_segments = []
                for seg_id in hop_segment_ids:
                    idx = seg_id - 1  # Convert 1-based to 0-based
                    if 0 <= idx < len(segments):
                        hop_segments.append(segments[idx])

                if hop_segments:
                    # Calculate hop time span
                    hop_starts = [seg['start'] for seg in hop_segments]
                    hop_ends = [seg['end'] for seg in hop_segments]

                    hop['start_time'] = format_timestamp(min(hop_starts))
                    hop['end_time'] = format_timestamp(max(hop_ends))

                    # Collect context text for this hop
                    hop['context_segments'] = [
                        seg['text'] for seg in hop_segments
                    ]

    # Validate and correct top-level segment_ids
    top_level_segment_ids = set(qa_pair.get('segment_ids', []))

    # If we have hop contexts, ensure top-level is at least the union
    if hop_segment_ids_union:
        # Use the larger set (union of both)
        final_segment_ids = sorted(
            top_level_segment_ids.union(hop_segment_ids_union))
        qa_pair['segment_ids'] = final_segment_ids
    elif top_level_segment_ids:
        # No hop contexts, just use top-level as-is
        final_segment_ids = sorted(top_level_segment_ids)
        qa_pair['segment_ids'] = final_segment_ids
    else:
        # No segment_ids at all
        final_segment_ids = []

    # Process main segment_ids
    if final_segment_ids:
        # Get all relevant segments (convert 1-based to 0-based index)
        relevant_segments = []
        for seg_id in final_segment_ids:
            idx = seg_id - 1  # Convert 1-based to 0-based
            if 0 <= idx < len(segments):
                relevant_segments.append(segments[idx])

        if relevant_segments:
            # Calculate overall time span
            all_starts = [seg['start'] for seg in relevant_segments]
            all_ends = [seg['end'] for seg in relevant_segments]

            start_sec = min(all_starts)
            end_sec = max(all_ends)

            # Format timestamps
            qa_pair['start_time'] = format_timestamp(start_sec)
            qa_pair['end_time'] = format_timestamp(end_sec)

            # Collect context text from all segments
            qa_pair['context_segments'] = [
                seg['text'] for seg in relevant_segments
            ]


async def generate_qa_pairs(state: QAGenerationState) -> Dict[str, Any]:
    """Generate QA pairs using selected model."""
    client = state.get('client')

    hard_mode = state.get('hard', False)
    logger.debug('hard mode: %s', hard_mode)

    query_type_distribution = json.loads(
        state.get(
            'query_type_distribution',
            '{"multi_hop":0.4,"structural":0.3,"contextual":0.3}',
        ))
    reasoning_type_distribution = state.get('reasoning_type_distribution',
                                            None)
    if reasoning_type_distribution:
        reasoning_type_distribution = json.loads(reasoning_type_distribution)
    min_complexity = state.get('min_complexity', 4)
    min_hops = state.get('min_hops', 2)
    max_hops = state.get('max_hops', 4)

    num_pairs = state.get('num_pairs', 10)  # Default to 10 if not specified
    summary = state.get('summary', [])
    max_retries = 3
    all_qa_pairs = []
    cross_part_contexts = []

    model = client.generation_model_name
    logger.info('Generating QA pairs using %s', model)

    try:
        artifacts_str_splits = state.get('top_artifacts', None)
        segments = state.get('segments', [])
        logger.debug('==> artifacts_str_splits: %s', artifacts_str_splits)

        if not segments:
            logger.warning('No segments found for task %s', state['task_id'])
            return {
                'qa_pairs': [],
                'status': TaskStatus.FAILED.value,
                'error_messages': ['No segments in transcription'],
            }

        # Split segments into parts for diverse sampling
        parts = state.get('parts', 3)
        logger.info('Split with timestamp into %d parts', parts)
        splits = split_segments_text_structured(segments, parts=parts)
        timeline_events: List[Dict[str, Any]] = []
        hard_mode = state.get('hard', False)
        logger.debug('hard mode: %s', hard_mode)
        if hard_mode:
            try:
                timeline_events = generate_structured_timeline(
                    segments, client=client)
                logger.info(
                    f'[{state["task_id"]}] Timeline events extracted: {len(timeline_events)}'
                )
            except Exception as e:
                logger.warning(
                    f'[{state["task_id"]}] Timeline generation failed: {e}')
                traceback.print_exc()

        if len(timeline_events) > 0:
            cross_part_contexts = build_cross_part_contexts_with_timeline(
                splits, timeline_events)
        else:
            cross_part_contexts = build_cross_part_contexts_from_segments(
                splits)

        # Calculate pairs per split
        pps = max(1, num_pairs // max(1, len(splits)))

        logger.info('Num splits: %d, pps: %d', len(splits), pps)

        # Generate QA pairs for each split
        all_qa_pairs = []

        # Infer video style from the transcript
        sample_text = transcription_segment_text(
            segments[:10])  # Use first 10 segments for inference
        video_style = infer_video_style(sample_text, client)
        logger.info('Inferred video style: %s', video_style)

        if hard_mode:
            min_complexity = state.get('min_complexity', 4)
            query_type_distribution = json.loads(
                state.get(
                    'query_type_distribution',
                    '{"multi_hop":0.4,"structural":0.3,"contextual":0.3}',
                ))
            reasoning_type_distribution_str = state.get(
                'reasoning_type_distribution', None)
            if reasoning_type_distribution_str:
                reasoning_type_distribution = json.loads(
                    reasoning_type_distribution_str)
            min_hops = state.get('min_hops', 2)
            max_hops = state.get('max_hops', 4)

        split_idx = 0
        for idx, seg_text in enumerate(splits):
            cross_context_info = (cross_part_contexts[idx]
                                  if idx < len(cross_part_contexts) else {
                                      'text': '',
                                      'ranges': [],
                                      'metadata': {}
                                  })
            cross_context = cross_context_info.get('text', '')
            cross_ranges = cross_context_info.get('ranges', [])
            context_meta = cross_context_info.get('metadata', {})
            if context_meta.get('sections_truncated'):
                logger.debug(
                    f'[{state["task_id"]}] Cross-part context for split {idx + 1} truncated sections: {context_meta["sections_truncated"]}'
                )
            if (context_meta.get('source') == 'timeline'
                    and context_meta.get('external_events', 0) == 0):
                logger.debug(
                    f'[{state["task_id"]}] Timeline context for split {idx + 1} had no external events'
                )
            local_ranges = extract_time_ranges_from_structured_text(seg_text)
            logger.info(
                f'Processing split {split_idx + 1}/{len(splits)}: len of seg_text: {len(seg_text)}, len of summary: {len(summary)}'
            )
            # Use the actual generate_question_answer function from helper
            retry_count = 0
            split_idx += 1
            top_artifacts = (artifacts_str_splits[idx]
                             if artifacts_str_splits else None)

            while retry_count < max_retries:
                try:
                    if hard_mode:
                        qa = generate_hard_question_answer_with_timestamps(  # FIXED: Use timestamp version
                            segment_text=seg_text,
                            video_style=video_style,
                            num_pairs=pps,
                            client=client,
                            summary=summary,
                            query_type_distribution=query_type_distribution,
                            reasoning_type_distribution=
                            reasoning_type_distribution,
                            min_complexity=min_complexity,
                            min_hops=min_hops,
                            max_hops=max_hops,
                            artifacts_context=top_artifacts,
                            cross_part_context=cross_context,
                            allowed_ranges_local=local_ranges,
                            allowed_ranges_cross=cross_ranges,
                            self_contained_question=state.get(
                                'self_contained_question', False),
                        )
                    else:
                        qa = generate_question_answer_with_timestamps(  # FIXED: Use timestamp version
                            segment_text=seg_text,
                            video_style=video_style,
                            num_pairs=pps,
                            client=client,
                            summary=summary,
                            reasoning_type_distribution=
                            reasoning_type_distribution,
                            artifacts_context=top_artifacts,
                            self_contained_question=state.get(
                                'self_contained_question', False),
                        )

                    # Process segment_ids to calculate timestamps
                    for q in qa:
                        process_segments_info(q, segments)
                        q['split'] = idx
                    all_qa_pairs.extend(qa)
                    break

                except Exception as e:
                    retry_count += 1
                    logger.warning(
                        f'QA generation failed (attempt {retry_count}/{max_retries}): {e}'
                    )
                    if retry_count >= max_retries:
                        raise
                    await asyncio.sleep(1)  # Brief delay before retry

            # Break if we've generated enough pairs
            if len(all_qa_pairs) >= num_pairs:
                break

        # Trim to requested number of pairs
        all_qa_pairs = all_qa_pairs[:num_pairs]

        # Add metadata to each QA pair
        for qa_pair in all_qa_pairs:
            qa_pair['task_id'] = state['task_id']
            qa_pair['model_used'] = model
            qa_pair['timestamp'] = datetime.now().isoformat()

        # Add metadata about the generation process
        generation_metadata = {
            'total_segments': len(segments),
            'splits_processed': len(splits),
            'qa_pairs_generated': len(all_qa_pairs),
            'model_used': model,
            'generation_time': datetime.now().isoformat(),
        }

        logger.info('Generated %d QA pairs', len(all_qa_pairs))

        return {
            'qa_pairs': all_qa_pairs,
            'status': TaskStatus.VALIDATING.value,
            'metrics': generation_metadata,
            'cross_part_contexts': cross_part_contexts,
        }

    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error('Error generating QA pairs: %s, %s', e, error_msg)
        return {
            'error_messages': state.get('error_messages', []) + [str(e)],
            'status': TaskStatus.FAILED.value,
        }


def get_embeddings(
    texts: List[str],
    client: LLMClient,
    input_type: Optional[str] = 'passage',
    batch_size: int = 20,
):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embed(
            prompts=batch,
            input_type=input_type,
        )
        embs.extend(resp)
    return embs


def save_index_and_metadata(embs, chunks, prefix, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)
    faiss.write_index(idx, os.path.join(out_dir, f'{prefix}_index.faiss'))
    with open(os.path.join(out_dir, f'{prefix}_metadata.pkl'), 'wb') as f:
        pickle.dump(chunks, f)


def validate_answer_spans_hybrid(
    qa_pairs: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    base_name: str,
    output_dir: str,
    client: LLMClient,
    sim_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """Hybrid span validation:
    1) Embedding‐based filter (cosine ≥ sim_threshold) via your FAISS index.
    2) Literal substring search to anchor exact segment start/end.
    """
    valid = []

    # 1) Load FAISS index & metadata
    edir = Path(output_dir) / 'embedding_data'
    idx_path = edir / f'{base_name}_index.faiss'
    meta_path = edir / f'{base_name}_metadata.pkl'

    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        logger.info(
            f'Creating {base_name} index.faiss and metadata.pkl in {edir}')
        texts = [s['text'] for s in segments]
        embs = get_embeddings(texts, client=client, input_type='passage')
        embs = np.array(embs, dtype='float32')
        save_index_and_metadata(embs, segments, base_name, edir)
        logger.info('index.faiss and metadata.pkl are created')

    index = faiss.read_index(str(idx_path))
    metadata = pickle.load(open(meta_path,
                                'rb'))  # list of {"start":ms,"end":ms,...}

    for qa in qa_pairs:
        ans = qa['answer'].strip()
        if not ans:
            continue

        # --- Stage 1: embedding check ---
        resp = client.embed(
            prompts=[ans],
            input_type='query',
        )
        q_emb = np.array(resp, dtype=np.float32)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, 1)
        if float(D[0][0]) < sim_threshold:
            continue  # drop if too dissimilar

        # at this point we have an index hit
        hit_idx = int(I[0][0])
        hit_meta = metadata[hit_idx]  # dict with ms timestamps

        # --- Stage 2: substring anchoring ---
        found_anchor = False
        for seg in segments:
            text = seg.get('text', '')
            if ans.lower() in text.lower():
                # anchor here
                st = seg['start']
                en = seg['end']
                qa['start_time'] = (
                    f'{int(st // 3600):02d}:{int((st % 3600) // 60):02d}:{int(st % 60):02d}'
                )
                qa['end_time'] = (
                    f'{int(en // 3600):02d}:{int((en % 3600) // 60):02d}:{int(en % 60):02d}'
                )
                found_anchor = True
                break

        if not found_anchor:
            # fallback: use FAISS‑nearest segment
            st = hit_meta['start']
            en = hit_meta['end']
            qa['start_time'] = (
                f'{int(st // 3600):02d}:{int((st % 3600) // 60):02d}:{int(st % 60):02d}'
            )
            qa['end_time'] = (
                f'{int(en // 3600):02d}:{int((en % 3600) // 60):02d}:{int(en % 60):02d}'
            )

        valid.append(qa)

    logger.info('total qa pairs: %d, valid pairs: %d', len(qa_pairs),
                len(valid))
    return valid


def deduplicate_qa(
    qa_pairs: List[Dict[str, Any]],
    client: LLMClient,
    threshold: float = 0.9,
) -> List[Dict[str, Any]]:
    """Removes near‑duplicate questions based on embedding cosine similarity.

    Args:
      qa_pairs: list of dicts with at least a "question" key.
      client: LLMClient instance
      threshold: cosine similarity above which two questions are considered duplicates.

    Returns:
      Filtered list of qa_pairs containing only the first instance of each cluster.
    """
    # 1) Extract questions
    questions = [qa['question'] for qa in qa_pairs]
    if not questions:
        return []

    # 2) Embed all questions in one batch
    # api_key = os.getenv("NVIDIA_API_KEY")
    # client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    resp = client.embed(prompts=questions, input_type='query')
    # 3) Build embedding matrix and L2‑normalize
    embs = np.array(resp, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / (norms + 1e-8)

    # 4) Compute pairwise cosine similarities
    sims = cosine_similarity(embs)

    # 5) Greedily keep first of each duplicate cluster
    keep = []
    seen = set()
    for i, qa in enumerate(qa_pairs):
        if i in seen:
            continue
        keep.append(qa)
        # mark all above‑threshold as seen (including self)
        duplicates = np.where(sims[i] >= threshold)[0].tolist()
        seen.update(duplicates)

    return keep


def convert_posixpath_to_string(data):
    if isinstance(data, dict):
        # Process dictionary recursively
        return {
            key: convert_posixpath_to_string(value)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        # Process list recursively
        return [convert_posixpath_to_string(item) for item in data]
    elif isinstance(data, Path):
        # Convert PosixPath or Path to string
        return str(data)
    else:
        # Return the value as-is
        return data


async def validate_qa_pairs(state: QAGenerationState) -> Dict[str, Any]:
    """Validate generated QA pairs using hybrid validation."""
    logger.info('Validating QA pairs for task %s', state['task_id'])

    qa_pairs = state.get('qa_pairs', [])
    segments = state.get('segments', [])
    client = state.get('client')

    dedup_threshold = state.get('dedup_threshold', 0.6)

    # Extract base_name and output_dir from file path
    file_path = Path(state['file_path'])
    base_name = file_path.stem.replace('_transcription', '')
    output_dir = state['output_dir']

    try:
        # First, enforce time order
        # qa_pairs = enforce_time_order(qa_pairs)

        # Validate answer spans using hybrid validation
        validated_pairs = validate_answer_spans_hybrid(
            qa_pairs=qa_pairs,
            segments=segments,
            base_name=base_name,
            output_dir=output_dir,
            client=client,
            sim_threshold=dedup_threshold,
        )

        logger.info('validated_pairs: %d', len(validated_pairs))

        # Deduplicate QA pairs
        validated_pairs = deduplicate_qa(
            qa_pairs=validated_pairs,
            threshold=0.9,  # Higher threshold for deduplication
            client=client,
        )
        logger.info('After dedup: validated_pairs: %d', len(validated_pairs))

        validation_results = {
            'total_pairs': len(qa_pairs),
            'validated_pairs': len(validated_pairs),
            'dropped_pairs': len(qa_pairs) - len(validated_pairs),
            'validation_method': 'hybrid_with_faiss',
        }

        logger.info(
            f'Validated {len(validated_pairs)}/{len(qa_pairs)} QA pairs')

        return {
            'validated_pairs':
            validated_pairs,
            'validation_results':
            validation_results,
            'status':
            TaskStatus.PROCESSING.value
            if validated_pairs else TaskStatus.FAILED.value,
        }

    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error('validate_qa_pairs: %s, %s', e, error_msg)
        return {
            'error_messages': state.get('error_messages', []) + [str(e)],
            'status': TaskStatus.FAILED.value,
            'validation_results': {
                'error': str(e)
            },
        }


async def mine_hard_negative_segments(
    state: QAGenerationState, ) -> Dict[str, Any]:
    """Mine hard negative segments for each QA pair using embedding similarity."""
    logger.info('Mining hard negative segments for task %s', state['task_id'])

    if not state.get('enable_hard_negatives', False):
        logger.info(
            'Hard negative mining disabled via configuration; skipping.')
        qa_pairs = state.get('validated_pairs', [])
        for qa_pair in qa_pairs:
            qa_pair['hard_negative_segment_ids'] = []
            qa_pair['hard_negatives_metadata'] = []
        return {
            'validated_pairs': qa_pairs,
            'hard_negatives_stats': {
                'total_hard_negatives': 0,
                'avg_per_question': 0.0,
                'enabled': False,
            },
        }

    qa_pairs = state.get('validated_pairs', [])
    segments = state.get('segments', [])

    # Configuration
    top_k = state.get('hard_negatives_top_k', 10)
    min_similarity = state.get('hard_negatives_min_sim', 0.5)
    max_similarity = state.get('hard_negatives_max_sim', 0.7)

    # Skip if no segments (like in image-only mode)
    if not segments or len(segments) == 0:
        logger.info(
            'No segments available, skipping hard negative segment mining')
        for qa_pair in qa_pairs:
            qa_pair['hard_negative_segment_ids'] = []
            qa_pair['hard_negatives_metadata'] = []
        return {
            'validated_pairs': qa_pairs,
            'hard_negatives_stats': {
                'total_hard_negatives': 0,
                'avg_per_question': 0.0,
                'enabled': state.get('enable_hard_negatives', True),
            },
        }

    texts = [s.get('text', '') for s in segments]
    texts = [t for t in texts if t.strip()]  # Filter empty texts

    if not texts:
        logger.warning(
            'No text content in segments, skipping hard negative segment mining'
        )
        for qa_pair in qa_pairs:
            qa_pair['hard_negative_segment_ids'] = []
            qa_pair['hard_negatives_metadata'] = []
        return {
            'validated_pairs': qa_pairs,
            'hard_negatives_stats': {
                'total_hard_negatives': 0,
                'avg_per_question': 0.0,
                'enabled': state.get('enable_hard_negatives', True),
            },
        }

    try:
        # Try to load existing FAISS index from validation step
        file_path = Path(state['file_path'])
        base_name = file_path.stem.replace('_transcription', '')
        output_dir = state['output_dir']
        edir = Path(output_dir) / 'embedding_data'
        idx_path = edir / f'{base_name}_index.faiss'

        os.getenv('NVIDIA_API_KEY')

        if idx_path.exists():
            logger.info('Loading existing FAISS index from %s', idx_path)
            index = faiss.read_index(str(idx_path))

            embs = index.reconstruct_n(0, index.ntotal)

            logger.info(
                f'Loaded {len(embs)} normalized embeddings from FAISS index')
        else:
            logger.info(
                'FAISS index not found, computing embeddings for all segments')

            embs = get_embeddings(texts, client=state.client,
                                  input_type='passage')
            embs = np.array(embs, dtype='float32')

            faiss.normalize_L2(embs)

            dimension = embs.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embs)

        min_distance = np.sqrt(
            2 * (1 - max_similarity))  # max similarity = min distance
        max_distance = np.sqrt(
            2 * (1 - min_similarity))  # min similarity = max distance

        for qa_pair in qa_pairs:
            positive_segment_ids = set(qa_pair.get('segment_ids', []))

            if not positive_segment_ids:
                qa_pair['hard_negative_segment_ids'] = []
                qa_pair['hard_negatives_metadata'] = []
                continue

            try:
                first_positive_id = list(positive_segment_ids)[0]
                first_positive_idx = first_positive_id - 1

                if not (0 <= first_positive_idx < len(embs)):
                    qa_pair['hard_negative_segment_ids'] = []
                    qa_pair['hard_negatives_metadata'] = []
                    continue

                # Use the first positive segment's embedding
                positive_emb = embs[first_positive_idx:first_positive_idx +
                                    1]  # Keep 2D shape

                # Search for top K + buffer similar segments
                search_k = min(top_k + len(positive_segment_ids) + 20,
                               len(segments))
                D, I = index.search(positive_emb, search_k)

                # Filter hard negative segments
                hard_negatives = []

                for distance, seg_idx in zip(D[0], I[0]):
                    segment_id = int(seg_idx) + 1  # Convert to 1-based

                    if segment_id in positive_segment_ids:
                        continue

                    if min_distance <= distance <= max_distance:
                        cosine_sim = 1 - (distance**2) / 2
                        hard_negatives.append({
                            'segment_id': segment_id,
                            'similarity': float(cosine_sim),
                        })

                    # Stop when we have enough hard negatives
                    if len(hard_negatives) >= top_k:
                        break

                # Store results
                qa_pair['hard_negative_segment_ids'] = [
                    hn['segment_id'] for hn in hard_negatives
                ]
                qa_pair['hard_negatives_metadata'] = hard_negatives

                logger.debug(
                    f'Found {len(hard_negatives)} hard negatives for question: {qa_pair["question"][:50]}...'
                )

            except Exception as e:
                logger.warning('Error mining hard negatives for question: %s',
                               e)
                qa_pair['hard_negative_segment_ids'] = []
                qa_pair['hard_negatives_metadata'] = []

        # Summary statistics
        total_hard_negs = sum(
            len(qa.get('hard_negative_segment_ids', [])) for qa in qa_pairs)
        avg_hard_negs = total_hard_negs / len(qa_pairs) if qa_pairs else 0

        logger.info(
            f'Hard negative mining complete. Avg {avg_hard_negs:.1f} hard negatives per QA pair'
        )

        return {
            'validated_pairs': qa_pairs,
            'hard_negatives_stats': {
                'total_hard_negatives': total_hard_negs,
                'avg_per_question': avg_hard_negs,
                'enabled': True,
            },
        }

    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error('Error in hard negative mining: %s, %s', e, error_msg)
        # Don't fail the pipeline, just skip hard negative mining
        for qa_pair in qa_pairs:
            qa_pair['hard_negative_segment_ids'] = []
            qa_pair['hard_negatives_metadata'] = []
        return {
            'validated_pairs': qa_pairs,
            'hard_negatives_stats': {
                'total_hard_negatives': 0,
                'avg_per_question': 0.0,
                'enabled': state.get('enable_hard_negatives', True),
            },
        }


generate_negative_answers_prompt = """
Given the following question from a transcript, generate {{num_negatives}} plausible but incorrect answers.

QUESTION: {{question}}
CORRECT ANSWER: {{correct_answer}}

Guidelines:
1. Create answers that are factually incorrect but sound reasonable
2. Make answers similar in structure and length to the correct answer
3. Ensure negative answers are clearly wrong when compared to the correct answer
4. Vary the types of incorrectness (wrong facts, wrong reasoning, wrong conclusions)

Output as a JSON array of strings containing exactly {{num_negatives}} negative answers.
Example: ["negative answer 1", "negative answer 2", "negative answer 3"]

Do not include any other text, only the JSON array.
"""
generate_negative_answers_prompt_template = Template(
    generate_negative_answers_prompt)


def generate_negative_answers(
    question: str,
    correct_answer: str,
    segment_text: str,
    num_negatives: int,
    client: LLMClient,
) -> List[str]:
    prompt = generate_negative_answers_prompt_template.render(
        question=question,
        correct_answer=correct_answer,
        segment_text=segment_text,
        num_negatives=num_negatives,
    )

    logger.debug('final_prompt: %s', prompt)

    messages = [
        {
            'role': 'system',
            'content': 'Expert distractor creator.'
        },
        {
            'role': 'user',
            'content': prompt
        },
    ]
    try:
        resp = client.generate(messages, temperature=0.3)
        return json.loads(re.search(r'\[.*?\]', resp, re.DOTALL).group(0))
    except Exception as e:
        logger.error(
            f'Error in generating negative answers: {e}. Response_text: {resp}'
        )
        logger.info('Try again')
        try:
            # Remove only from the very beginning and very end
            cleaned = re.sub(r'^```json\s*', '', resp)  # Remove from start
            cleaned = re.sub(r'\s*```$', '', cleaned)  # Remove from end
            return json.loads(cleaned)
        except Exception as e1:
            logger.error('Error again in generating negative answers: %s', e1)
    return []


async def generate_negative_answers_node(
    state: QAGenerationState, ) -> Dict[str, Any]:
    """Generate negative answers for multiple choice questions."""
    logger.info('Generating negative answers for task %s', state['task_id'])

    qa_pairs = state.get('qa_pairs', [])
    validated_pairs = state.get('validated_pairs', [])
    segments = state.get('segments', [])
    client = state.get('client')
    num_negatives = state.get('num_negatives', 3)
    if num_negatives == 0:
        logger.info(
            f'num_negatives is 0. Negative answers will not be generated')
        return {'qa_pairs': qa_pairs}

    logger.info(
        f'Generating negative answers using model: {client.generation_model_name}'
    )

    # Get full transcript text
    segment_text = transcription_segment_text(segments)

    # Add negative answers to each QA pair
    for qa_pair in validated_pairs:
        try:
            # Use the actual generate_negative_answers function from helper
            negative_answers = generate_negative_answers(
                question=qa_pair['question'],
                correct_answer=qa_pair['answer'],
                segment_text=segment_text[:
                                          1000],  # Limit context for efficiency
                num_negatives=num_negatives,
                client=client,
            )
            qa_pair['negative_answers'] = negative_answers
        except Exception as e:
            logger.warning('Error generating negative answers: %s', e)
            qa_pair['negative_answers'] = []

    return {'validated_pairs': validated_pairs}


llm_evaluate_qa_pair_prompt = """
You are an expert evaluator of question-answer pairs.

Please evaluate the following question and answer based on the provided context:

CONTEXT:
```
{{context}}
```

{{cross_part_context_block}}

QUESTION: {{question}}

ANSWER: {{answer}}

Evaluate this QA pair on the following criteria (score 1-10):
1. Relevance: Is the question relevant to the main topics in the context?
2. Accuracy: Is the answer factually correct according to the context?
3. Context Support: Is the answer well-supported by the context?
4. Clarity: Is the question clear and unambiguous?

For each criterion, provide:
- Score (1-10)
- Brief justification

Then provide an overall score (1-10) and final assessment.

Output your evaluation as a valid JSON object with the following structure:
```json
{
  "relevance": { "score": X, "justification": "..." },
  "accuracy": { "score": X, "justification": "..." },
  "context_support": { "score": X, "justification": "..." },
  "clarity": { "score": X, "justification": "..." },
  "overall": { "score": X, "assessment": "..." },
  "improvements": "Suggestions for improving this QA pair"
}

Please ensure your output is ONLY the JSON object with no preamble or additional text.
"""
llm_evaluate_qa_pair_prompt_template = Template(llm_evaluate_qa_pair_prompt)


async def llm_evaluate_qa_pair(
    question: str,
    answer: str,
    context: str,
    cross_part_context_block: str,
    client: LLMClient,
) -> Dict[str, Any]:
    """Use an LLM to evaluate the quality of a QA pair.

    Args:
        question: The generated question
        answer: The generated answer
        context: The context from which the QA was generated
        cross_part_context_block: The cross-part context block
        client: LLMClient

    Returns:
        Dictionary with evaluation scores and feedback
    """
    prompt = llm_evaluate_qa_pair_prompt_template.render(
        context=context,
        question=question,
        answer=answer,
        cross_part_context_block=cross_part_context_block,
    )
    logger.debug('final_prompt: %s', prompt)

    messages = [
        {
            'role':
            'system',
            'content':
            'You are an expert QA evaluator that provides structured JSON evaluations.',
        },
        {
            'role': 'user',
            'content': prompt
        },
    ]

    logger.debug('llm_evaluate_qa_pair using %s', client.evaluation_model_name)
    try:
        response_text = client.evaluate(messages, temperature=0)
        return json.loads(response_text)
    except Exception as e:
        logger.error('Error in QA evaluation: %s.', e)
        logger.info('Try again')
        try:
            # Remove only from the very beginning and very end
            cleaned = re.sub(r'^```json\s*', '',
                             response_text)  # Remove from start
            cleaned = re.sub(r'\s*```$', '', cleaned)  # Remove from end
            return json.loads(cleaned)
        except Exception as e1:
            logger.error(
                f'Error again in QA evaluation: {e1}. Response_text: {response_text}'
            )

        # Return a default evaluation on error
        return {
            'relevance': {
                'score': 0,
                'justification': 'Evaluation failed'
            },
            'accuracy': {
                'score': 0,
                'justification': 'Evaluation failed'
            },
            'context_support': {
                'score': 0,
                'justification': 'Evaluation failed',
            },
            'clarity': {
                'score': 0,
                'justification': 'Evaluation failed'
            },
            'overall': {
                'score': 0,
                'assessment': f'Evaluation error: {str(e)}',
            },
            'improvements': 'N/A due to evaluation error',
        }


# Conditional edge functions
async def evaluate_qa_pairs(state: QAGenerationState) -> Dict[str, Any]:
    """Evaluate QA pairs using an LLM judge."""
    logger.info('Evaluating QA pairs for task %s', state['task_id'])

    qa_pairs = state.get('qa_pairs', [])
    validated_pairs = state.get('validated_pairs', [])
    client = state.get('client')
    evaluation_obj = state.get('evaluation', {})
    question_only = state.get('question_only', False)
    enable = evaluation_obj['enable'] if 'enable' in evaluation_obj else True
    logger.debug(
        f'Enable Evaluating QA pairs? {enable}, question_only? {question_only}'
    )
    if question_only or not enable:
        logger.info(
            f'Skip evaluation because of question_only is {question_only} or enable_evaluation is {enable} '
        )
        return {'qa_pairs': qa_pairs, 'evaluation_metrics': {}}

    # Sample at most 5 QA pairs for evaluation if there are many
    sample_size = evaluation_obj.get('sample_size', 5)

    logger.info(
        f'Evaluating QA pairs using {client.evaluation_model_name}, sample size: {sample_size}, validated_pairs: {len(validated_pairs)}'
    )

    eval_sample = (validated_pairs if len(validated_pairs) <= sample_size else
                   random.sample(validated_pairs, sample_size))
    logger.info(
        f'Evaluating QA pairs: eval sample size: {len(eval_sample)}, validated_pairs: {len(validated_pairs)}'
    )

    overall_scores = []
    evaluated_pairs = []

    cross_part_contexts = state['cross_part_contexts']
    for idx, qa_pair in enumerate(eval_sample):
        logger.debug('Evaluating QA pair %d/%d', idx + 1, len(eval_sample))

        question = qa_pair['question']
        answer = qa_pair['answer']
        split_idx = qa_pair['split'] if 'split' in qa_pair else -1
        if split_idx != -1:
            cross_context_info = (cross_part_contexts[split_idx] if split_idx
                                  < len(cross_part_contexts) else {
                                      'text': '',
                                      'ranges': [],
                                      'metadata': {}
                                  })
            cross_context = cross_context_info.get('text', '')
        else:
            cross_context = ''

        cross_part_context_block = get_cross_part_context_block(cross_context)

        # Get context for this QA pair
        context = ''
        try:
            start_time = qa_pair.get('start_time', '00:00:00')
            end_time = qa_pair.get('end_time', '00:00:00')
            segments = state.get('segments', [])

            relevant_segments = []
            for segment in segments:
                seg_start = segment.get('start', 0)
                seg_end = segment.get('end', 0)

                # Check if segment overlaps with QA time range
                if seg_start <= parse_timestamp(
                        end_time) and seg_end >= parse_timestamp(start_time):
                    relevant_segments.append(segment.get('text', ''))

            context = ' '.join(relevant_segments)

        except Exception as e:
            logger.warning('Error extracting context for evaluation: %s', e)
            error_msg = traceback.format_exc()
            logger.error('Validation error: %s, %s', e, error_msg)
            segments = state.get('segments', [])

            context = transcription_segment_text(
                segments)[:1000]  # Use first 1000 chars as fallback

        # Evaluate this QA pair
        try:
            evaluation_result = await llm_evaluate_qa_pair(
                question=question,
                answer=answer,
                context=context,
                cross_part_context_block=cross_part_context_block,
                client=client,
            )

            # Store evaluation in the QA pair
            qa_pair['evaluation'] = evaluation_result
            evaluated_pairs.append(qa_pair)

            # Track overall scores
            overall_scores.append(
                evaluation_result.get('overall', {}).get('score', 0))

        except Exception as e:
            logger.warning('Failed to evaluate QA pair: %s', e)
            qa_pair['evaluation'] = {
                'overall': {
                    'score': 0,
                    'assessment': f'Evaluation failed: {str(e)}',
                }
            }

    # Update metrics with evaluation results
    avg_score = sum(overall_scores) / max(len(overall_scores), 1)
    evaluation_metrics = {
        'average_quality_score':
        avg_score,
        'evaluated_pairs':
        len(evaluated_pairs),
        'evaluation_model':
        client.evaluation_model_name,
        'high_quality_ratio':
        sum(1 for s in overall_scores if s >= 7) / max(len(overall_scores), 1),
    }

    # Add evaluations back to original QA pairs
    evaluated_ids = {
        qa['question']: qa['evaluation']
        for qa in evaluated_pairs
    }
    for qa_pair in qa_pairs:
        if qa_pair['question'] in evaluated_ids:
            qa_pair['evaluation'] = evaluated_ids[qa_pair['question']]
        else:
            qa_pair['evaluation'] = {
                'relevance': {
                    'score': -99,
                    'justification': 'Not evaluated'
                },
                'accuracy': {
                    'score': -99,
                    'justification': 'Not evaluated'
                },
                'context_support': {
                    'score': -99,
                    'justification': 'Not evaluated',
                },
                'clarity': {
                    'score': -99,
                    'justification': 'Not evaluated'
                },
                'overall': {
                    'score': -99,
                    'assessment': 'Not evaluated'
                },
                'improvements': 'N/A',
            }

    logger.info(
        f'Evaluation complete. num qa_pairs: {len(qa_pairs)}, num evaluated_pairs: {len(evaluated_pairs)}, Average score: {avg_score:.2f}/10'
    )

    return {'qa_pairs': qa_pairs, 'evaluation_metrics': evaluation_metrics}


async def filter_by_quality(state: QAGenerationState) -> Dict[str, Any]:
    """Filter QA pairs based on quality score from evaluation."""
    logger.info('Filtering QA pairs by quality for task %s', state['task_id'])

    qa_pairs = state.get('qa_pairs', [])
    validated_pairs = state.get('validated_pairs', [])

    logger.debug(
        f'Filtering QA pairs by quality for validated_pairs: {len(validated_pairs)} qa_pairs, total: {len(qa_pairs)}'
    )

    quality_threshold = state.get('evaluation',
                                  {}).get('quality_threshold', 6.0)
    logger.debug('quality_threshold: %s', quality_threshold)

    # Count pairs with quality scores
    high_quality_pairs = []
    low_quality_pairs = []
    not_evaluated_pairs = []

    for qa_pair in validated_pairs:
        evaluation = qa_pair.get('evaluation', {})
        quality_score = evaluation.get('overall', {}).get('score', 0)

        if quality_score >= quality_threshold:
            high_quality_pairs.append(qa_pair)
        elif quality_score == -99:
            not_evaluated_pairs.append(qa_pair)
        else:
            low_quality_pairs.append(qa_pair)

    evaluated_pairs = high_quality_pairs + low_quality_pairs
    num_evaluated = len(evaluated_pairs)

    # Log quality filtering results
    logger.info(
        f'{state["file_path"]}: Total qa_pairs: {len(qa_pairs)}, validated_pairs: {len(validated_pairs)}, not_evaluated_pairs: {len(not_evaluated_pairs)}, high_quality_pairs: {len(high_quality_pairs)}, low_quality_pairs: {len(low_quality_pairs)}, evaluated_pairs: {num_evaluated}'
    )
    logger.info(
        f'Quality filtering: {len(high_quality_pairs)}/{num_evaluated} pairs above threshold {quality_threshold}'
    )

    # If we have evaluations but all pairs are below threshold,
    # either lower the threshold or keep the best ones
    if num_evaluated > 0 and len(high_quality_pairs) == 0:
        logger.warning(
            f'No QA pairs meet quality threshold {quality_threshold}.')

        # Option 1: Sort by quality and keep top 30%
        if validated_pairs:
            sorted_pairs = sorted(
                validated_pairs,
                key=lambda x: x.get('evaluation', {}).get('overall', {}).get(
                    'score', 0),
                reverse=True,
            )
            top_n = max(1, int(len(sorted_pairs) *
                               0.3))  # Keep at least 1, up to 30%
            high_quality_pairs = sorted_pairs[:top_n]
            logger.info(
                f'Keeping top {len(high_quality_pairs)} pairs by quality score.'
            )

    sum_overall_score = sum([
        qa.get('evaluation', {}).get('overall', {}).get('score', 0)
        for qa in evaluated_pairs
    ])

    logger.debug('sum_overall_score: %s', sum_overall_score)
    quality_metrics = {
        'total_pairs': len(qa_pairs),
        'evaluated_pairs': len(evaluated_pairs),
        'high_quality_pairs': len(high_quality_pairs),
        'low_quality_pairs': len(low_quality_pairs),
        'not_evaluated_pairs': len(not_evaluated_pairs),
        'quality_threshold': quality_threshold,
        'average_quality': sum_overall_score / max(1, num_evaluated),
    }

    return {
        'high_quality_qa_pairs': high_quality_pairs,
        'low_quality_qa_pairs': low_quality_pairs,
        'not_evaluated_pairs': not_evaluated_pairs,
        'quality_metrics': quality_metrics,
    }


async def monitor_progress(state: QAGenerationState) -> Dict[str, Any]:
    """Monitor and log progress metrics."""
    logger.debug('Monitoring progress for task %s', state['task_id'])

    # Basic metrics
    base_metrics = {
        'task_id':
        state['task_id'],
        'qa_pairs_generated':
        len(state.get('qa_pairs', [])),
        'validation_rate':
        state.get('validation_results', {}).get('validated_pairs', 0) /
        max(state.get('validation_results', {}).get('total_pairs', 1), 1),
        'model_used':
        state.get('model_selection', 'unknown'),
        'retry_count':
        state.get('retry_count', 0),
        'status':
        state.get('status', 'unknown'),
        'timestamp':
        datetime.now().isoformat(),
    }

    # Add evaluation metrics if available
    evaluation_obj = state.get('evaluation', {})
    question_only = state.get('question_only', False)
    enable = evaluation_obj['enable'] if 'enable' in evaluation_obj else True
    logger.debug(
        f'Enable Evaluating QA pairs? {enable}, question_only? {question_only}'
    )
    if question_only or not enable:
        logger.info(
            f'Skip evaluation because of question_only is {question_only} or enable_evaluation is {enable} '
        )
        return {'metrics': None, 'status': TaskStatus.PROCESSING.value}

    evaluation_metrics = state.get('evaluation_metrics', {})
    quality_metrics = state.get('quality_metrics', {})

    # Add per-category average scores if we have evaluated QA pairs
    state.get('qa_pairs', [])
    high_quality_qa_pairs = state.get('high_quality_qa_pairs', [])
    low_quality_qa_pairs = state.get('low_quality_qa_pairs', [])
    evaluated_pairs = high_quality_qa_pairs + low_quality_qa_pairs

    if evaluated_pairs:
        # Calculate average scores for each category
        avg_scores = {
            'relevance_avg':
            sum(qa['evaluation'].get('relevance', {}).get('score', 0)
                for qa in evaluated_pairs) / len(evaluated_pairs),
            'accuracy_avg':
            sum(qa['evaluation'].get('accuracy', {}).get('score', 0)
                for qa in evaluated_pairs) / len(evaluated_pairs),
            'context_support_avg':
            sum(qa['evaluation'].get('context_support', {}).get('score', 0)
                for qa in evaluated_pairs) / len(evaluated_pairs),
            'clarity_avg':
            sum(qa['evaluation'].get('clarity', {}).get('score', 0)
                for qa in evaluated_pairs) / len(evaluated_pairs),
            'overall_avg':
            sum(qa['evaluation'].get('overall', {}).get('score', 0)
                for qa in evaluated_pairs) / len(evaluated_pairs),
        }

        # Add detailed scoring to evaluation metrics
        evaluation_metrics = {
            **evaluation_metrics,
            'detailed_scores': avg_scores,
        }

        # Log the detailed scores
        logger.info(f'Avg scores: rel={avg_scores["relevance_avg"]:.2f}, ' +
                    f'acc={avg_scores["accuracy_avg"]:.2f}, ' +
                    f'ctx={avg_scores["context_support_avg"]:.2f}, ' +
                    f'clarity={avg_scores["clarity_avg"]:.2f}, ' +
                    f'overall={avg_scores["overall_avg"]:.2f}')
    else:
        if evaluation_metrics:
            logger.info(
                f'Evaluation metrics: avg_score={evaluation_metrics.get("average_quality_score", 0):.2f}, '
                +
                f'evaluated_pairs={evaluation_metrics.get("evaluated_pairs", 0)}'
            )

    if quality_metrics:
        logger.info(
            f'Quality metrics: high_quality={quality_metrics.get("high_quality_pairs", 0)}/{quality_metrics.get("total_pairs", 0)}, '
            + f'threshold={quality_metrics.get("quality_threshold", 0)}')

    # Combine metrics
    combined_metrics = {
        **base_metrics,
        'evaluation': evaluation_metrics,
        'quality_filtering': quality_metrics,
    }

    state['metrics'] = combined_metrics

    return {'metrics': combined_metrics, 'status': TaskStatus.PROCESSING.value}


async def store_results(state: QAGenerationState) -> Dict[str, Any]:
    """Store results to CSV and JSON files."""
    state['task_id']
    logger.info('Storing results for task %s', state['task_id'])

    if state.get('skip_store'):
        logger.info(
            f'skip_store flag set for task {state["task_id"]}; skipping persistence'
        )
        return {'status': state.get('status', TaskStatus.PROCESSING.value)}

    output_dir = Path(state['output_dir'])
    output_dir.mkdir(exist_ok=True)

    try:
        # Prepare data for storage
        high_quality_qa_pairs = state.get('high_quality_qa_pairs', [])
        low_quality_qa_pairs = state.get('low_quality_qa_pairs', [])
        not_evaluated_pairs = state.get('not_evaluated_pairs', [])
        qa_pairs = state.get('qa_pairs', [])

        evaluated_pairs = high_quality_qa_pairs + low_quality_qa_pairs
        qa_pairs_to_write = high_quality_qa_pairs + not_evaluated_pairs

        segments = state.get('segments', [])

        # Generate a mapping of timestamp to segment text
        segment_map = {}
        for segment in segments:
            start_sec = segment.get('start', 0)
            end_sec = segment.get('end', 0)
            segment_map[(start_sec, end_sec)] = segment.get('text', '')

        logger.info(
            f'Num qa_pair to write: {len(qa_pairs_to_write)}, total_qa_pairs = {len(qa_pairs)}'
        )
        qa_data = []

        for qa_pair in qa_pairs_to_write:
            # Get context from relevant segments based on segment_ids
            context = ''
            try:
                # Check if we have context_segments from processing
                if 'context_segments' in qa_pair:
                    context = ' '.join(qa_pair['context_segments'])
                elif 'segment_ids' in qa_pair and qa_pair['segment_ids']:
                    # Fallback to extracting based on segment_ids
                    segment_ids = qa_pair['segment_ids']
                    relevant_segments = []
                    for seg_id in segment_ids:
                        if 0 <= seg_id < len(segments):
                            relevant_segments.append(segments[seg_id].get(
                                'text', ''))
                    context = ' '.join(relevant_segments)
                else:
                    # Old fallback based on timestamps
                    start_time = qa_pair.get('start_time', '00:00:00')
                    end_time = qa_pair.get('end_time', '00:00:00')
                    start_sec = parse_timestamp(start_time)
                    end_sec = parse_timestamp(end_time)

                    # Find all segments that overlap with this time range
                    relevant_segments = []
                    for (seg_start, seg_end), text in segment_map.items():
                        # Check for overlap
                        if seg_start <= end_sec and seg_end >= start_sec:
                            relevant_segments.append(text)

                    context = ' '.join(relevant_segments)
            except Exception as e:
                logger.warning('Error extracting context: %s', e)
                # Fallback to using the first 500 chars of all segments
                context = transcription_segment_text(segments)[:500]

            # Get evaluation data if available
            evaluation = qa_pair.get('evaluation', {})

            # Extract all individual scores
            relevance_score = evaluation.get('relevance', {}).get('score', 0)
            relevance_justification = evaluation.get('relevance', {}).get(
                'justification', '')

            accuracy_score = evaluation.get('accuracy', {}).get('score', 0)
            accuracy_justification = evaluation.get('accuracy', {}).get(
                'justification', '')

            context_support_score = evaluation.get('context_support',
                                                   {}).get('score', 0)
            context_support_justification = evaluation.get(
                'context_support', {}).get('justification', '')

            clarity_score = evaluation.get('clarity', {}).get('score', 0)
            clarity_justification = evaluation.get('clarity', {}).get(
                'justification', '')

            overall_score = evaluation.get('overall', {}).get('score', 0)
            assessment = evaluation.get('overall',
                                        {}).get('assessment', 'Not evaluated')
            improvements = evaluation.get('improvements', '')
            image_id = 'Not applicable'
            start_time = 'Not applicable'
            end_time = 'Not applicalbe'

            row = {
                'task_id':
                state['task_id'],
                'file_path':
                str(state['file_path']),
                'image_id':
                image_id,
                'question':
                qa_pair['question'],
                'answer':
                qa_pair.get('answer', 'N/A'),
                'query_type':
                qa_pair.get('query_type', 'unspecified'),
                'reasoning_type':
                qa_pair.get('reasoning_type', 'unspecified'),
                'hop_count':
                qa_pair['hop_count'] if 'hop_count' in qa_pair else -1,
                'hop_contexts':
                qa_pair['hop_contexts']
                if 'hop_contexts' in qa_pair else 'N/A',
                'context':
                context[:500],  # Limit context length
                'context_segments':
                qa_pair.get('context_segments',
                            []),  # Full context text from each segment
                'segment_ids':
                qa_pair.get('segment_ids',
                            []),  # Add segment IDs for traceability
                'hard_negative_segment_ids':
                qa_pair.get('hard_negative_segment_ids', []),
                'hard_negatives_metadata':
                qa_pair.get('hard_negatives_metadata', []),
                'start_time':
                start_time,
                'end_time':
                end_time,
                'question_complexity':
                qa_pair.get('question_complexity', 1),
                'model_used':
                qa_pair.get('model_used', state.get('model_selection', '')),
                'negative_answers':
                qa_pair.get('negative_answers', []),
                'timestamp':
                qa_pair.get('timestamp',
                            datetime.now().isoformat()),
                # Overall evaluation
                'quality_score':
                overall_score,
                'assessment':
                assessment,
                'improvements':
                improvements,
                # Individual evaluation scores
                'relevance_score':
                relevance_score,
                'accuracy_score':
                accuracy_score,
                'context_support_score':
                context_support_score,
                'clarity_score':
                clarity_score,
                # Justifications (optional - can remove if too verbose)
                'relevance_justification':
                relevance_justification,
                'accuracy_justification':
                accuracy_justification,
                'context_support_justification':
                context_support_justification,
                'clarity_justification':
                clarity_justification,
            }
            qa_data.append(row)

        logger.info('final qa_data: %d', len(qa_data))

        # check if evaluation is off
        evaluation_obj = state.get('evaluation', {})
        question_only = state.get('question_only', False)
        enable = (evaluation_obj['enable']
                  if 'enable' in evaluation_obj else True)
        logger.debug(
            f'Enable Evaluating QA pairs? {enable}, question_only? {question_only}'
        )
        if question_only or not enable:
            logger.info(
                f'Skip evaluation because of question_only is {question_only} or enable_evaluation is {enable} '
            )
        else:
            evaluation_metrics = state.get('evaluation_metrics', {})
            # Log evaluation summary
            if evaluation_metrics:
                avg_scores = {
                    'relevance':
                    sum(
                        qa.get('evaluation', {}).get('relevance', {}).get(
                            'score', 0)
                        for qa in evaluated_pairs if 'evaluation' in qa) /
                    max(
                        sum(1 for qa in evaluated_pairs if 'evaluation' in qa),
                        1,
                    ),
                    'accuracy':
                    sum(
                        qa.get('evaluation', {}).get('accuracy', {}).get(
                            'score', 0)
                        for qa in evaluated_pairs if 'evaluation' in qa) /
                    max(
                        sum(1 for qa in evaluated_pairs if 'evaluation' in qa),
                        1,
                    ),
                    'context_support':
                    sum(
                        qa.get('evaluation', {}).get('context_support',
                                                     {}).get('score', 0)
                        for qa in evaluated_pairs if 'evaluation' in qa) /
                    max(
                        sum(1 for qa in evaluated_pairs if 'evaluation' in qa),
                        1,
                    ),
                    'clarity':
                    sum(
                        qa.get('evaluation', {}).get('clarity', {}).get(
                            'score', 0)
                        for qa in evaluated_pairs if 'evaluation' in qa) /
                    max(
                        sum(1 for qa in evaluated_pairs if 'evaluation' in qa),
                        1,
                    ),
                    'overall':
                    evaluation_metrics.get('average_quality_score', 0),
                }

                logger.info(
                    f'Avg scores: relevance={avg_scores["relevance"]:.2f}, ' +
                    f'accuracy={avg_scores["accuracy"]:.2f}, ' +
                    f'context={avg_scores["context_support"]:.2f}, ' +
                    f'clarity={avg_scores["clarity"]:.2f}, ' +
                    f'overall={avg_scores["overall"]:.2f}, ' +
                    f'status={TaskStatus.COMPLETED.value}')

        logger.info('Complete!')
        return {
            'qa_data_to_write': qa_data,
            'status': TaskStatus.COMPLETED.value,
        }

    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error('Error storing results: %s, %s', e, error_msg)
        return {
            'error_messages': state.get('error_messages', []) + [str(e)],
            'status': TaskStatus.FAILED.value,
        }


async def feedback_controller(state: QAGenerationState) -> Dict[str, Any]:
    """Handle feedback and retry logic."""
    logger.debug('Processing feedback for task %s', state['task_id'])

    client = state['client']
    model_progression = state.model_progression
    if state.get('status') == TaskStatus.FAILED.value:
        retry_count = state.get('retry_count', 0)

        if retry_count < 3:
            # Model upgrade strategy based on retry count

            # First retry - move up one model level
            current_model = client.generation_model_name
            idx = model_progression.find(current_model)

            if idx < len(model_progression) - 1:
                next_model = model_progression[idx + 1]
                client.update_generation_model(next_model)

            logger.info(
                f'Retry {retry_count + 1}/3: Upgrading model to {client.generation_model_name.get("model_selection", "same")}'
            )

            return {
                'retry_count': retry_count + 1,
                'status': TaskStatus.RETRYING.value,
                'feedback': {
                    'action': 'retry',
                    'reason': state.get('error_messages',
                                        ['Unknown error'])[-1],
                    'strategy': 'model_upgrade',
                    'model': client.generation_model_name,
                },
            }
        else:
            return {
                'status': TaskStatus.FAILED.value,
                'feedback': {
                    'action': 'final_failure',
                    'reason': 'Max retries exceeded',
                },
            }

    return {'feedback': {'action': 'continue'}}


async def end(state: QAGenerationState) -> Dict[str, Any]:
    return {'status': state['status']}


def compute_metrics(
    records: List[dict],
    query_type_distribution: Optional[str] = None,
    reasoning_type_distribution: Optional[str] = None,
) -> Dict[str, Any]:
    df = pd.DataFrame(records)
    if df.empty:
        return {}

    metrics = {
        'average_question_length':
        df['question'].str.len().mean(),
        'average_answer_length':
        df['answer'].str.len().mean(),
        'complexity_distribution':
        df['question_complexity'].value_counts().to_dict(),
        'total_pairs':
        len(df),
    }

    # Extract valid values from the config distributions
    valid_query_types = set()
    valid_reasoning_types = set()

    # Parse query type distribution if provided
    if query_type_distribution:
        try:
            query_dist = json.loads(query_type_distribution)
            valid_query_types = set(query_dist.keys())
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                'Failed to parse query_type_distribution: %s',
                query_type_distribution,
            )
            # Fall back to defaults if parsing fails
            valid_query_types = {'multi_hop', 'structural', 'contextual'}

    # Parse reasoning type distribution if provided
    if reasoning_type_distribution:
        try:
            reasoning_dist = json.loads(reasoning_type_distribution)
            valid_reasoning_types = set(reasoning_dist.keys())
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                'Failed to parse reasoning_type_distribution: %s',
                reasoning_type_distribution,
            )
            # Fall back to defaults if parsing fails
            valid_reasoning_types = {
                'factual',
                'relational',
                'inferential',
                'temporal',
                'procedural',
                'visual',
                'causal',
            }

    # Add metrics specific to hard questions if available
    if 'query_type' in df.columns and valid_query_types:
        # Filter to only valid query types
        valid_query_df = df[df['query_type'].isin(valid_query_types)]
        metrics['query_type_distribution'] = (
            valid_query_df['query_type'].value_counts().to_dict())

        # Check for misclassified entries
        invalid_query_types = df[~df['query_type'].isin(valid_query_types)][
            'query_type'].unique()
        if len(invalid_query_types) > 0:
            logger.warning('Found invalid query_type values: %s',
                           invalid_query_types)
            metrics['invalid_query_types'] = list(invalid_query_types)

    if 'reasoning_type' in df.columns and valid_reasoning_types:
        # Filter to only valid reasoning types
        valid_reasoning_df = df[df['reasoning_type'].isin(
            valid_reasoning_types)]
        metrics['reasoning_type_distribution'] = (
            valid_reasoning_df['reasoning_type'].value_counts().to_dict())

        # Check for misclassified entries
        invalid_reasoning_types = df[~df['reasoning_type'].isin(
            valid_reasoning_types)]['reasoning_type'].unique()
        if len(invalid_reasoning_types) > 0:
            logger.warning(
                'Found invalid reasoning_type values: %s',
                invalid_reasoning_types,
            )
            metrics['invalid_reasoning_types'] = list(invalid_reasoning_types)

    if 'is_hard' in df.columns:
        metrics['hard_queries_count'] = df['is_hard'].sum()
        metrics['hard_queries_percentage'] = ((df['is_hard'].sum() / len(df)) *
                                              100 if len(df) > 0 else 0)

    # Add multi-hop metrics if available
    if 'hop_count' in df.columns:
        multihop_df = df[df['query_type'] == 'multi_hop']
        if len(multihop_df) > 0:
            metrics['multi_hop_metrics'] = {
                'average_hop_count':
                multihop_df['hop_count'].mean(),
                'hop_count_distribution':
                multihop_df['hop_count'].value_counts().to_dict(),
            }

    return metrics


# Add evaluation function
def route_after_validation(state: QAGenerationState) -> str:
    """Route after validation based on results."""
    qa_pairs = state.get('qa_pairs', [])

    if not qa_pairs or len(qa_pairs) == 0:
        logger.warning('No QA pairs passed validation, routing to feedback')
        return 'feedback'
    else:
        logger.info(
            f'{len(qa_pairs)} QA pairs passed validation, continuing to mine hard negatives'
        )
        return 'continue'


def should_retry(state: QAGenerationState) -> str:
    """Determine if task should be retried."""
    if state.get('status') == TaskStatus.RETRYING.value:
        return 'retry'
    elif state.get('status') == TaskStatus.FAILED.value:
        return 'feedback_node'
    else:
        return 'continue'
