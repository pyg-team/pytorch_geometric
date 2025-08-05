"""Data warehouse utilities for PyTorch Geometric with G-Retriever integration.

Provides warehouse graph analysis using G-Retriever for multi-task learning
on lineage, silo, and quality prediction tasks.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import GATConv
from torch_geometric.typing import OptTensor

# Constants
EMBEDDING_DIM = 384
HIDDEN_DIM = 256
NUM_LINEAGE_TYPES = 5
NUM_SILO_TYPES = 2
NUM_ANOMALY_TYPES = 2
DROPOUT = 0.1
DEFAULT_MAX_TOKENS = 32
SILO_THRESHOLD = 0.5
SILO_CRITICAL_RATIO = 0.5
SILO_MODERATE_RATIO = 0.2
QUALITY_THRESHOLD = 0.5
EXCELLENCE_THRESHOLD = 0.8
IMPACT_HIGH_RATIO = 0.3
IMPACT_MEDIUM_RATIO = 0.1
QUALITY_ATTENTION_RATIO = 0.2

# Response templates for different tasks
RESPONSE_TEMPLATES = {
    "lineage": {
        "predicted": ("Model Lineage Analysis of {num_nodes} entities:\n"
                      "• Predicted lineage: {prediction}\n"
                      "• Distribution: {distribution}\n"
                      "• Average confidence: {confidence:.3f}")
    },
    "silo": {
        "predicted": ("Model Silo Analysis of {num_nodes} entities:\n"
                      "• Predicted isolated silos: {isolated_count}\n"
                      "• Connected entities: {connected_count}\n"
                      "• Predicted silo ratio: {silo_ratio:.2f}%\n"
                      "• Average silo confidence: {confidence:.3f}")
    },
    "quality": {
        "predicted": ("Data quality assessment for {num_nodes} entities:\n"
                      "• Average quality score: {avg_score:.3f}\n"
                      "• Poor quality entities: {poor_count}\n"
                      "• High quality entities: {high_count}\n"
                      "• Overall status: {status}")
    },
    "impact": {
        "predicted": ("Impact analysis across {num_nodes} entities:\n"
                      "• High impact entities: {high_impact}\n"
                      "• Impact distribution: Low({low}), "
                      "Medium({medium}), High({high})\n"
                      "• Risk assessment: {risk_level}")
    }
}

# Set up logging
logger = logging.getLogger(__name__)


class GATWrapper(nn.Module):
    """GAT wrapper with proper interface for GRetriever compatibility."""
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        # Use actual GATConv - it's not abstract, mypy error is false positive
        self.gat = GATConv(in_channels, out_channels // heads,
                           heads=heads)  # type: ignore[abstract]
        self.out_channels = out_channels

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: OptTensor = None) -> Tensor:
        """Forward pass using graph attention."""
        return self.gat(x, edge_index)


# PyG imports with fallbacks
try:
    from torch_geometric.nn.models import GRetriever
    from torch_geometric.nn.nlp import LLM
    HAS_GRETRIEVER = True
except ImportError:
    logger.warning("G-Retriever not available, using fallback")
    HAS_GRETRIEVER = False

    class GRetriever:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("GRetriever requires PyG with LLM support")

    class LLM:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("LLM requires PyG with LLM support")


class WarehouseGRetriever(nn.Module):
    """Warehouse analysis using G-Retriever architecture."""
    def __init__(self,
                 llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v0.1",
                 gnn_hidden_channels: int = HIDDEN_DIM,
                 gnn_num_layers: int = 2, gnn_heads: int = 4,
                 dropout: float = DROPOUT):
        super().__init__()

        if not HAS_GRETRIEVER:
            raise ImportError("G-Retriever not available")

        # GNN for graph encoding (output must match LLM embedding dimension)
        llm_embedding_dim = 2048  # TinyLlama embedding dimension
        self.gnn = GATWrapper(EMBEDDING_DIM, llm_embedding_dim,
                              heads=gnn_heads)

        # LLM for text generation (following PyG standards)
        self.llm = LLM(
            model_name=llm_model_name,
            num_params=1,  # TinyLlama is ~1B parameters
            dtype=torch.float16,  # Standard for PyG LLM tests
        )

        # G-Retriever combining GNN + LLM (match LLM embedding dimension)
        self.g_retriever = GRetriever(
            llm=self.llm,
            gnn=self.gnn,
            mlp_out_channels=llm_embedding_dim  # Match LLM embedding dimension
        )

        # Task-specific heads (input dimension matches GNN output)
        self.task_head = WarehouseTaskHead(hidden_dim=llm_embedding_dim,
                                           dropout=dropout)

    def forward(self, question: list[str], x: Tensor, edge_index: Tensor,
                batch: Tensor, label: list[str], **kwargs: Any) -> Tensor:
        """Training forward pass following G-Retriever pattern."""
        return self.g_retriever(question=question, x=x, edge_index=edge_index,
                                batch=batch, label=label, **kwargs)

    def inference(self, question: list[str], x: Tensor, edge_index: Tensor,
                  batch: Tensor, max_out_tokens: int = DEFAULT_MAX_TOKENS,
                  **kwargs: Any) -> list[str]:
        """Enhanced two-stage inference for business-focused responses."""
        try:
            # Stage 1: Generate domain context
            domain_context = self._generate_domain_context(x, edge_index)

            # Stage 2: Generate GNN insights
            gnn_insights = self._generate_gnn_insights(x, edge_index)

            # Stage 3: Combine context with user question for business response
            enhanced_questions = []
            for q in question:
                # Simplified prompt format for TinyLlama
                enhanced_prompt = (f"Warehouse Analysis: {domain_context}. "
                                   f"{gnn_insights}. Question: {q} Answer:")
                enhanced_questions.append(enhanced_prompt)

            # Use enhanced prompts with G-Retriever
            return self.g_retriever.inference(question=enhanced_questions, x=x,
                                              edge_index=edge_index,
                                              batch=batch,
                                              max_out_tokens=max_out_tokens,
                                              **kwargs)
        except Exception:
            # Fallback to original inference
            return self.g_retriever.inference(question=question, x=x,
                                              edge_index=edge_index,
                                              batch=batch,
                                              max_out_tokens=max_out_tokens,
                                              **kwargs)

    def predict_task(self, x: Tensor, edge_index: Tensor, task: str) -> Tensor:
        """Predict specific warehouse task."""
        # Encode graph with GNN
        node_emb = self.gnn(x, edge_index)

        # Global pooling for graph-level prediction
        graph_emb = torch.mean(node_emb, dim=0, keepdim=True)

        # Task-specific prediction
        return self.task_head(graph_emb, task)

    def _generate_domain_context(self, x: Tensor, edge_index: Tensor) -> str:
        """Generate domain-specific context for warehouse intelligence."""
        try:
            num_nodes = x.size(0)
            num_edges = edge_index.size(1)

            # Analyze graph structure
            avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

            if avg_degree < 1.5:
                connectivity = "sparse data connections"
            elif avg_degree < 3.0:
                connectivity = "moderate data integration"
            else:
                connectivity = "highly integrated data ecosystem"

            return (f"a data warehouse with {connectivity} across "
                    f"{num_nodes} data entities")
        except Exception:
            return "a structured data warehouse with multiple data sources"

    def _generate_gnn_insights(self, x: Tensor, edge_index: Tensor) -> str:
        """Generate GNN-based insights for warehouse analysis."""
        try:
            # Get graph embeddings
            node_emb = self.gnn(x, edge_index)

            # Analyze embedding patterns for business insights
            emb_std = torch.std(node_emb).item()
            torch.mean(node_emb).item()

            if emb_std > 0.5:
                pattern = ("diverse data patterns indicating "
                           "multiple data domains")
            elif emb_std > 0.2:
                pattern = ("moderate data variation suggesting "
                           "some specialization")
            else:
                pattern = ("consistent data patterns indicating "
                           "unified structure")

            return (f"Graph analysis reveals {pattern} with "
                    f"embedding variance {emb_std:.3f}")
        except Exception:
            return "Graph analysis indicates standard warehouse data patterns"


class WarehouseTaskHead(nn.Module):
    """Multi-task head for warehouse intelligence operations.

    Supports various warehouse tasks including lineage prediction,
    impact analysis, and data quality assessment.

    Args:
        hidden_dim: Hidden dimension size for task heads
        num_lineage_types: Number of lineage categories
        num_impact_levels: Number of impact severity levels
        dropout: Dropout probability for regularization
    """
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_lineage_types: int = NUM_LINEAGE_TYPES,
        num_impact_levels: int = 3,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Create task heads with factory method
        self.heads = nn.ModuleDict({
            'lineage':
            self._create_head(hidden_dim, num_lineage_types, dropout),
            'impact':
            self._create_head(hidden_dim, num_impact_levels, dropout),
            'quality':
            self._create_head(hidden_dim, 1, dropout, sigmoid=True),
            'silo':
            self._create_head(hidden_dim, 1, dropout, sigmoid=True),
        })

    def _create_head(self, in_dim: int, out_dim: int, dropout: float,
                     sigmoid: bool = False) -> nn.Module:
        """Create a task-specific head.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            dropout: Dropout probability
            sigmoid: Whether to apply sigmoid activation

        Returns:
            Task-specific neural network head
        """
        layers = [
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, out_dim)
        ]

        if sigmoid:
            layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, task: str = "lineage") -> Tensor:
        """Forward pass for specific task.

        Args:
            x: Input node embeddings [num_nodes, hidden_dim]
            task: Task type ('lineage', 'impact', 'quality', 'silo')

        Returns:
            Task-specific predictions

        Raises:
            ValueError: If task is not supported
        """
        if task not in self.heads:
            raise ValueError(f"Unsupported task: {task}. "
                             f"Available tasks: {list(self.heads.keys())}")

        return self.heads[task](x)


class WarehouseConversationSystem:
    """Conversation system for warehouse intelligence queries.

    Provides natural language interface for warehouse analysis tasks
    including lineage tracing, silo detection, and quality assessment.

    Args:
        model: Warehouse model for predictions
        device: Computing device ('cpu' or 'cuda')
    """

    # Response templates for different query types
    RESPONSE_TEMPLATES = {
        'lineage': {
            'with_labels':
            "RelBench Lineage Analysis of {num_nodes} entities:\n"
            "• Real lineage pattern: {lineage_type}\n"
            "• Actual distribution: {distribution}\n"
            "• Data source: RelBench warehouse labels",
            'predicted':
            "Model Lineage Analysis of {num_nodes} entities:\n"
            "• Predicted lineage: {lineage_type}\n"
            "• Distribution: {distribution}\n"
            "• Average confidence: {confidence:.3f}"
        },
        'silo': {
            'with_labels':
            "RelBench Silo Analysis of {num_nodes} entities:\n"
            "• Isolated silos detected: {silo_count}\n"
            "• Connected entities: {connected_count}\n"
            "• Silo ratio: {silo_ratio:.2%}\n"
            "• Status: {status}\n"
            "• Data source: RelBench silo labels",
            'predicted':
            "Model Silo Analysis of {num_nodes} entities:\n"
            "• Predicted isolated silos: {silo_count}\n"
            "• Connected entities: {connected_count}\n"
            "• Predicted silo ratio: {silo_ratio:.2%}\n"
            "• Average silo confidence: {confidence:.3f}"
        },
        'impact': {
            'predicted':
            "Impact analysis across {num_nodes} entities:\n"
            "• High impact entities: {high_impact}\n"
            "• Impact: Low({low}), Medium({medium}), High({high})\n"
            "• Risk assessment: {risk_level}"
        },
        'quality': {
            'predicted':
            "Data quality assessment for {num_nodes} entities:\n"
            "• Average quality score: {avg_quality:.3f}\n"
            "• Poor quality entities: {poor_quality}\n"
            "• High quality entities: {high_quality}\n"
            "• Overall status: {status}"
        }
    }

    def __init__(self, model: WarehouseGRetriever | Any,
                 device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self.conversation_history: list[dict[str, str]] = []

        # Query classification keywords
        self.query_keywords = {
            "lineage": ["lineage", "source", "origin", "flow"],
            "silo": ["silo", "isolated", "disconnect", "separate"],
            "impact": ["impact", "downstream", "depend", "affect"],
            "quality": ["quality", "reliable", "issue", "problem"]
        }

    def classify_query(self, query: str) -> str:
        """Classify query type based on keywords.

        Args:
            query: Natural language query

        Returns:
            Query type ('lineage', 'silo', 'impact', 'quality', or 'general')
        """
        query_lower = query.lower()

        for query_type, keywords in self.query_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type

        return "general"

    def process_query(self, query: str, graph_data: dict[str, Any],
                      context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Process a warehouse intelligence query.

        Args:
            query: Natural language query
            graph_data: Graph data dictionary with x, edge_index, etc.
            context: Optional context with labels and metadata

        Returns:
            Dictionary with answer and metadata

        Raises:
            ValueError: If required graph data is missing
            RuntimeError: If model prediction fails
        """
        if 'x' not in graph_data or 'edge_index' not in graph_data:
            raise ValueError("Graph data must contain 'x' and 'edge_index'")

        try:
            # Classify query type
            query_type = self.classify_query(query)

            # Get model predictions
            x = graph_data['x']
            edge_index = graph_data['edge_index']
            batch = graph_data.get('batch')

            # Get integrated LLM + Analytics response with proper context
            if hasattr(self.model, 'g_retriever'):
                # Ensure batch is properly formatted
                if batch is None:
                    batch = torch.zeros(x.size(0), dtype=torch.long)

                # Create context-aware prompt
                contextual_prompt = self._create_contextual_prompt(
                    query, x, edge_index, query_type, context)

                # Use full G-Retriever integration (LLM + GNN) with context
                llm_response = self.model.inference(
                    question=[contextual_prompt], x=x, edge_index=edge_index,
                    batch=batch, max_out_tokens=150)

                # Also get analytics predictions for structured data
                pred = self.model.predict_task(
                    x=x, edge_index=edge_index,
                    task=query_type if query_type != "general" else "lineage")

                # Combine LLM response with analytics
                llm_text = llm_response[0] if llm_response else ""
            else:
                # Fallback to analytics only
                pred = self.model.predict_task(
                    x=x, edge_index=edge_index,
                    task=query_type if query_type != "general" else "lineage")
                llm_text = ""
            num_nodes = x.size(0)

            # Check for real labels
            has_labels = (context is not None
                          and context.get('warehouse_labels') is not None)

            # Extract data based on query type
            if query_type == "lineage":
                data = self._extract_lineage_data(
                    pred, num_nodes, context if has_labels else None)
            elif query_type == "silo":
                data = self._extract_silo_data(pred, num_nodes,
                                               context if has_labels else None)
            elif query_type == "impact":
                data = self._extract_impact_data(pred, num_nodes)
            elif query_type == "quality":
                data = self._extract_quality_data(pred, num_nodes)
            else:
                data = {
                    "num_nodes": num_nodes,
                    "confidence": pred.mean().item()
                }

            # Create integrated response combining LLM + Analytics
            if llm_text and len(llm_text.strip()) > 10:
                # Use LLM response as primary answer with analytics support
                answer = self._create_integrated_response(
                    llm_text, data, query_type, has_labels)
            else:
                # Fallback to template-based response
                template_key = 'with_labels' if has_labels else 'predicted'
                template = self.RESPONSE_TEMPLATES.get(query_type, {}).get(
                    template_key,
                    "Analyzed {num_nodes} entities (conf {confidence:.3f})")
                answer = template.format(**data)

            # Store in conversation history
            import datetime
            self.conversation_history.append({
                'query':
                query,
                'answer':
                answer,
                'query_type':
                query_type,
                'has_labels':
                str(has_labels),
                'timestamp':
                datetime.datetime.now().isoformat()
            })

            return {
                'answer': answer,
                'query_type': query_type,
                'data': data,
                'predictions': pred,
                'has_labels': has_labels,
                'llm_response': llm_text if 'llm_text' in locals() else None
            }

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise RuntimeError(
                f"Failed to process query '{query}': {e}") from e

    def _create_integrated_response(self, llm_text: str, data: dict,
                                    query_type: str, has_labels: bool) -> str:
        """Create integrated response combining LLM text with analytics."""
        # Clean and process LLM response
        llm_clean = self._clean_llm_response(llm_text)
        if len(llm_clean) > 300:
            llm_clean = llm_clean[:300] + "..."

        # Create analytics summary based on query type
        if query_type == "lineage":
            analytics_summary = (
                f"{data.get('predicted_lineage', 'Unknown')} lineage detected"
                f"across {data.get('num_nodes', 0)} entities "
                f"(confidence: {data.get('confidence', 0):.3f})")
        elif query_type == "silo":
            analytics_summary = (
                f"Analytics: {data.get('isolated_silos', 0)} isolated silos"
                f"out of {data.get('num_nodes', 0)} entities "
                f"({data.get('silo_ratio', 0):.1%} isolation rate)")
        elif query_type == "quality":
            analytics_summary = (
                f"Analytics: Quality score {data.get('avg_quality', 0):.3f}"
                f"({data.get('status', 'UNKNOWN')} overall status)")
        elif query_type == "impact":
            analytics_summary = (
                f"Analytics: {data.get('high_impact', 0)} high-impact entities"
                f"detected ({data.get('risk_level', 'UNKNOWN')} risk)")
        else:
            analytics_summary = f"{data.get('num_nodes', 0)} entities analyzed"

        # Create coherent integrated response
        if llm_clean and len(llm_clean.strip()) > 20:
            # Use LLM response as primary content with analytics as validation
            integrated_response = f"""{llm_clean}

Quantitative Analysis: {analytics_summary}"""
        else:
            # Fallback to analytics-focused response
            integrated_response = (
                f"Based on the warehouse structure analysis:\n\n"
                f"{analytics_summary}\n\n"
                f"The system shows typical patterns for this type of "
                f"data warehouse configuration.")

        return integrated_response

    def _create_contextual_prompt(self, query: str, x: Tensor,
                                  edge_index: Tensor, query_type: str,
                                  context: dict | None = None) -> str:
        """Create context-aware prompt with graph information."""
        # Analyze graph structure
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

        # Determine graph density
        max_edges = num_nodes * (num_nodes - 1)
        density = (2 * num_edges) / max_edges if max_edges > 0 else 0

        if density > 0.1:
            connectivity = "highly connected"
        elif density > 0.05:
            connectivity = "moderately connected"
        else:
            connectivity = "sparsely connected"

        # Create domain-specific context
        domain_context = ""
        if context and 'node_types' in context:
            node_types = context['node_types']
            domain_context = (
                f"The data contains {len(node_types)} entity types: "
                f"{', '.join(node_types[:5])}.")
        else:
            # Infer domain from graph characteristics
            if num_nodes > 300 and connectivity == "sparsely connected":
                domain_context = (
                    "This appears to be a large-scale relational database "
                    "with multiple entity types.")
            elif avg_degree > 10:
                domain_context = (
                    "This appears to be a highly interconnected system with "
                    "complex relationships.")
            else:
                domain_context = (
                    "This appears to be a structured data warehouse with "
                    "defined relationships.")

        # Create task-specific prompt
        if query_type == "lineage":
            task_context = """
Data lineage analysis focuses on tracing data flow and transformations.
Key aspects to consider:
- Direct connections (raw data sources)
- Staged data (intermediate processing)
- Transformed data (modified/computed values)
- Aggregated data (summarized information)
- Derived data (calculated from other data)
"""
        elif query_type == "silo":
            task_context = """
Data silo analysis identifies isolated or poorly connected data sources.
Key aspects to consider:
- Isolated nodes (disconnected data sources)
- Bridge nodes (connecting different data domains)
- Cluster formation (related data groups)
- Integration opportunities (potential connections)
"""
        elif query_type == "quality":
            task_context = """
Data quality analysis evaluates the reliability and completeness of data.
Key aspects to consider:
- Data completeness (missing values)
- Data consistency (conflicting information)
- Data accuracy (correctness of values)
- Data freshness (how up-to-date the data is)
"""
        elif query_type == "impact":
            task_context = """
Impact analysis assesses the downstream effects of data changes.
Key aspects to consider:
- High-impact nodes (affecting many other entities)
- Dependency chains (cascading effects)
- Critical paths (essential data flows)
- Risk assessment (potential failure points)
"""
        else:
            task_context = "General warehouse intelligence analysis."

        # Construct the contextual prompt
        contextual_prompt = f"""You are a data warehouse intelligence expert
analyzing a specific data warehouse.

WAREHOUSE CONTEXT:
- Graph structure: {num_nodes} entities, {num_edges} relationships
- Connectivity: {connectivity} (average degree: {avg_degree:.1f})
- {domain_context}

ANALYSIS TASK:
{task_context.strip()}

USER QUESTION: {query}

Please provide a specific analysis of this warehouse based on the graph
structure and relationships. Focus on concrete insights rather than general
definitions. Be concise and actionable."""

        return contextual_prompt

    def _clean_llm_response(self, llm_text: str) -> str:
        """Clean and format LLM response for better coherence."""
        if not llm_text:
            return ""

        # Remove common artifacts
        cleaned = llm_text.strip()

        # Remove markdown artifacts
        cleaned = cleaned.replace('### Assistant:',
                                  '').replace('### Human:', '')
        cleaned = cleaned.replace('[ST:', '').replace('[O:',
                                                      '').replace(']', '')

        # Remove conversation fragments
        if '### Human:' in cleaned:
            cleaned = cleaned.split('### Human:')[0].strip()

        # Remove incomplete sentences at the end
        sentences = cleaned.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            cleaned = '.'.join(sentences[:-1]) + '.'

        # Remove leading/trailing quotes or brackets
        cleaned = cleaned.strip('"\'[]{}()')

        # Ensure it starts with a capital letter
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned

    def _extract_lineage_data(
            self, pred: Tensor, num_nodes: int,
            context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract data for lineage analysis.

        Args:
            pred: Model predictions
            num_nodes: Number of nodes
            context: Optional context with real labels

        Returns:
            Dictionary with lineage analysis data
        """
        if context and 'warehouse_labels' in context:
            # Use real labels
            labels = context['warehouse_labels']['lineage']
            if labels is not None:
                distribution = torch.bincount(
                    labels, minlength=NUM_LINEAGE_TYPES).tolist()
                dominant_type = int(torch.mode(labels).values.item())
                lineage_types = [
                    'Direct', 'Staged', 'Transformed', 'Aggregated', 'Derived'
                ]

                return {
                    'num_nodes': num_nodes,
                    'lineage_type': lineage_types[dominant_type],
                    'distribution': distribution,
                    'confidence': 1.0
                }

        # Use predictions
        probs = F.softmax(pred, dim=-1)
        predicted_labels = torch.argmax(probs, dim=-1)
        distribution = torch.bincount(predicted_labels,
                                      minlength=NUM_LINEAGE_TYPES).tolist()
        dominant_type = int(torch.mode(predicted_labels).values.item())
        confidence = probs.max(dim=-1).values.mean().item()

        lineage_types = [
            'Direct', 'Staged', 'Transformed', 'Aggregated', 'Derived'
        ]

        return {
            'num_nodes': num_nodes,
            'lineage_type': lineage_types[dominant_type],
            'distribution': distribution,
            'confidence': confidence
        }

    def _extract_silo_data(
            self, pred: Tensor, num_nodes: int,
            context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract data for silo analysis.

        Args:
            pred: Model predictions
            num_nodes: Number of nodes
            context: Optional context with real labels

        Returns:
            Dictionary with silo analysis data
        """
        if context and 'warehouse_labels' in context:
            # Use real labels
            labels = context['warehouse_labels']['silo']
            if labels is not None:
                silo_count = (labels > SILO_THRESHOLD).sum().item()
                connected_count = num_nodes - silo_count
                silo_ratio = silo_count / num_nodes
                if silo_ratio > SILO_CRITICAL_RATIO:
                    status = 'CRITICAL'
                elif silo_ratio > SILO_MODERATE_RATIO:
                    status = 'MODERATE'
                else:
                    status = 'GOOD'

                return {
                    'num_nodes': num_nodes,
                    'silo_count': silo_count,
                    'connected_count': connected_count,
                    'silo_ratio': silo_ratio,
                    'status': status,
                    'confidence': 1.0
                }

        # Use predictions
        silo_probs = torch.sigmoid(pred).squeeze()
        silo_count = (silo_probs > SILO_THRESHOLD).sum().item()
        connected_count = num_nodes - silo_count
        silo_ratio = silo_count / num_nodes
        confidence = silo_probs.mean().item()

        if silo_ratio > SILO_CRITICAL_RATIO:
            status = 'CRITICAL'
        elif silo_ratio > SILO_MODERATE_RATIO:
            status = 'MODERATE'
        else:
            status = 'GOOD'

        return {
            'num_nodes': num_nodes,
            'silo_count': silo_count,
            'connected_count': connected_count,
            'silo_ratio': silo_ratio,
            'status': status,
            'confidence': confidence
        }

    def _extract_impact_data(self, pred: Tensor,
                             num_nodes: int) -> dict[str, Any]:
        """Extract data for impact analysis.

        Args:
            pred: Model predictions
            num_nodes: Number of nodes

        Returns:
            Dictionary with impact analysis data
        """
        probs = F.softmax(pred, dim=-1)
        predicted_labels = torch.argmax(probs, dim=-1)

        low_impact = (predicted_labels == 0).sum().item()
        medium_impact = (predicted_labels == 1).sum().item()
        high_impact = (predicted_labels == 2).sum().item()

        if high_impact > num_nodes * IMPACT_HIGH_RATIO:
            risk_level = 'HIGH'
        elif high_impact > num_nodes * IMPACT_MEDIUM_RATIO:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        return {
            'num_nodes': num_nodes,
            'high_impact': high_impact,
            'low': low_impact,
            'medium': medium_impact,
            'high': high_impact,
            'risk_level': risk_level,
            'confidence': probs.max(dim=-1).values.mean().item()
        }

    def _extract_quality_data(self, pred: Tensor,
                              num_nodes: int) -> dict[str, Any]:
        """Extract data for quality analysis.

        Args:
            pred: Model predictions
            num_nodes: Number of nodes

        Returns:
            Dictionary with quality analysis data
        """
        scores = torch.sigmoid(pred).squeeze()
        avg_quality = scores.mean().item()
        poor_quality = (scores < QUALITY_THRESHOLD).sum().item()
        high_quality = (scores > EXCELLENCE_THRESHOLD).sum().item()

        threshold = num_nodes * QUALITY_ATTENTION_RATIO
        status = 'NEEDS ATTENTION' if poor_quality > threshold else 'GOOD'

        return {
            'num_nodes': num_nodes,
            'avg_quality': avg_quality,
            'poor_quality': poor_quality,
            'high_quality': high_quality,
            'status': status,
            'confidence': avg_quality
        }

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get the conversation history.

        Returns:
            List of conversation entries
        """
        return self.conversation_history.copy()


class SimpleWarehouseModel(nn.Module):
    """Simplified warehouse model for demo purposes without LLM dependencies.

    This model provides basic warehouse intelligence functionality
    using only GAT and task heads, suitable for testing and demos.

    Args:
        hidden_channels: Hidden dimension for GNN layers
        input_channels: Input feature dimension
    """
    def __init__(self, hidden_channels: int = HIDDEN_DIM,
                 input_channels: int = EMBEDDING_DIM) -> None:
        super().__init__()

        if not HAS_GRETRIEVER:
            raise ImportError(
                "GNN component required. Install PyTorch Geometric.")

        self.gnn = GATWrapper(input_channels, hidden_channels, heads=4)
        self.task_head = WarehouseTaskHead(hidden_channels)

    def __call__(self, question: list[str], x: Tensor, edge_index: Tensor,
                 batch: Tensor | None = None, task: str = "lineage",
                 **kwargs: Any) -> dict[str, Tensor]:
        """Forward pass through simplified model.

        Args:
            question: List of questions (for compatibility)
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
            task: Task type
            **kwargs: Additional arguments

        Returns:
            Dictionary with predictions and embeddings
        """
        try:
            # Simple GNN forward pass
            node_emb = self.gnn(x, edge_index)
            pred = self.task_head(node_emb, task=task)

            return {
                'pred': pred,
                'node_emb': node_emb,
                'graph_emb': pred.mean(dim=0)
            }
        except Exception as e:
            logger.error(f"SimpleWarehouseModel forward failed: {e}")
            raise RuntimeError(f"Model forward pass failed: {e}") from e

    def inference(self, question: list[str], **kwargs: Any) -> list[str]:
        """Simple inference returning basic responses."""
        return [f"Analysis result for: {q}" for q in question]

    def predict_task(self, x: Tensor, edge_index: Tensor,
                     task: str = "lineage") -> Tensor:
        """Predict task-specific outputs.

        Args:
            x: Node features
            edge_index: Edge connectivity
            task: Task type

        Returns:
            Task predictions
        """
        try:
            node_emb = self.gnn(x, edge_index)
            pred = self.task_head(node_emb, task=task)
            return pred
        except Exception as e:
            logger.error(f"SimpleWarehouseModel predict_task failed: {e}")
            raise RuntimeError(f"Task prediction failed: {e}") from e


def create_warehouse_demo(
        llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v0.1",
        llm_temperature: float = 0.7, llm_top_k: int = 50,
        llm_top_p: float = 0.9, llm_max_tokens: int = DEFAULT_MAX_TOKENS,
        gnn_hidden_channels: int = HIDDEN_DIM, gnn_num_layers: int = 2,
        gnn_heads: int = 4, dropout: float = DROPOUT, device: str = "cpu",
        use_gretriever: bool = True) -> WarehouseConversationSystem:
    """Create a warehouse demo system with configurable parameters.

    Args:
        llm_model_name: HuggingFace model name for the LLM
        llm_temperature: Temperature for LLM text generation
        llm_top_k: Top-k sampling for LLM
        llm_top_p: Top-p (nucleus) sampling for LLM
        llm_max_tokens: Maximum tokens to generate
        gnn_hidden_channels: Hidden channels for GNN layers
        gnn_num_layers: Number of GNN layers
        gnn_heads: Number of attention heads for GAT
        dropout: Dropout rate for regularization
        device: Device to run models on
        use_gretriever: Whether to use G-Retriever integration

    Returns:
        Configured warehouse conversation system

    Raises:
        ImportError: If required components are not available
    """
    try:
        model: WarehouseGRetriever | SimpleWarehouseModel
        if HAS_GRETRIEVER and use_gretriever:
            model = WarehouseGRetriever(
                llm_model_name=llm_model_name,
                gnn_hidden_channels=gnn_hidden_channels,
                gnn_num_layers=gnn_num_layers, gnn_heads=gnn_heads,
                dropout=dropout)
        else:
            model = SimpleWarehouseModel(hidden_channels=gnn_hidden_channels,
                                         input_channels=EMBEDDING_DIM)

        return WarehouseConversationSystem(model)

    except Exception as e:
        logger.error(f"Failed to create warehouse demo: {e}")
        # Fallback to simple model
        model = SimpleWarehouseModel()
        return WarehouseConversationSystem(model)
