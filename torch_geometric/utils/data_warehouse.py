"""Data Warehouse Intelligence utilities for PyTorch Geometric.

Neural components for warehouse graph analysis with multi-task learning
for lineage detection, silo analysis, and quality assessment.
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG components
try:
    from torch_geometric.nn import GAT, GRetriever
    from torch_geometric.nn.nlp import LLM
    _HAS_PYG_COMPONENTS = True
except ImportError:
    # Fallback for missing components
    GAT = Any
    GRetriever = Any
    LLM = Any
    _HAS_PYG_COMPONENTS = False
try:
    from torch_geometric.typing import Tensor
except ImportError:
    from torch import Tensor


class WarehouseTaskHead(nn.Module):
    """Multi-task head for warehouse intelligence operations.

    Supports various warehouse tasks including lineage prediction,
    impact analysis, and data quality assessment.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        num_lineage_types: int = 5,
        num_impact_levels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Task-specific heads
        self.lineage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_lineage_types))

        self.impact_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_impact_levels))

        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

        # Silo detection head (binary classification)
        self.silo_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                       nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: Tensor, task: str = "lineage") -> Tensor:
        """Forward pass for specific warehouse task.

        Args:
            x: Node embeddings [num_nodes, hidden_dim]
            task: Task type ("lineage", "impact", "quality", "silo")

        Returns:
            Task-specific predictions
        """
        if task == "lineage":
            return self.lineage_head(x)
        elif task == "impact":
            return self.impact_head(x)
        elif task == "quality":
            return torch.sigmoid(self.quality_head(x))
        elif task == "silo":
            return torch.sigmoid(self.silo_head(x))
        else:
            raise ValueError(f"Unknown task: {task}")


class WarehouseGRetriever(nn.Module):
    """Warehouse Graph Retriever for data analysis.

    Graph neural network with warehouse-specific task heads for
    lineage tracking, impact analysis, and quality assessment.
    """
    def __init__(self, hidden_channels: int = 256, num_gnn_layers: int = 4,
                 llm_model_name: str = 'TinyLlama/TinyLlama-1.1B-Chat-v0.1',
                 **kwargs: Any) -> None:
        super().__init__()

        # Create GNN component
        if _HAS_PYG_COMPONENTS:
            self.gnn = GAT(
                in_channels=1024,
                hidden_channels=hidden_channels,
                out_channels=1024,
                num_layers=num_gnn_layers,
                heads=4,
            )
        else:
            self.gnn = None

        # Create LLM component
        if _HAS_PYG_COMPONENTS:
            self.llm = LLM(
                model_name=llm_model_name,
                num_params=1,  # For TinyLlama
            )
        else:
            self.llm = None

        # Core GRetriever
        if _HAS_PYG_COMPONENTS:
            self.gretriever = GRetriever(llm=self.llm, gnn=self.gnn, **kwargs)
        else:
            self.gretriever = None

        # Warehouse-specific components
        self.task_head = WarehouseTaskHead(hidden_dim=hidden_channels,
                                           dropout=0.1)

        # Lineage-specific embeddings
        self.lineage_encoder = nn.Embedding(
            10, hidden_channels)  # Common lineage types
        self.temporal_encoder = nn.Linear(1, hidden_channels)  # Time encoding

    def forward(self, question: List[str], x: Tensor, edge_index: Tensor,
                batch: Optional[Tensor] = None,
                edge_attr: Optional[Tensor] = None, task: str = "lineage",
                **kwargs: Any) -> Dict[str, Tensor]:
        """Forward pass with warehouse intelligence.

        Args:
            question: Natural language queries
            x: Node features
            edge_index: Graph connectivity
            batch: Batch assignment for nodes
            edge_attr: Edge attributes
            task: Warehouse task type
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing:
            - 'pred': Task predictions
            - 'node_emb': Node embeddings
            - 'graph_emb': Graph-level embeddings
        """
        # Get base GRetriever embeddings
        out = self.gretriever(question=question, x=x, edge_index=edge_index,
                              batch=batch, edge_attr=edge_attr, **kwargs)

        # Extract embeddings
        if isinstance(out, dict):
            node_emb = out.get('node_emb', out.get('pred', x))
            graph_emb = out.get('graph_emb', None)
        else:
            node_emb = out
            graph_emb = None

        # Apply warehouse-specific task head
        task_pred = self.task_head(node_emb, task=task)

        return {
            'pred':
            task_pred,
            'node_emb':
            node_emb,
            'graph_emb':
            (graph_emb if graph_emb is not None else task_pred.mean(dim=0)),
            'base_out':
            out
        }

    def encode_lineage(self, lineage_types: Tensor,
                       timestamps: Optional[Tensor] = None) -> Tensor:
        """Encode lineage information.

        Args:
            lineage_types: Lineage type indices [num_nodes]
            timestamps: Optional timestamps [num_nodes, 1]

        Returns:
            Lineage embeddings [num_nodes, hidden_channels]
        """
        lineage_emb = self.lineage_encoder(lineage_types)

        if timestamps is not None:
            temporal_emb = self.temporal_encoder(timestamps)
            lineage_emb = lineage_emb + temporal_emb

        return lineage_emb


class WarehouseConversationSystem:
    """Natural language interface for warehouse queries.

    Processes warehouse intelligence queries and generates
    structured analysis responses.
    """
    def __init__(self, model: Union[WarehouseGRetriever, Any],
                 device: str = "cpu"):
        self.model = model
        self.device = device
        self.conversation_history: List[Dict[str, str]] = []

        # Common warehouse query templates
        self.query_templates = {
            "lineage": [
                "What is the lineage of {table}?",
                "Show me the data flow for {table}",
                "Where does {table} get its data from?"
            ],
            "impact": [
                "What would be impacted if {table} changes?",
                "Show me downstream dependencies of {table}",
                "What tables depend on {table}?"
            ],
            "quality": [
                "What is the data quality of {table}?",
                "Are there any quality issues with {table}?",
                "How reliable is the data in {table}?"
            ]
        }

    def process_query(
            self, query: str, graph_data: Dict[str, Tensor],
            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a natural language warehouse query.

        Args:
            query: Natural language question
            graph_data: Graph data dictionary with x, edge_index, etc.
            context: Optional context information

        Returns:
            Response dictionary with answer and metadata
        """
        # Determine query type
        query_type = self._classify_query(query)

        # Prepare model inputs
        question = [query]
        x = graph_data['x'].to(self.device)
        edge_index = graph_data['edge_index'].to(self.device)
        batch = graph_data.get('batch', None)
        if batch is not None:
            batch = batch.to(self.device)

        # Get model predictions
        with torch.no_grad():
            output = self.model(question=question, x=x, edge_index=edge_index,
                                batch=batch, task=query_type)

        # Format response
        response = self._format_response(query, output, query_type, context)

        # Update conversation history
        self.conversation_history.append({
            "query": query,
            "response": response["answer"],
            "type": query_type
        })

        return response

    def _classify_query(self, query: str) -> str:
        """Classify query type based on keywords."""
        query_lower = query.lower()

        if any(word in query_lower
               for word in ["lineage", "source", "origin", "flow"]):
            return "lineage"
        elif any(word in query_lower
                 for word in ["silo", "isolated", "disconnect", "separate"]):
            return "silo"
        elif any(word in query_lower
                 for word in ["impact", "downstream", "depend", "affect"]):
            return "impact"
        elif any(word in query_lower
                 for word in ["quality", "reliable", "issue", "problem"]):
            return "quality"
        else:
            return "lineage"  # Default

    def _format_response(
            self, query: str, output: Dict[str, Tensor], query_type: str,
            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format model output into conversational response."""
        pred = output['pred']

        # Get basic statistics
        node_emb = output['node_emb']
        num_nodes = node_emb.shape[0]
        avg_confidence = pred.mean().item()

        if query_type == "lineage":
            # Check if we have real RelBench labels
            has_labels = (context is not None
                          and context.get('warehouse_labels') is not None
                          and context['warehouse_labels'].get('lineage')
                          is not None)
            if has_labels:
                # Use real labels from RelBench
                real_labels = context['warehouse_labels']['lineage']
                lineage_counts = torch.bincount(real_labels, minlength=5)
                dominant_type = int(torch.argmax(lineage_counts).item())

                lineage_names = [
                    "Direct", "Derived", "Aggregated", "Joined", "Transformed"
                ]

                answer = (
                    f"RelBench Analysis of {num_nodes} entities:\n"
                    f"• Real lineage pattern: {lineage_names[dominant_type]}\n"
                    f"• Actual distribution: {lineage_counts.tolist()}\n"
                    f"• Data source: RelBench warehouse labels")
            else:
                # Use model predictions
                lineage_scores = F.softmax(pred, dim=-1)
                top_lineage = torch.argmax(lineage_scores, dim=-1)

                lineage_counts = torch.bincount(top_lineage, minlength=5)
                dominant_type = int(torch.argmax(lineage_counts).item())

                lineage_names = [
                    "Direct", "Derived", "Aggregated", "Joined", "Transformed"
                ]

                answer = (
                    f"Model Analysis of {num_nodes} entities:\n"
                    f"• Predicted lineage: {lineage_names[dominant_type]}\n"
                    f"• Distribution: {lineage_counts.tolist()}\n"
                    f"• Average confidence: {avg_confidence:.3f}")

        elif query_type == "silo":
            # Check if we have real RelBench silo labels
            has_silo_labels = (context is not None
                               and context.get('warehouse_labels') is not None
                               and context['warehouse_labels'].get('silo')
                               is not None)
            if has_silo_labels:
                # Use real silo labels from RelBench
                real_silo_labels = context['warehouse_labels']['silo']
                silo_count = (real_silo_labels == 1).sum().item()
                connected_count = (real_silo_labels == 0).sum().item()

                threshold = num_nodes * 0.2
                status = 'CRITICAL' if silo_count > threshold else 'ACCEPTABLE'
                answer = (f"RelBench Silo Analysis of {num_nodes} entities:\n"
                          f"• Isolated silos detected: {silo_count}\n"
                          f"• Connected entities: {connected_count}\n"
                          f"• Silo ratio: {silo_count/num_nodes:.2%}\n"
                          f"• Status: {status}\n"
                          f"• Data source: RelBench silo labels")
            else:
                # Use model predictions for silo detection
                silo_scores = torch.sigmoid(pred).squeeze()
                silo_predictions = (silo_scores > 0.5).long()
                silo_count = silo_predictions.sum().item()

                connected_count = num_nodes - silo_count
                answer = (
                    f"Model Silo Analysis of {num_nodes} entities:\n"
                    f"• Predicted isolated silos: {silo_count}\n"
                    f"• Connected entities: {connected_count}\n"
                    f"• Predicted silo ratio: {silo_count/num_nodes:.2%}\n"
                    f"• Average silo confidence: {silo_scores.mean():.3f}")

        elif query_type == "impact":
            # Format impact predictions with risk assessment
            impact_scores = F.softmax(pred, dim=-1)
            impact_levels = torch.argmax(impact_scores, dim=-1)

            # Count impact levels
            impact_counts = torch.bincount(impact_levels, minlength=3)
            high_impact = (impact_levels == 2).sum().item()

            threshold = num_nodes * 0.3
            risk_level = 'HIGH' if high_impact > threshold else 'MODERATE'
            answer = (f"Impact analysis across {num_nodes} entities:\n"
                      f"• High impact entities: {high_impact}\n"
                      f"• Impact distribution: Low({impact_counts[0]}), "
                      f"Medium({impact_counts[1]}), High({impact_counts[2]})\n"
                      f"• Risk assessment: {risk_level}")

        elif query_type == "quality":
            # Format quality predictions with detailed metrics
            quality_scores = torch.sigmoid(pred).squeeze()
            avg_quality = quality_scores.mean().item()
            poor_quality = (quality_scores < 0.5).sum().item()
            excellent_quality = (quality_scores > 0.8).sum().item()

            threshold = num_nodes * 0.2
            status = 'NEEDS ATTENTION' if poor_quality > threshold else 'GOOD'
            answer = (f"Data quality assessment for {num_nodes} entities:\n"
                      f"• Average quality score: {avg_quality:.3f}\n"
                      f"• Poor quality entities: {poor_quality}\n"
                      f"• Excellent quality entities: {excellent_quality}\n"
                      f"• Overall status: {status}")

        else:
            answer = (f"Analyzed {num_nodes} entities with "
                      f"confidence {avg_confidence:.3f}")

        return {
            "answer": answer,
            "confidence": avg_confidence,
            "query_type": query_type,
            "raw_output": output
        }

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []


class SimpleWarehouseModel(nn.Module):
    """Simplified warehouse model for demo purposes without LLM deps."""
    def __init__(self, hidden_channels: int = 128):
        super().__init__()
        self.gnn = GAT(
            in_channels=384,  # Match demo data
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=2,
            heads=4,
        )
        self.task_head = WarehouseTaskHead(hidden_channels)

    def __call__(self, question: List[str], x: Tensor, edge_index: Tensor,
                 batch: Optional[Tensor] = None, task: str = "lineage",
                 **kwargs: Any) -> Dict[str, Tensor]:
        # Simple GNN forward pass
        node_emb = self.gnn(x, edge_index)
        pred = self.task_head(node_emb, task=task)

        return {
            'pred': pred,
            'node_emb': node_emb,
            'graph_emb': pred.mean(dim=0),
            'base_out': pred
        }


def create_warehouse_demo() -> WarehouseConversationSystem:
    """Create warehouse conversation system for demo.

    Returns:
        WarehouseConversationSystem instance.
    """
    # Create simplified model for demo
    model = SimpleWarehouseModel(hidden_channels=128)

    # Create conversation system
    conversation_system = WarehouseConversationSystem(model)

    return conversation_system
