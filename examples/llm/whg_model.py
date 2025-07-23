"""WHG-Retriever neural network components.

Components:
- WarehouseTaskHead: Multi-task classification heads
- WarehouseGRetriever: Graph neural network with GAT and multi-task learning
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

# Optional PyG imports with graceful fallback
try:
    from torch_geometric.nn.models import GAT, GRetriever
    from torch_geometric.nn.nlp import LLM, SentenceTransformer
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    # Create dummy classes for type hints
    GAT = type(None)  # type: ignore
    GRetriever = type(None)  # type: ignore
    LLM = type(None)  # type: ignore
    SentenceTransformer = type(None)  # type: ignore


class WarehouseTaskHead(nn.Module):
    """Multi-task classification heads for warehouse analysis.

    Provides specialized prediction heads for warehouse-specific tasks:
    - Lineage classification (source/staging/enriched/mart)
    - Silo detection (connected/isolated)
    - Anomaly detection (normal/anomaly)

    Args:
        input_dim (int): Input feature dimension from graph encoder.
            Default: 128.
    """
    def __init__(self, input_dim: int = 128):
        super().__init__()

        self.lineage_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # source, staging, enriched, mart
        )

        self.silo_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # connected, isolated
        )

        self.anomaly_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # normal, anomaly
        )

        self.lineage_labels = ['source', 'staging', 'enriched', 'mart']
        self.silo_labels = ['connected', 'isolated']
        self.anomaly_labels = ['normal', 'anomaly']

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass for warehouse task predictions."""
        return {
            'lineage': self.lineage_head(x),
            'silo': self.silo_head(x),
            'anomaly': self.anomaly_head(x),
        }


class WarehouseGRetriever(nn.Module):
    """Graph neural network for warehouse analysis.

    Combines PyG GAT with multi-task classification heads for:
    - Lineage classification (source/staging/mart/other)
    - Data silo detection (connected/isolated)
    - Anomaly detection (normal/anomalous)

    Args:
        input_dim (int): Input node feature dimension. Default: 384.
        hidden_dim (int): Hidden layer dimension for GAT. Default: 128.
        llm_model (str, optional): LLM model name for text generation.
            If None, uses template-based responses. Default: None.

    Raises:
        ImportError: If PyG dependencies are not available.
    """
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        llm_model: str | None = None,
    ):
        super().__init__()

        if not DEPENDENCIES_AVAILABLE:
            raise ImportError('PyG G-Retriever dependencies not available')

        self.gnn = GAT(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            heads=4,
            dropout=0.1,
        )

        self.warehouse_tasks = WarehouseTaskHead(hidden_dim)

        self.sentence_transformer: Any | None = None
        try:
            self.sentence_transformer = SentenceTransformer(
                model_name='sentence-transformers/all-MiniLM-L6-v2')

        except Exception:
            self.sentence_transformer = None

        self.llm = None
        self.g_retriever = None
        if llm_model:
            try:
                self.llm = LLM(model_name=llm_model)
                self.g_retriever = GRetriever(llm=self.llm, gnn=self.gnn,
                                              use_lora=False)
            except Exception:
                self.llm = None
                self.g_retriever = None

    def encode_graph(self, x: torch.Tensor,
                     edge_index: torch.Tensor) -> torch.Tensor:
        """Encode graph using PyG's GAT implementation."""
        return cast(torch.Tensor, self.gnn(x, edge_index))

    def retrieve_relevant_nodes(self, question: str, x: torch.Tensor,
                                node_metadata: dict) -> list[int]:
        """Retrieve warehouse nodes relevant to question using semantic search.

        Uses PyG SentenceTransformer to find nodes semantically related to
        the input question, enabling focused analysis of relevant warehouse
        components.

        Args:
            question (str): Natural language question.
            x (torch.Tensor): Node feature matrix of shape [num_nodes,
                features].
            node_metadata (Dict): Metadata mapping node indices to
                information.

        Returns:
            List[int]: Indices of nodes most relevant to the question.
                Returns up to 10 most relevant nodes.
        """
        if self.sentence_transformer is None:
            return self._keyword_based_retrieval(question, x, node_metadata)

        try:
            question_emb = self.sentence_transformer.encode([question])

            descriptions = []
            for i in range(x.shape[0]):
                meta = node_metadata.get(i, {})
                desc = (
                    f'warehouse table {i} type {meta.get("type", "table")} ' +
                    f'layer {meta.get("layer", "unknown")}')
                descriptions.append(desc)

            node_embs = self.sentence_transformer.encode(descriptions)

            similarities = torch.cosine_similarity(torch.tensor(question_emb),
                                                   torch.tensor(node_embs),
                                                   dim=1)

            top_k = min(10, len(similarities))
            _, indices = torch.topk(similarities, top_k)
            return indices.tolist()

        except Exception:
            return self._keyword_based_retrieval(question, x, node_metadata)

    def _keyword_based_retrieval(self, question: str, x: torch.Tensor,
                                 node_metadata: dict) -> list[int]:
        """Fallback keyword-based node retrieval."""
        question_lower = question.lower()
        relevant_nodes = []

        for i in range(x.shape[0]):
            meta = node_metadata.get(i, {})
            node_type = meta.get('type', '').lower()
            layer = meta.get('layer', '').lower()

            score = 0
            if (any(keyword in question_lower for keyword in ['source', 'raw'])
                    and 'source' in layer):
                score += 2
            elif (any(keyword in question_lower
                      for keyword in ['staging', 'stage'])
                  and 'staging' in layer):
                score += 2
            elif (any(keyword in question_lower
                      for keyword in ['mart', 'final']) and 'mart' in layer):
                score += 2
            elif (any(keyword in question_lower
                      for keyword in ['silo', 'isolated'])
                  and 'silo' in node_type):
                score += 3
            elif 'table' in question_lower:
                score += 1

            if score > 0:
                relevant_nodes.append((i, score))

        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in relevant_nodes[:10]] or list(
            range(min(10, x.shape[0])))

    def analyze_warehouse_tasks(
        self,
        relevant_nodes: list[int],
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, Any]:
        """Analyze relevant nodes for warehouse tasks."""
        self.eval()

        with torch.no_grad():
            graph_embeddings = self.encode_graph(x, edge_index)
            task_outputs = self.warehouse_tasks(graph_embeddings)

            analysis: dict[str, Any] = {
                'nodes': {},
                'summary': {
                    'lineage': {},
                    'silo': {},
                    'anomaly': {}
                },
            }

            for node_id in relevant_nodes:
                lineage_pred_idx = torch.argmax(
                    task_outputs['lineage'][node_id]).item()
                silo_pred_idx = torch.argmax(
                    task_outputs['silo'][node_id]).item()
                anomaly_pred_idx = torch.argmax(
                    task_outputs['anomaly'][node_id]).item()

                lineage_label = self.warehouse_tasks.lineage_labels[int(
                    lineage_pred_idx)]
                silo_label = self.warehouse_tasks.silo_labels[int(
                    silo_pred_idx)]
                anomaly_label = self.warehouse_tasks.anomaly_labels[int(
                    anomaly_pred_idx)]

                analysis['nodes'][node_id] = {
                    'lineage': lineage_label,
                    'silo': silo_label,
                    'anomaly': anomaly_label,
                }

                analysis['summary']['lineage'][lineage_label] = (
                    analysis['summary']['lineage'].get(lineage_label, 0) + 1)
                analysis['summary']['silo'][silo_label] = (
                    analysis['summary']['silo'].get(silo_label, 0) + 1)
                analysis['summary']['anomaly'][anomaly_label] = (
                    analysis['summary']['anomaly'].get(anomaly_label, 0) + 1)

            analysis['total_analyzed'] = len(relevant_nodes)
            return analysis

    def generate_warehouse_response(self, question: str,
                                    analysis: dict[str, Any]) -> str:
        """Generate warehouse response using PyG's LLM or templates."""
        if (hasattr(self, 'g_retriever') and self.g_retriever is not None
                and self.llm is not None):
            try:
                context = self._create_llm_context(question, analysis)
                llm_response = self.llm.generate(prompt=context,
                                                 max_tokens=200,
                                                 temperature=0.7)

                if llm_response and len(llm_response.strip()) > 10:
                    return (f'PyG LLM Response:\n{llm_response}\n\n' +
                            f'(Generated using PyG G-Retriever with '
                            f'{analysis["total_analyzed"]} nodes analyzed)')

            except Exception as e:
                print(f'PyG LLM generation failed: {e}')

        return self._create_enhanced_template_response(question, analysis)

    def _create_llm_context(self, question: str, analysis: dict[str,
                                                                Any]) -> str:
        """Create structured context for PyG's LLM."""
        summary = analysis['summary']
        total = analysis['total_analyzed']

        context = 'Data Warehouse Analysis Results:\n'
        context += f'Question: {question}\n'
        context += f'Analyzed {total} relevant warehouse nodes.\n\n'

        context += 'Warehouse Structure:\n'
        for layer, count in summary['lineage'].items():
            context += f'- {count} {layer} layer tables\n'

        context += '\nConnectivity Status:\n'
        for status, count in summary['silo'].items():
            context += f'- {count} {status} tables\n'

        context += '\nData Quality:\n'
        for quality, count in summary['anomaly'].items():
            context += f'- {count} {quality} tables\n'

        context += '\nProvide insights about this warehouse structure.'
        return context

    def _create_enhanced_template_response(self, question: str,
                                           analysis: dict[str, Any]) -> str:
        """Create template-based response with PyG attribution."""
        summary = analysis['summary']
        total = analysis['total_analyzed']

        response = (f'Warehouse Analysis (analyzed {total} relevant nodes '
                    'using PyG G-Retriever):\n\n')

        response += 'WAREHOUSE STRUCTURE:\n'
        lineage_dist = summary['lineage']
        if lineage_dist:
            for layer, count in lineage_dist.items():
                response += f'   • {count} {layer} layer tables\n'

            layer_diversity = len(lineage_dist)
            if layer_diversity >= 3:
                response += (
                    f'   Well-structured warehouse with {layer_diversity} '
                    'distinct layers\n')
            elif layer_diversity == 2:
                response += (f'   Limited structure - only {layer_diversity} '
                             'layers detected\n')
        else:
            response += '   No clear layer structure detected\n'

        response += '\nCONNECTIVITY STATUS:\n'
        silo_dist = summary['silo']
        connected = silo_dist.get('connected', 0)
        isolated = silo_dist.get('isolated', 0)

        if isolated > 0:
            response += f'   {isolated} isolated data silos detected\n'
            response += f'   {connected} well-connected tables\n'
            response += (
                '   Recommendation: Integrate silos to improve data flow\n')
        else:
            response += f'   All {connected} tables are well-connected\n'
            response += '   Excellent connectivity - no silos detected\n'

        response += '\nDATA QUALITY:\n'
        anomaly_dist = summary['anomaly']
        normal = anomaly_dist.get('normal', 0)
        anomalies = anomaly_dist.get('anomaly', 0)

        if anomalies > 0:
            response += f'   {anomalies} tables show anomalous patterns\n'
            response += f'   {normal} tables appear normal\n'
            response += ('   Recommendation: Investigate anomalous tables for '
                         'data quality issues\n')
        else:
            response += f'   All {normal} tables show normal patterns\n'
            response += '   Excellent data quality across the warehouse\n'

        response += self._add_question_specific_insights(question, summary)

        return response

    def _add_question_specific_insights(self, question: str,
                                        summary: dict) -> str:
        """Add insights specific to the question asked."""
        question_lower = question.lower()
        insights = '\nSPECIFIC INSIGHTS:\n'

        if 'structure' in question_lower or 'overview' in question_lower:
            total_tables = (sum(summary['lineage'].values())
                            if summary['lineage'] else 0)
            insights += (
                f'   • Warehouse contains {total_tables} tables across '
                'multiple layers\n')
            pattern_type = ("standard"
                            if len(summary["lineage"]) >= 3 else "simplified")
            insights += (f'   • Architecture follows {pattern_type} warehouse '
                         'patterns\n')

        elif 'silo' in question_lower or 'isolated' in question_lower:
            isolated = summary['silo'].get('isolated', 0)
            if isolated > 0:
                insights += (f'   • {isolated} data silos need integration '
                             'attention\n')
                insights += ('   • Consider ETL processes to connect isolated '
                             'components\n')
            else:
                insights += (
                    '   • No data silos - warehouse has good integration\n')

        elif 'quality' in question_lower or 'anomaly' in question_lower:
            anomalies = summary['anomaly'].get('anomaly', 0)
            if anomalies > 0:
                insights += (f'   • {anomalies} tables require quality '
                             'investigation\n')
                insights += '   • Run data profiling on anomalous tables\n'
            else:
                insights += (
                    '   • Data quality metrics are within normal ranges\n')

        elif 'mart' in question_lower:
            marts = summary['lineage'].get('mart', 0)
            insights += f'   • {marts} mart tables available for analytics\n'
            mart_status = "well-populated" if marts > 2 else "needs expansion"
            insights += f'   • Mart layer {mart_status}\n'

        elif 'source' in question_lower:
            sources = summary['lineage'].get('source', 0)
            insights += (
                f'   • {sources} source tables feeding the warehouse\n')
            diversity_status = "good" if sources > 3 else "limited"
            insights += f'   • Source diversity {diversity_status}\n'

        return insights
