"""RelBench dataset integration for PyTorch Geometric.

Converts RelBench datasets to PyG HeteroData with warehouse task labels.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData, InMemoryDataset

# Constants
DEFAULT_SBERT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_BATCH_SIZE = 64
DEFAULT_SAMPLE_SIZE = 100
EMBEDDING_DIM = 384

# Heuristic labeling constants
SILO_THRESHOLD_RATIO = 0.1
SILO_MIN_SIZE = 5
DEGREE_THRESHOLD_MULTIPLIER = 0.5
CENTRALITY_K_LIMIT = 100
CENTRALITY_DEGREE_WEIGHT = 0.7
CENTRALITY_BETWEENNESS_WEIGHT = 0.3
EDGE_LIMIT_PER_TYPE = 10
FALLBACK_SILO_PROB = 0.3
FALLBACK_ANOMALY_PROB = 0.1
MAX_LINEAGE_DEPTH = 4
SILO_ISOLATED = 1.0
DUMMY_EDGE_LIMIT = 50

# Set up logging
logger = logging.getLogger(__name__)

# Optional dependencies with proper error handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("pandas not available, some functionality will be limited")
    PANDAS_AVAILABLE = False
    pd = None

# Import SentenceTransformer with fallbacks
try:
    # yapf: disable
    from torch_geometric.nn.nlp import \
        SentenceTransformer as PyGSentenceTransformer  # isort: skip

    # yapf: enable
    PYG_NLP_AVAILABLE = True
except ImportError:
    logger.debug("PyG NLP not available")
    PYG_NLP_AVAILABLE = False

    class PyGSentenceTransformer:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyG SentenceTransformer not available")


try:
    # yapf: disable
    from sentence_transformers import \
        SentenceTransformer as STSentenceTransformer  # isort: skip

    # yapf: enable
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.debug("sentence-transformers not available")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

    class STSentenceTransformer:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("sentence-transformers not available")


# Check if any SBERT implementation is available
if not PYG_NLP_AVAILABLE and not SENTENCE_TRANSFORMERS_AVAILABLE:
    warnings.warn(
        'Neither PyG NLP nor sentence-transformers available. '
        'Install PyG 2.6.0+ or sentence-transformers for full functionality',
        stacklevel=2,
    )

try:
    from relbench.datasets import get_dataset
    RELBENCH_AVAILABLE = True
except ImportError:
    logger.warning("RelBench not available, using fallback implementations")
    RELBENCH_AVAILABLE = False

# Additional imports for heuristic labeling
try:

    from torch_geometric.utils import degree, to_networkx
    GRAPH_ANALYSIS_AVAILABLE = True
except ImportError:
    logger.warning("PyG graph analysis not available, using basic heuristics")
    GRAPH_ANALYSIS_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    logger.warning("NetworkX not available, using simplified graph analysis")
    NETWORKX_AVAILABLE = False


class RelBenchError(Exception):
    """Custom exception for RelBench-related errors."""


class HeuristicLabeler:
    """Generate warehouse task labels from foreign keys and graph structure."""
    def __init__(self) -> None:
        self.use_networkx = NETWORKX_AVAILABLE
        self.use_graph_analysis = GRAPH_ANALYSIS_AVAILABLE

    def generate_labels(self, hetero_data: HeteroData, db: Any,
                        tasks: list[str] | None = None) -> dict[str, Tensor]:
        """Generate warehouse task labels from database and graph structure."""
        if tasks is None:
            tasks = ['lineage', 'silo', 'anomaly']
        labels = {}
        homo_data = hetero_data.to_homogeneous()
        num_nodes = homo_data.num_nodes

        if num_nodes is None:
            raise ValueError("Cannot generate labels: graph has no nodes")

        if 'lineage' in tasks:
            labels['lineage'] = self._generate_lineage_labels(
                hetero_data, db, num_nodes)
        if 'silo' in tasks:
            labels['silo'] = self._generate_silo_labels(homo_data, num_nodes)
        if 'anomaly' in tasks:
            labels['anomaly'] = self._generate_quality_labels(
                homo_data, num_nodes)

        return labels

    def _generate_lineage_labels(self, hetero_data: HeteroData, db: Any,
                                 num_nodes: int) -> Tensor:
        """Generate lineage labels based on foreign key depth."""
        lineage_labels = torch.zeros(num_nodes, dtype=torch.long)
        fk_depth_map = self._analyze_foreign_key_depth(db)

        node_idx = 0
        for table_name in hetero_data.node_types:
            table_nodes = hetero_data[table_name].num_nodes

            if table_name in fk_depth_map:
                depth = fk_depth_map[table_name]
                lineage_type = min(depth, MAX_LINEAGE_DEPTH)
            else:
                lineage_type = MAX_LINEAGE_DEPTH

            lineage_labels[node_idx:node_idx + table_nodes] = lineage_type
            node_idx += table_nodes

        return lineage_labels

    def _generate_silo_labels(self, homo_data: Data, num_nodes: int) -> Tensor:
        """Generate silo labels based on graph connectivity."""
        if homo_data.num_edges == 0:
            return torch.ones(num_nodes, dtype=torch.float)

        edge_index = homo_data.edge_index
        if edge_index is None:
            return torch.ones(num_nodes, dtype=torch.float)

        if self.use_graph_analysis:
            node_degrees = degree(edge_index[0], num_nodes=num_nodes)
        else:
            node_degrees = torch.bincount(edge_index[0],
                                          minlength=num_nodes).float()

        if self.use_networkx:
            try:
                G = to_networkx(homo_data, to_undirected=True)
                components = list(nx.connected_components(G))
                largest_component_size = max(
                    len(comp) for comp in components) if components else 0
                silo_threshold = max(
                    largest_component_size * SILO_THRESHOLD_RATIO,
                    SILO_MIN_SIZE)

                silo_labels = torch.zeros(num_nodes, dtype=torch.float)
                for comp in components:
                    if len(comp) < silo_threshold:
                        for node in comp:
                            if node < num_nodes:
                                silo_labels[node] = SILO_ISOLATED
                return silo_labels
            except Exception as e:
                logger.warning(f"NetworkX analysis failed: {e}")

        degree_threshold = node_degrees.mean() * DEGREE_THRESHOLD_MULTIPLIER
        return (node_degrees < degree_threshold).float()

    def _generate_quality_labels(self, homo_data: Data,
                                 num_nodes: int) -> Tensor:
        """Generate anomaly labels based on node centrality."""
        if homo_data.num_edges == 0:
            return torch.zeros(num_nodes, dtype=torch.long)

        edge_index = homo_data.edge_index
        if edge_index is None:
            return torch.zeros(num_nodes, dtype=torch.long)

        if self.use_graph_analysis:
            node_degrees = degree(edge_index[0], num_nodes=num_nodes)
        else:
            node_degrees = torch.bincount(edge_index[0],
                                          minlength=num_nodes).float()

        if self.use_networkx:
            try:
                G = to_networkx(homo_data, to_undirected=True)
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(
                    G, k=min(CENTRALITY_K_LIMIT, len(G)))

                quality_scores = torch.zeros(num_nodes, dtype=torch.float)
                for node in range(num_nodes):
                    if node in G:
                        deg_cent = degree_centrality.get(node, 0)
                        bet_cent = betweenness_centrality.get(node, 0)
                        quality_scores[node] = (
                            CENTRALITY_DEGREE_WEIGHT * deg_cent +
                            CENTRALITY_BETWEENNESS_WEIGHT * bet_cent)

                quality_threshold = quality_scores.mean() - quality_scores.std(
                )
                return (quality_scores < quality_threshold).long()
            except Exception as e:
                logger.warning(f"NetworkX centrality analysis failed: {e}")

        degree_threshold = node_degrees.mean() - node_degrees.std()
        return (node_degrees < degree_threshold).long()

    def _analyze_foreign_key_depth(self, db: Any) -> dict[str, int]:
        """Calculate table depth based on foreign key dependencies."""
        if db is None:
            return {}  # Return empty dict when no database provided

        fk_graph = {}
        for table_name, table_obj in db.table_dict.items():
            fk_relations = table_obj.fkey_col_to_pkey_table
            fk_graph[table_name] = list(
                fk_relations.values()) if fk_relations else []

        def calculate_depth(table: str, visited: set,
                            depth_cache: dict) -> int:
            if table in visited or table in depth_cache:
                return depth_cache.get(table, 0)

            visited.add(table)
            if not fk_graph.get(table):
                depth = 0
            else:
                max_ref_depth = 0
                for ref_table in fk_graph[table]:
                    if ref_table in fk_graph:
                        ref_depth = calculate_depth(ref_table, visited.copy(),
                                                    depth_cache)
                        max_ref_depth = max(max_ref_depth, ref_depth)
                depth = max_ref_depth + 1

            depth_cache[table] = depth
            return depth

        depth_cache: dict[str, int] = {}
        fk_depth = {}
        for table_name in db.table_dict.keys():
            fk_depth[table_name] = calculate_depth(table_name, set(),
                                                   depth_cache)

        return fk_depth


class RelBenchProcessor:
    """Processor for converting RelBench data to PyG format.

    This class handles the conversion of RelBench datasets to PyTorch Geometric
    HeteroData objects with semantic embeddings and warehouse task labels.

    Args:
        sbert_model: Sentence transformer model name or path
        batch_size: Batch size for embedding computation
    """
    def __init__(self, sbert_model: str = DEFAULT_SBERT_MODEL,
                 batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.sbert_model = sbert_model
        self.batch_size = batch_size
        self.encoder: Any = None
        self._initialize_encoder()

    def _initialize_encoder(self) -> None:
        """Initialize the sentence transformer encoder.

        Raises:
            RelBenchError: If no suitable encoder is available
        """
        if PYG_NLP_AVAILABLE:
            try:
                self.encoder = PyGSentenceTransformer(self.sbert_model)
                logger.info(
                    f"Using PyG SentenceTransformer: {self.sbert_model}")
                return
            except Exception as e:
                logger.warning(f"Failed to load PyG SentenceTransformer: {e}")

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.encoder = STSentenceTransformer(self.sbert_model)
                logger.info(f"Using sentence-transformers: {self.sbert_model}")
                return
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformers: {e}")

        raise RelBenchError(
            "No suitable sentence transformer available. "
            "Install PyG with NLP support or sentence-transformers.")

    def encode_texts(self, texts: list[str]) -> Tensor:
        """Encode texts to embeddings with fallback.

        Args:
            texts: List of text strings to encode

        Returns:
            Tensor of embeddings [num_texts, embedding_dim]
        """
        if not texts:
            return torch.empty(0, EMBEDDING_DIM)

        try:
            if PYG_NLP_AVAILABLE and isinstance(self.encoder,
                                                PyGSentenceTransformer):
                # PyG encoder expects tokenized input - let's tokenize properly
                try:
                    # Use the encoder's tokenizer to tokenize texts
                    tokenized = self.encoder.tokenizer(texts,
                                                       return_tensors='pt',
                                                       padding=True,
                                                       truncation=True,
                                                       max_length=512)
                    # Remove token_type_ids if present (not needed for sentence
                    # transformers)
                    if 'token_type_ids' in tokenized:
                        del tokenized['token_type_ids']

                    # Get embeddings from PyG SentenceTransformer
                    with torch.no_grad():
                        embeddings = self.encoder(**tokenized)
                        # PyG SentenceTransformer returns embeddings directly
                        if embeddings.dim() == 3:  # [batch, seq_len, hidden]
                            embeddings = embeddings.mean(dim=1)  # Mean pooling
                    logger.debug(
                        f"PyG SentenceTransformer encoded {len(texts)} texts")
                except Exception as e:
                    logger.warning(
                        f"PyG SentenceTransformer failed: {e}, using fallback")
                    embeddings = torch.randn(len(texts), EMBEDDING_DIM)
            elif self.encoder is not None:
                # sentence-transformers returns numpy array
                embeddings = self.encoder.encode(texts,
                                                 batch_size=self.batch_size,
                                                 convert_to_tensor=True,
                                                 show_progress_bar=False)
            else:
                # Fallback if no encoder available
                embeddings = torch.randn(len(texts), EMBEDDING_DIM)
                if not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.tensor(embeddings)

            return embeddings.float()

        except Exception as e:
            logger.warning(
                f"Text encoding failed: {e}, using random embeddings")
            # Fallback to random embeddings when encoding fails
            return torch.randn(len(texts), EMBEDDING_DIM)

    def process_relbench_data(
            self, dataset_name: str,
            sample_size: int | None = None) -> tuple[HeteroData, Any]:
        """Process RelBench dataset to HeteroData.

        Args:
            dataset_name: Name of the RelBench dataset
            sample_size: Optional sample size for large datasets

        Returns:
            Tuple of (HeteroData with embeddings, RelBench database)

        Raises:
            RelBenchError: If processing fails
        """
        if not RELBENCH_AVAILABLE:
            raise RelBenchError(
                "RelBench not available. Install relbench package.")

        try:
            # Load RelBench dataset
            dataset = get_dataset(dataset_name)
            db = dataset.get_db()

            # Convert to HeteroData with sampling during processing
            hetero_data = self._convert_to_hetero_data(db, sample_size)

            # Add metadata
            hetero_data.dataset_name = dataset_name
            hetero_data.embedding_dim = EMBEDDING_DIM
            hetero_data.processor_info = {
                'sbert_model': self.sbert_model,
                'batch_size': self.batch_size,
                'sample_size': sample_size
            }

            return hetero_data, db

        except Exception as e:
            logger.error(f"RelBench processing failed: {e}")
            raise RelBenchError(
                f"Failed to process {dataset_name}: {e}") from e

    def _convert_to_hetero_data(self, db: Any,
                                sample_size: int | None = None) -> HeteroData:
        """Convert RelBench database to HeteroData with sampling.

        Args:
            db: RelBench database object
            sample_size: Optional sample size per table

        Returns:
            HeteroData object with node and edge data
        """
        hetero_data = HeteroData()

        # Process tables as node types - preserve warehouse structure
        table_names = list(db.table_dict.keys())
        logger.info(
            f"Processing {len(table_names)} tables: {table_names[:5]}...")

        total_nodes = 0
        for table_name in table_names:
            try:
                table_obj = db.table_dict[table_name]

                # Get DataFrame and sample using original approach
                table_df = table_obj.df

                # Sample data if requested (original working approach)
                if sample_size is not None:
                    table_df = table_df.head(min(sample_size, len(table_df)))
                    logger.debug(
                        f"  {table_name}: {len(table_obj.df)}->{len(table_df)}"
                    )

                # Create text representations for embedding
                if PANDAS_AVAILABLE and hasattr(table_df, 'to_dict'):
                    texts = self._create_text_representations(table_df)
                    embeddings = self.encode_texts(texts)

                    hetero_data[table_name].x = embeddings
                    hetero_data[table_name].num_nodes = len(embeddings)
                    total_nodes += len(embeddings)
                    logger.debug(f"  {table_name}: {len(embeddings)} nodes")
                else:
                    # Fallback for non-pandas data
                    num_nodes = len(table_df) if hasattr(table_df,
                                                         '__len__') else 1
                    hetero_data[table_name].x = torch.randn(
                        num_nodes, EMBEDDING_DIM)
                    hetero_data[table_name].num_nodes = num_nodes
                    total_nodes += num_nodes
                    logger.debug(
                        f"  {table_name}: {num_nodes} nodes (fallback)")

            except Exception as e:
                logger.warning(f"Failed to process table {table_name}: {e}")
                # Create minimal node data as fallback
                hetero_data[table_name].x = torch.randn(1, EMBEDDING_DIM)
                hetero_data[table_name].num_nodes = 1
                total_nodes += 1

        logger.info(f"Total nodes across all tables: {total_nodes}")

        # Add basic edge connectivity (simplified)
        self._add_basic_edges(hetero_data, db)

        return hetero_data

    def _create_text_representations(self, table_data: Any) -> list[str]:
        """Create text representations from table data.

        Args:
            table_data: Table data (pandas DataFrame or similar)

        Returns:
            List of text representations
        """
        texts = []

        try:
            if hasattr(table_data, 'iterrows'):
                # pandas DataFrame
                for _, row in table_data.iterrows():
                    text_parts = []
                    for col, val in row.items():
                        if pd.notna(val):
                            text_parts.append(f"{col}: {val}")
                    texts.append(" | ".join(text_parts))
            else:
                # Fallback for other data types
                texts = [str(item) for item in table_data]

        except Exception as e:
            logger.warning(f"Text representation creation failed: {e}")
            texts = ["unknown data"]

        return texts if texts else ["empty table"]

    def _add_basic_edges(self, hetero_data: HeteroData, db: Any) -> None:
        """Add edges based on RelBench foreign key relationships.

        Args:
            hetero_data: HeteroData object to modify
            db: RelBench database object with foreign key metadata
        """
        if db is None:
            logger.warning(
                "No database provided, using fallback edge creation")
            self._add_fallback_edges(hetero_data)
            return

        try:
            # Create node ID mappings for each table
            node_mappings = self._create_node_mappings(hetero_data, db)

            # Create edges based on actual foreign key relationships
            edges_created = 0
            for table_name, table_obj in db.table_dict.items():
                if table_name not in hetero_data.node_types:
                    continue

                # Process each foreign key relationship
                for fkey_col, target_table in (
                        table_obj.fkey_col_to_pkey_table.items()):
                    if target_table not in hetero_data.node_types:
                        continue

                    edges_created += self._create_fkey_edges(
                        hetero_data, table_name, target_table, table_obj,
                        fkey_col, node_mappings)

            logger.info(f"Created {edges_created} real foreign key edges")

        except Exception as e:
            logger.warning(
                f"Foreign key edge creation failed: {e}, using fallback")
            self._add_fallback_edges(hetero_data)

    def _create_node_mappings(self, hetero_data: HeteroData, db: Any) -> dict:
        """Create mappings from primary key values to node indices.

        Args:
            hetero_data: HeteroData object
            db: RelBench database object

        Returns:
            Dictionary mapping table_name -> {pk_value: node_index}
        """
        node_mappings = {}

        for table_name, table_obj in db.table_dict.items():
            if table_name not in hetero_data.node_types:
                continue

            # Get primary key column
            pkey_col = table_obj.pkey_col
            if not pkey_col or not hasattr(table_obj, 'df'):
                continue

            # Get the actual number of nodes in the hetero_data for this table
            actual_num_nodes = hetero_data[table_name].num_nodes

            # Create mapping from primary key values to node indices
            df = table_obj.df
            if pkey_col in df.columns:
                # Only map the first N primary keys where N = actual_num_nodes
                # This handles sampling correctly
                pk_values = df[pkey_col].head(actual_num_nodes).tolist()
                node_mappings[table_name] = {
                    pk_val: idx
                    for idx, pk_val in enumerate(pk_values)
                }

        return node_mappings

    def _create_fkey_edges(self, hetero_data: HeteroData, src_table: str,
                           dst_table: str, table_obj: Any, fkey_col: str,
                           node_mappings: dict) -> int:
        """Create edges based on foreign key relationships.

        Args:
            hetero_data: HeteroData object to modify
            src_table: Source table name
            dst_table: Destination table name
            table_obj: Source table object
            fkey_col: Foreign key column name
            node_mappings: Node ID mappings

        Returns:
            Number of edges created
        """
        if (src_table not in node_mappings or dst_table not in node_mappings
                or not hasattr(table_obj, 'df')):
            return 0

        df = table_obj.df
        if fkey_col not in df.columns:
            return 0

        # Get mappings
        src_mapping = node_mappings[src_table]
        dst_mapping = node_mappings[dst_table]

        # Create edges from foreign key relationships
        src_indices = []
        dst_indices = []

        # Get the actual number of source nodes to limit iteration
        max_src_nodes = len(src_mapping)

        for src_idx, row in enumerate(df.head(max_src_nodes).itertuples()):
            if src_idx >= max_src_nodes:
                break

            fkey_value = getattr(row, fkey_col, None)
            if fkey_value is not None and fkey_value in dst_mapping:
                src_indices.append(src_idx)
                dst_indices.append(dst_mapping[fkey_value])

        if src_indices:
            edge_index = torch.tensor([src_indices, dst_indices],
                                      dtype=torch.long)
            edge_type = (src_table, 'references', dst_table)
            hetero_data[edge_type].edge_index = edge_index

            logger.debug(f"Created {len(src_indices)} edges: {edge_type}")
            return len(src_indices)

        return 0

    def _add_fallback_edges(self, hetero_data: HeteroData) -> None:
        """Add fallback random edges when foreign key processing fails."""
        node_types = list(hetero_data.node_types)

        for i, src_type in enumerate(node_types):
            for j, dst_type in enumerate(node_types):
                if i != j:  # No self-loops for simplicity
                    # Create minimal edge connectivity
                    src_nodes = hetero_data[src_type].num_nodes
                    dst_nodes = hetero_data[dst_type].num_nodes

                    if src_nodes > 0 and dst_nodes > 0:
                        # Create sparse connectivity
                        num_edges = min(src_nodes, dst_nodes,
                                        EDGE_LIMIT_PER_TYPE)
                        src_indices = torch.randint(0, src_nodes,
                                                    (num_edges, ))
                        dst_indices = torch.randint(0, dst_nodes,
                                                    (num_edges, ))

                        edge_index = torch.stack([src_indices, dst_indices])
                        hetero_data[src_type, 'connects_to',
                                    dst_type].edge_index = edge_index


def _add_warehouse_labels_to_hetero_data(
        hetero_data: HeteroData, db: Any = None,
        external_labels: dict[str, Tensor] | None = None,
        create_lineage_labels: bool = True, create_silo_labels: bool = True,
        create_anomaly_labels: bool = True) -> HeteroData:
    """Add warehouse intelligence task labels to HeteroData.

    Args:
        hetero_data: HeteroData object to enhance
        db: RelBench database object for heuristic labeling
        external_labels: Optional external labels to use instead of heuristics
        create_lineage_labels: Whether to create lineage labels
        create_silo_labels: Whether to create silo labels
        create_anomaly_labels: Whether to create anomaly labels

    Returns:
        Enhanced HeteroData with warehouse labels
    """
    try:
        # Convert to homogeneous for label creation
        homo_data = hetero_data.to_homogeneous()
        num_nodes = homo_data.num_nodes
        if num_nodes is None:
            num_nodes = 0

        # Use external labels if provided, otherwise generate heuristics
        if external_labels:
            logger.info("Using provided external labels")
            for task_name, labels in external_labels.items():
                if task_name == 'lineage' and create_lineage_labels:
                    homo_data.lineage_label = labels
                elif task_name == 'silo' and create_silo_labels:
                    homo_data.silo_label = labels
                elif task_name == 'anomaly' and create_anomaly_labels:
                    homo_data.anomaly_label = labels
        else:
            # Generate heuristic labels
            logger.info("Generating heuristic labels")
            labeler = HeuristicLabeler()

            tasks_to_generate = []
            if create_lineage_labels:
                tasks_to_generate.append('lineage')
            if create_silo_labels:
                tasks_to_generate.append('silo')
            if create_anomaly_labels:
                tasks_to_generate.append('anomaly')

            if tasks_to_generate and db is not None:
                labels_dict = labeler.generate_labels(hetero_data, db,
                                                      tasks_to_generate)

                for task_name, task_labels in labels_dict.items():
                    if task_name == 'lineage':
                        homo_data.lineage_label = task_labels
                    elif task_name == 'silo':
                        homo_data.silo_label = task_labels
                    elif task_name == 'anomaly':
                        homo_data.anomaly_label = task_labels
            else:
                # Fallback to random labels if no database provided
                logger.warning(
                    "No database provided for heuristic labeling, using random"
                )
                if create_lineage_labels:
                    homo_data.lineage_label = torch.randint(
                        0, 5, (num_nodes, ))
                if create_silo_labels:
                    homo_data.silo_label = torch.bernoulli(
                        torch.full((num_nodes, ), FALLBACK_SILO_PROB))
                if create_anomaly_labels:
                    homo_data.anomaly_label = torch.bernoulli(
                        torch.full((num_nodes, ), FALLBACK_ANOMALY_PROB))

        # Add labels back to original hetero_data
        hetero_data._homo_data_with_labels = homo_data

        return hetero_data

    except Exception as e:
        logger.error(f"Failed to add warehouse labels: {e}")
        raise RelBenchError(f"Label creation failed: {e}") from e


def create_relbench_hetero_data(
    dataset_name: str,
    sbert_model: str = DEFAULT_SBERT_MODEL,
    sample_size: int | None = None,
    create_lineage_labels: bool = False,
    create_silo_labels: bool = False,
    create_anomaly_labels: bool = False,
    external_labels: dict[str, Tensor] | None = None,
    use_dummy_fallback: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> HeteroData:
    """Create HeteroData from RelBench dataset with warehouse enhancements.

    Args:
        dataset_name: Name of the RelBench dataset
        sbert_model: Sentence transformer model for embeddings
        sample_size: Optional sample size for large datasets
        create_lineage_labels: Whether to create lineage labels
        create_silo_labels: Whether to create silo labels
        create_anomaly_labels: Whether to create anomaly labels
        external_labels: Pre-computed labels to use instead of heuristics
        use_dummy_fallback: Whether to use dummy data if RelBench fails
        batch_size: Batch size for embedding computation

    Returns:
        HeteroData object with embeddings and optional warehouse labels

    Raises:
        RelBenchError: If processing fails and no fallback is requested
    """
    try:
        processor = RelBenchProcessor(sbert_model, batch_size)
        hetero_data, db = processor.process_relbench_data(
            dataset_name, sample_size)

        # Add warehouse labels if requested - use heuristics with database
        if (create_lineage_labels or create_silo_labels
                or create_anomaly_labels):
            hetero_data = _add_warehouse_labels_to_hetero_data(
                hetero_data, db=db, external_labels=external_labels,
                create_lineage_labels=create_lineage_labels,
                create_silo_labels=create_silo_labels,
                create_anomaly_labels=create_anomaly_labels)

        logger.info(f"Successfully created HeteroData from {dataset_name}")
        return hetero_data

    except Exception as e:
        logger.error(f"RelBench processing failed: {e}")

        if use_dummy_fallback:
            logger.info("Using dummy fallback data")
            return _create_dummy_hetero_data(
                sample_size or DEFAULT_SAMPLE_SIZE, create_lineage_labels,
                create_silo_labels, create_anomaly_labels)
        else:
            raise RelBenchError(f"Failed to create HeteroData: {e}") from e


def _create_dummy_hetero_data(sample_size: int, create_lineage_labels: bool,
                              create_silo_labels: bool,
                              create_anomaly_labels: bool) -> HeteroData:
    """Create dummy HeteroData for testing/fallback.

    Args:
        sample_size: Number of nodes to create
        create_lineage_labels: Whether to create lineage labels
        create_silo_labels: Whether to create silo labels
        create_anomaly_labels: Whether to create anomaly labels

    Returns:
        Dummy HeteroData object
    """
    hetero_data = HeteroData()

    # Create single node type with random embeddings
    hetero_data['record'].x = torch.randn(sample_size, EMBEDDING_DIM)
    hetero_data['record'].num_nodes = sample_size

    # Add minimal self-connectivity
    edge_indices = torch.randint(0, sample_size,
                                 (2, min(sample_size, DUMMY_EDGE_LIMIT)))
    hetero_data['record', 'relates_to', 'record'].edge_index = edge_indices

    # Add warehouse labels if requested
    if create_lineage_labels or create_silo_labels or create_anomaly_labels:
        hetero_data = _add_warehouse_labels_to_hetero_data(
            hetero_data, create_lineage_labels=create_lineage_labels,
            create_silo_labels=create_silo_labels,
            create_anomaly_labels=create_anomaly_labels)

    hetero_data.dataset_name = 'dummy_warehouse_data'
    hetero_data.embedding_dim = EMBEDDING_DIM

    return hetero_data


class RelBenchDataset(InMemoryDataset):
    """PyG Dataset wrapper for RelBench data with warehouse task labels.

    This dataset provides a PyTorch Geometric interface to RelBench data
    with optional warehouse intelligence task labels.

    Args:
        root: Root directory for dataset storage
        dataset_name: Name of the RelBench dataset
        sample_size: Optional sample size for large datasets
        transform: Optional transform to apply to data
        pre_transform: Optional pre-transform to apply to data
    """
    def __init__(
        self,
        root: str,
        dataset_name: str = 'rel-trial',
        sample_size: int | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.sample_size = sample_size

        super().__init__(root, transform, pre_transform)

        try:
            self.data, self.slices = torch.load(self.processed_paths[0],
                                                weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            raise RelBenchError(f"Dataset loading failed: {e}") from e

    @property
    def raw_file_names(self) -> list[str]:
        """Raw file names (RelBench downloads data automatically)."""
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Processed file names."""
        return ['data.pt']

    def download(self) -> None:
        """Download raw data (handled by RelBench)."""

    def process(self) -> None:
        """Process raw data into PyG format."""
        try:
            hetero_data = create_relbench_hetero_data(
                self.dataset_name, sample_size=self.sample_size,
                create_lineage_labels=True, create_silo_labels=True,
                create_anomaly_labels=True, use_dummy_fallback=True)

            # Convert to list format for InMemoryDataset
            data_list = [hetero_data]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise RelBenchError(f"Processing failed: {e}") from e
