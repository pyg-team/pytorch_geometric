"""RelBench dataset integration for PyTorch Geometric.

Converts RelBench datasets to PyG HeteroData with warehouse task labels.

Warehouse Intelligence Labels:
    Labels are generated heuristically from graph structure when external
    labels are not provided. The heuristics are based on:

    - **lineage**: Foreign key depth analysis. Values 0-4 represent data flow
      hierarchy (0=source/no FK refs, 4=deeply derived). Shape: [num_nodes]
    - **silo**: Connected component analysis. Binary 0.0/1.0 where 1.0
      indicates isolated/siloed nodes. Shape: [num_nodes]
    - **anomaly**: Node centrality outlier detection. Binary 0/1 based on
      degree/betweenness centrality statistics. Shape: [num_nodes]

    external_labels: dict[str, Tensor] where keys are 'lineage', 'silo',
    'anomaly' and values are Tensors of shape [num_total_nodes].
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.utils import degree, to_networkx

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
FALLBACK_SILO_PROB = 0.3
FALLBACK_ANOMALY_PROB = 0.1
MAX_LINEAGE_DEPTH = 4
SILO_ISOLATED = 1.0
DUMMY_EDGE_LIMIT = 50

logger = logging.getLogger(__name__)

# Optional: pandas for data processing
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Optional: PyG LLM SentenceTransformer
PYG_LLM_AVAILABLE = False
PyGSentenceTransformer: Any = None
try:
    from torch_geometric.llm.models import \
        SentenceTransformer as PyGSentenceTransformer
    PYG_LLM_AVAILABLE = True
except ImportError:
    pass

# Optional: sentence-transformers library
STSentenceTransformer: Any = None
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import \
        SentenceTransformer as STSentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

# Optional: RelBench for dataset loading and graph conversion
try:
    from relbench.datasets import get_dataset
    from relbench.modeling.graph import make_pkey_fkey_graph
    RELBENCH_AVAILABLE = True
except ImportError:
    RELBENCH_AVAILABLE = False
    get_dataset = None
    make_pkey_fkey_graph = None

# Optional: NetworkX for advanced graph analysis
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


class RelBenchError(Exception):
    """Custom exception for RelBench-related errors."""


class HeuristicLabeler:
    """Generate warehouse task labels from foreign keys and graph structure.

    Labels are generated heuristically:
    - lineage: FK depth (0=source, 1-4=derived depth)
    - silo: Connected component analysis (0=connected, 1=isolated)
    - anomaly: Centrality outliers (0=normal, 1=anomalous)
    """
    def generate_labels(
        self,
        hetero_data: HeteroData,
        db: Any,
        tasks: list[str] | None = None,
    ) -> dict[str, Tensor]:
        """Generate warehouse task labels from database and graph structure.

        Args:
            hetero_data: HeteroData graph to analyze
            db: RelBench database object with FK metadata
            tasks: List of tasks to generate ('lineage', 'silo', 'anomaly')

        Returns:
            Dict mapping task name to label tensor of shape [num_nodes]
        """
        if tasks is None:
            tasks = ['lineage', 'silo', 'anomaly']

        homo_data = hetero_data.to_homogeneous()
        num_nodes = homo_data.num_nodes
        if num_nodes is None:
            raise ValueError("Cannot generate labels: graph has no nodes")

        labels: dict[str, Tensor] = {}
        if 'lineage' in tasks:
            labels['lineage'] = self._generate_lineage_labels(
                hetero_data, db, num_nodes)
        if 'silo' in tasks:
            labels['silo'] = self._generate_silo_labels(homo_data, num_nodes)
        if 'anomaly' in tasks:
            labels['anomaly'] = self._generate_quality_labels(
                homo_data, num_nodes)

        return labels

    def _generate_lineage_labels(
        self,
        hetero_data: HeteroData,
        db: Any,
        num_nodes: int,
    ) -> Tensor:
        """Generate lineage labels based on foreign key depth."""
        lineage_labels = torch.zeros(num_nodes, dtype=torch.long)
        fk_depth_map = self._analyze_foreign_key_depth(db)

        node_idx = 0
        for table_name in hetero_data.node_types:
            table_nodes = hetero_data[table_name].num_nodes
            if table_name in fk_depth_map:
                depth = min(fk_depth_map[table_name], MAX_LINEAGE_DEPTH)
            else:
                depth = MAX_LINEAGE_DEPTH
            lineage_labels[node_idx:node_idx + table_nodes] = depth
            node_idx += table_nodes

        return lineage_labels

    def _generate_silo_labels(self, homo_data: Data, num_nodes: int) -> Tensor:
        """Generate silo labels based on graph connectivity."""
        if homo_data.num_edges == 0 or homo_data.edge_index is None:
            return torch.ones(num_nodes, dtype=torch.float)

        edge_index = homo_data.edge_index
        node_degrees = degree(edge_index[0], num_nodes=num_nodes)

        # Use NetworkX for accurate connected component analysis if available
        if NETWORKX_AVAILABLE:
            try:
                G = to_networkx(homo_data, to_undirected=True)
                components = list(nx.connected_components(G))
                largest_size = max(len(c)
                                   for c in components) if components else 0
                threshold = max(largest_size * SILO_THRESHOLD_RATIO,
                                SILO_MIN_SIZE)

                silo_labels = torch.zeros(num_nodes, dtype=torch.float)
                for comp in components:
                    if len(comp) < threshold:
                        for node in comp:
                            if node < num_nodes:
                                silo_labels[node] = SILO_ISOLATED
                return silo_labels
            except Exception as e:
                logger.warning(f"NetworkX analysis failed: {e}")

        # Fallback: use degree-based heuristic
        deg_threshold = node_degrees.mean() * DEGREE_THRESHOLD_MULTIPLIER
        return (node_degrees < deg_threshold).float()

    def _generate_quality_labels(
        self,
        homo_data: Data,
        num_nodes: int,
    ) -> Tensor:
        """Generate anomaly labels based on node centrality."""
        if homo_data.num_edges == 0 or homo_data.edge_index is None:
            return torch.zeros(num_nodes, dtype=torch.long)

        edge_index = homo_data.edge_index
        node_degrees = degree(edge_index[0], num_nodes=num_nodes)

        # Use NetworkX for centrality analysis if available
        if NETWORKX_AVAILABLE:
            try:
                G = to_networkx(homo_data, to_undirected=True)
                deg_cent = nx.degree_centrality(G)
                bet_cent = nx.betweenness_centrality(
                    G, k=min(CENTRALITY_K_LIMIT, len(G)))

                scores = torch.zeros(num_nodes, dtype=torch.float)
                for node in range(num_nodes):
                    if node in G:
                        scores[node] = (
                            CENTRALITY_DEGREE_WEIGHT * deg_cent.get(node, 0) +
                            CENTRALITY_BETWEENNESS_WEIGHT *
                            bet_cent.get(node, 0))
                threshold = scores.mean() - scores.std()
                return (scores < threshold).long()
            except Exception as e:
                logger.warning(f"NetworkX centrality analysis failed: {e}")

        # Fallback: use degree-based heuristic
        threshold = node_degrees.mean() - node_degrees.std()
        return (node_degrees < threshold).long()

    def _analyze_foreign_key_depth(self, db: Any) -> dict[str, int]:
        """Calculate table depth based on foreign key dependencies."""
        if db is None:
            return {}

        # Build FK dependency graph
        fk_graph: dict[str, list[str]] = {}
        for table_name, table_obj in db.table_dict.items():
            fk_relations = table_obj.fkey_col_to_pkey_table
            fk_graph[table_name] = list(
                fk_relations.values()) if fk_relations else []

        def calc_depth(table: str, visited: set[str], cache: dict[str,
                                                                  int]) -> int:
            if table in cache:
                return cache[table]
            if table in visited:
                return 0  # Cycle detected

            visited.add(table)
            refs = fk_graph.get(table, [])
            if not refs:
                depth = 0
            else:
                depth = 1 + max(
                    (calc_depth(r, visited.copy(), cache)
                     for r in refs if r in fk_graph),
                    default=0,
                )
            cache[table] = depth
            return depth

        cache: dict[str, int] = {}
        return {t: calc_depth(t, set(), cache) for t in db.table_dict}


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
        if PYG_LLM_AVAILABLE:
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
            if PYG_LLM_AVAILABLE and isinstance(self.encoder,
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
        """Process RelBench dataset to HeteroData using make_pkey_fkey_graph.

        Uses RelBench's make_pkey_fkey_graph for graph structure, then adds
        semantic embeddings for warehouse intelligence applications.

        Args:
            dataset_name: Name of the RelBench dataset
            sample_size: Optional sample size for large datasets (not used
                with make_pkey_fkey_graph, kept for API compatibility)

        Returns:
            Tuple of (HeteroData with embeddings, RelBench database)

        Raises:
            RelBenchError: If processing fails
        """
        if not RELBENCH_AVAILABLE or make_pkey_fkey_graph is None:
            raise RelBenchError(
                "RelBench not available. Install relbench package.")

        try:
            # Load RelBench dataset
            dataset = get_dataset(dataset_name)
            db = dataset.get_db()

            # Use RelBench's make_pkey_fkey_graph for graph structure
            hetero_data, _ = make_pkey_fkey_graph(db)

            # Add semantic embeddings if encoder is available
            self._add_semantic_embeddings(hetero_data, db, sample_size)

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

    def _add_semantic_embeddings(self, hetero_data: HeteroData, db: Any,
                                 sample_size: int | None = None) -> None:
        """Add semantic text embeddings to HeteroData nodes.

        This enhances the torch_frame features from make_pkey_fkey_graph
        with semantic embeddings for warehouse intelligence tasks.

        Args:
            hetero_data: HeteroData from make_pkey_fkey_graph
            db: RelBench database object
            sample_size: Optional limit on rows to embed per table
        """
        for table_name in hetero_data.node_types:
            try:
                if table_name not in db.table_dict:
                    continue

                table_obj = db.table_dict[table_name]
                table_df = table_obj.df

                # Sample if requested
                if sample_size is not None:
                    table_df = table_df.head(min(sample_size, len(table_df)))

                # Create text representations and encode
                texts = self._create_text_representations(table_df)
                embeddings = self.encode_texts(texts)

                # Store as semantic_x (keep original x from RelBench)
                hetero_data[table_name].semantic_x = embeddings

            except Exception as e:
                logger.warning(f"Semantic embedding failed for {table_name}: "
                               f"{e}")

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
    """Create HeteroData from a RelBench dataset with warehouse enhancements.

    Labels (on homogeneous graph ``h = data._homo_data_with_labels``):
      - ``lineage_label``: ``LongTensor[num_total_nodes]``, values ``0..4``
        (FK depth, capped). Descriptive mapping (not enforced):
        0=source, 1=staged, 2=transformed, 3=aggregated, 4=derived
      - ``silo_label``:    ``FloatTensor[num_total_nodes]``,
        values ``{0.0, 1.0}``
        (connectivity). Descriptive mapping (not enforced):
        0.0=connected, 1.0=isolated
      - ``anomaly_label``: ``LongTensor[num_total_nodes]`` from heuristics
        (``FloatTensor`` in dummy fallback)

    ``external_labels``: Optional dict with keys in
    {``'lineage'``, ``'silo'``, ``'anomaly'``} mapping to tensors of shape
    ``(num_total_nodes,)`` matching the dtypes above.

    Args:
        dataset_name: Name of the RelBench dataset
        sbert_model: Sentence transformer model for embeddings
        sample_size: Optional sample size for large datasets
        create_lineage_labels: Whether to create lineage labels
            (FK depth analysis)
        create_silo_labels: Whether to create silo labels
            (connectivity/degree analysis)
        create_anomaly_labels: Whether to create anomaly labels
            (graph centrality/degree statistics)
        external_labels: Pre-computed labels to use instead of heuristics
        use_dummy_fallback: Whether to use dummy data if RelBench fails
        batch_size: Batch size for embedding computation

    Returns:
        HeteroData object with embeddings and optional warehouse labels

    Example:
        >>> data = create_relbench_hetero_data(
        ...     'rel-amazon',
        ...     create_lineage_labels=True,
        ...     create_silo_labels=True,
        ...     create_anomaly_labels=True,
        ... )
        >>> h = data._homo_data_with_labels
        >>> (h.lineage_label.dtype, h.silo_label.dtype)  # doctest: +SKIP
        (torch.int64, torch.float32)

        >>> import torch
        >>> N = 1000
        >>> external_labels = {
        ...     'lineage': torch.randint(0, 5, (N,), dtype=torch.long),
        ...     'silo': torch.randint(0, 2, (N,), dtype=torch.float),
        ...     'anomaly': torch.randint(0, 2, (N,), dtype=torch.long),
        ... }
        >>> data = create_relbench_hetero_data(
        ...     'rel-amazon',
        ...     create_lineage_labels=True,
        ...     create_silo_labels=True,
        ...     create_anomaly_labels=True,
        ...     external_labels=external_labels,
        ... )  # doctest: +SKIP

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
