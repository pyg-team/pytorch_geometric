"""RelBench integration utilities for PyTorch Geometric.

Provides utilities for converting RelBench datasets to PyG HeteroData objects
with semantic embeddings and warehouse-specific enhancements.

Complements examples/rdl.py with G-Retriever preparation and warehouse tasks.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch

from torch_geometric.data import HeteroData, InMemoryDataset

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Define types for type checking
from typing import Type

# Import SentenceTransformer types
PyGSentenceTransformer: Optional[Type[Any]] = None
STSentenceTransformer: Optional[Type[Any]] = None

try:
    from torch_geometric.nn.nlp import \
        SentenceTransformer as PyGSentenceTransformer  # noqa: E501

    PYG_NLP_AVAILABLE = True
except ImportError:
    PYG_NLP_AVAILABLE = False
    PyGSentenceTransformer = None

try:
    from sentence_transformers import \
        SentenceTransformer as STSentenceTransformer  # noqa: E501

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    STSentenceTransformer = None

# Check if any SBERT implementation is available
if not PYG_NLP_AVAILABLE and not SENTENCE_TRANSFORMERS_AVAILABLE:
    warnings.warn(
        'Neither PyG NLP nor sentence-transformers available. '
        'Install PyG 2.6.0+ or sentence-transformers for support',
        stacklevel=2,
    )

try:
    from relbench.datasets import get_dataset

    RELBENCH_AVAILABLE = True
except ImportError:
    RELBENCH_AVAILABLE = False
    warnings.warn(
        'RelBench not available. Install with: pip install relbench[full]',
        stacklevel=2,
    )


class RelBenchProcessor:
    """Utility for converting RelBench datasets to PyG HeteroData format."""
    def __init__(self, sbert_model: str = 'all-MiniLM-L6-v2') -> None:
        """Initialize processor with SBERT model."""
        if not RELBENCH_AVAILABLE:
            raise ImportError('RelBench is required. Install with: '
                              'pip install relbench[full]')

        if not (PYG_NLP_AVAILABLE or SENTENCE_TRANSFORMERS_AVAILABLE):
            raise ImportError(
                'Neither PyG NLP nor sentence-transformers available. '
                'Install PyG 2.6.0+ or sentence-transformers for embedding '
                'support')

        self.sbert_model_name = sbert_model
        self._sbert_model: Optional[Any] = None
        self._embedding_dim: Optional[int] = None

    @property
    def sbert_model(self) -> Any:
        """Lazy loading of SBERT model to speed up import time."""
        if self._sbert_model is None:
            if PYG_NLP_AVAILABLE and PyGSentenceTransformer is not None:
                self._sbert_model = PyGSentenceTransformer(
                    self.sbert_model_name)
            elif (SENTENCE_TRANSFORMERS_AVAILABLE
                  and STSentenceTransformer is not None):
                self._sbert_model = STSentenceTransformer(
                    self.sbert_model_name)
            else:
                raise ImportError(
                    'Neither PyG NLP nor sentence-transformers available')
        return self._sbert_model

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension from SBERT model."""
        if self._embedding_dim is None:
            self._embedding_dim = (
                self.sbert_model.get_sentence_embedding_dimension())
            # Guard against invalid embedding dimensions
            assert self._embedding_dim > 0, (
                f'Invalid embedding dimension: {self._embedding_dim}')
            assert isinstance(self._embedding_dim,
                              int), (f'Embedding dimension must be int, got '
                                     f'{type(self._embedding_dim)}')
        return self._embedding_dim

    def process_dataset(
        self,
        dataset_name: str,
        sample_size: Optional[int] = None,
        add_warehouse_labels: bool = False,
        batch_size: int = 64,
    ) -> HeteroData:
        """Process RelBench dataset into unified record HeteroData.

        Returns HeteroData with all records as unified 'record' node type.
        """
        dataset = get_dataset(name=dataset_name, download=True)
        db = dataset.get_db()
        hetero_data = HeteroData()
        all_embeddings = []
        all_table_ids = []
        all_record_ids = []
        table_name_to_id = {}

        for table_idx, (table_name,
                        table_obj) in enumerate(db.table_dict.items()):
            table_df = table_obj.df
            table_name_to_id[table_name] = table_idx

            # Sample data if requested
            if sample_size is not None:
                table_df = table_df.head(min(sample_size, len(table_df)))

            # Create semantic text representations
            node_texts = self._create_node_texts(table_name, table_df)

            # Generate SBERT embeddings
            if node_texts:
                embeddings = self._generate_embeddings(node_texts, batch_size)
                all_embeddings.append(embeddings)

                # Track which table each record belongs to
                table_ids = torch.full((len(embeddings), ), table_idx,
                                       dtype=torch.long)
                all_table_ids.append(table_ids)

                # Track original record IDs within table
                record_ids = torch.arange(len(embeddings), dtype=torch.long)
                all_record_ids.append(record_ids)

        # Combine all records into unified node space
        if all_embeddings:
            hetero_data['record'].x = torch.cat(all_embeddings, dim=0)
            hetero_data['record'].table_id = torch.cat(all_table_ids, dim=0)
            hetero_data['record'].record_id = torch.cat(all_record_ids, dim=0)
            hetero_data['record'].num_nodes = hetero_data['record'].x.shape[0]

            # Store table metadata
            hetero_data['record'].table_names = list(table_name_to_id.keys())
            hetero_data['record'].table_name_to_id = table_name_to_id

            # Add warehouse task labels if requested
            if add_warehouse_labels:
                self._add_warehouse_labels_unified(hetero_data['record'],
                                                   db=db,
                                                   use_dummy_fallback=True)

        # Create edges based on RelBench schema
        self._create_edges(hetero_data, db)

        return hetero_data

    def _create_node_texts(self, table_name: str,
                           table_df: 'pd.DataFrame') -> List[str]:
        """Create text representations for SBERT embedding."""
        node_texts = []
        for _, row in table_df.iterrows():
            text_parts = [f'Table: {table_name}']
            for col, val in row.items():
                val_str = str(val) if val is not None else 'NULL'
                val_str = val_str[:100] if len(val_str) > 100 else val_str
                text_parts.append(f'{col}: {val_str}')
            node_texts.append('. '.join(text_parts))

        return node_texts

    def _add_warehouse_labels_unified(
            self, node_store: Any, db: Any,
            use_dummy_fallback: bool = False) -> None:
        """Add lineage, silo, and anomaly labels to unified records."""
        num_nodes = node_store.num_nodes
        table_ids = node_store.table_id
        table_names = node_store.table_names

        # Initialize label tensors
        lineage_labels = torch.zeros(num_nodes, dtype=torch.long)
        silo_labels = torch.zeros(num_nodes, dtype=torch.long)
        anomaly_labels = torch.zeros(num_nodes, dtype=torch.long)

        # Apply record-level inference for each table
        for table_idx, table_name in enumerate(table_names):
            # Get records from this table
            table_mask = table_ids == table_idx
            table_record_count = table_mask.sum().item()

            if table_record_count > 0:
                # Get table-level inference
                table_lineage = self._infer_record_lineage(
                    table_name, db, table_record_count)
                table_silo = self._infer_record_silo(table_name, db,
                                                     table_record_count)
                table_anomaly = self._infer_record_anomaly(
                    table_name, db, table_record_count)

                # Apply to records
                lineage_labels[table_mask] = table_lineage
                silo_labels[table_mask] = table_silo
                anomaly_labels[table_mask] = table_anomaly

        # Store labels
        node_store.lineage_label = lineage_labels
        node_store.silo_label = silo_labels
        node_store.anomaly_label = anomaly_labels

    def _generate_embeddings(self, texts: List[str],
                             batch_size: int = 64) -> torch.Tensor:
        """Generate SBERT embeddings with batch processing."""
        if not texts:
            return torch.empty(0, self.embedding_dim, dtype=torch.float)

        # Process in batches to handle large datasets and GPU memory limits
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            try:
                # Try with current batch size
                batch_embeddings = self.sbert_model.encode(batch_texts)

                # Handle different return types from PyG NLP vs
                # sentence-transformers
                if not isinstance(batch_embeddings, torch.Tensor):
                    batch_embeddings = torch.tensor(batch_embeddings,
                                                    dtype=torch.float)

                all_embeddings.append(batch_embeddings)

            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and batch_size > 1:
                    # GPU OOM: reduce batch size and retry
                    warnings.warn(
                        f'GPU OOM with batch_size={batch_size}. '
                        f'Reducing to {batch_size // 2}',
                        stacklevel=2,
                    )

                    # Process with smaller batches
                    smaller_batch_size = max(1, batch_size // 2)
                    for j in range(i, min(i + batch_size, len(texts)),
                                   smaller_batch_size):
                        small_batch = texts[j:j + smaller_batch_size]
                        small_embeddings = self.sbert_model.encode(small_batch)

                        if not isinstance(small_embeddings, torch.Tensor):
                            small_embeddings = torch.tensor(
                                small_embeddings, dtype=torch.float)
                        all_embeddings.append(small_embeddings)
                else:
                    raise e

        return torch.cat(all_embeddings, dim=0)

    def _add_warehouse_labels(
        self,
        node_store: Any,
        num_nodes: int,
        table_name: Optional[str] = None,
        db: Any = None,
        create_lineage_labels: bool = True,
        create_silo_labels: bool = True,
        create_anomaly_labels: bool = True,
        use_dummy_fallback: bool = False,
    ) -> None:
        """Add warehouse task labels with 'Ready-for-Real-Data' pattern.

        Precedence order: Real Data > Structural Inference > Dummy
        Fallback > None

        Args:
            node_store: PyG node store to add labels to
            num_nodes: Number of nodes
            table_name: Name of the table (for inference)
            db: RelBench database object (for real data lookup)
            create_lineage_labels: Whether to create ETL lineage labels
            create_silo_labels: Whether to create silo detection labels
            create_anomaly_labels: Whether to create anomaly detection labels
            use_dummy_fallback: Whether to use dummy data as last resort
        """
        # ETL Lineage Labels
        if create_lineage_labels:
            lineage_labels = self._get_lineage_labels(table_name, db,
                                                      num_nodes,
                                                      use_dummy_fallback)
            node_store.lineage_label = lineage_labels

        # Silo Detection Labels
        if create_silo_labels:
            silo_labels = self._get_silo_labels(table_name, db, num_nodes,
                                                use_dummy_fallback)
            node_store.silo_label = silo_labels

        # Anomaly Detection Labels
        if create_anomaly_labels:
            anomaly_labels = self._get_anomaly_labels(table_name, db,
                                                      num_nodes,
                                                      use_dummy_fallback)
            node_store.anomaly_label = anomaly_labels

    def _get_lineage_labels(
        self,
        table_name: Optional[str],
        db: Any,
        num_nodes: int,
        use_dummy_fallback: bool,
    ) -> Optional[torch.Tensor]:
        """Get ETL lineage labels with precedence: real > inferred >
        dummy > None.
        """
        # Method 1: Check for real lineage data
        if table_name is not None and self._has_real_lineage(db, table_name):
            return self._load_real_lineage(db, table_name)

        # Method 2: Structural inference
        elif (table_name is not None
              and self._can_infer_lineage(table_name, db)):
            return self._infer_lineage_from_structure(table_name, db)

        # Method 3: Dummy fallback (with warning)
        elif use_dummy_fallback:
            import warnings

            warnings.warn(
                f'Using synthetic ETL lineage labels for {table_name}. '
                'These are placeholders for demonstration only.',
                UserWarning,
                stacklevel=2,
            )
            return torch.randint(0, 3, (num_nodes, ))

        # Method 4: None (no labels available)
        else:
            return None

    def _get_silo_labels(
        self,
        table_name: Optional[str],
        db: Any,
        num_nodes: int,
        use_dummy_fallback: bool,
    ) -> Optional[torch.Tensor]:
        """Get silo detection labels with precedence: real > inferred >
        dummy > None.
        """
        # Method 1: Check for real silo data
        if table_name is not None and self._has_real_silo_data(db, table_name):
            return self._load_real_silo_labels(db, table_name)

        # Method 2: Structural inference (always available)
        elif table_name is not None:
            return self._infer_silo_from_connectivity(table_name, db,
                                                      num_nodes)
        else:
            return None

    def _get_anomaly_labels(
        self,
        table_name: Optional[str],
        db: Any,
        num_nodes: int,
        use_dummy_fallback: bool,
    ) -> Optional[torch.Tensor]:
        """Get anomaly detection labels with precedence: real > inferred >
        dummy > None.
        """
        # Method 1: Check for real anomaly data
        if table_name is not None and self._has_real_anomaly_data(
                db, table_name):
            return self._load_real_anomaly_labels(db, table_name)

        # Method 2: Statistical inference
        elif table_name is not None and self._can_infer_anomalies(
                table_name, db):
            return self._infer_anomalies_from_statistics(table_name, db)

        # Method 3: Dummy fallback (with warning)
        elif use_dummy_fallback:
            import warnings

            warnings.warn(
                f'Using synthetic anomaly detection labels for {table_name}. '
                'These are placeholders for demonstration only.',
                UserWarning,
                stacklevel=2,
            )
            return torch.randint(0, 2, (num_nodes, ))

        # Method 4: None (no labels available)
        else:
            return None

    # Real data checking methods
    def _has_real_lineage(self, db: Any, table_name: Optional[str]) -> bool:
        """Check if real ETL lineage data is available.

        Args:
            db: RelBench database object
            table_name: Name of the table to check

        Returns:
            True if real lineage data is available, False otherwise
        """
        return (hasattr(db, 'lineage_metadata')
                and table_name in getattr(db, 'lineage_metadata', {})
                and 'etl_stages' in db.lineage_metadata[table_name])

    def _has_real_silo_data(self, db: Any, table_name: Optional[str]) -> bool:
        """Check if real silo detection data is available.

        Args:
            db: RelBench database object
            table_name: Name of the table to check

        Returns:
            True if real silo data is available, False otherwise
        """
        return hasattr(db, 'silo_metadata') and table_name in getattr(
            db, 'silo_metadata', {})

    def _has_real_anomaly_data(self, db: Any,
                               table_name: Optional[str]) -> bool:
        """Check if real anomaly detection data is available.

        Args:
            db: RelBench database object
            table_name: Name of the table to check

        Returns:
            True if real anomaly data is available, False otherwise
        """
        return hasattr(db, 'anomaly_metadata') and table_name in getattr(
            db, 'anomaly_metadata', {})

    # Real data loading methods (placeholders for when real data is available)
    def _load_real_lineage(self, db: Any, table_name: str) -> torch.Tensor:
        """Load real ETL lineage labels."""
        return torch.tensor(db.lineage_metadata[table_name]['etl_stages'],
                            dtype=torch.long)

    def _load_real_silo_labels(self, db: Any, table_name: str) -> torch.Tensor:
        """Load real silo detection labels."""
        return torch.tensor(db.silo_metadata[table_name]['silo_labels'],
                            dtype=torch.long)

    def _load_real_anomaly_labels(self, db: Any,
                                  table_name: str) -> torch.Tensor:
        """Load real anomaly detection labels."""
        return torch.tensor(db.anomaly_metadata[table_name]['anomaly_labels'],
                            dtype=torch.long)

    # Inference capability checking methods
    def _can_infer_lineage(self, table_name: Optional[str], db: Any) -> bool:
        """Check if we can infer lineage from table structure."""
        return (table_name is not None and db is not None
                and table_name in db.table_dict)

    def _can_infer_anomalies(self, table_name: Optional[str], db: Any) -> bool:
        """Check if we can infer anomalies from statistics."""
        if not (table_name and db and table_name in db.table_dict):
            return False

        table_df = db.table_dict[table_name].df
        # Check if table has numeric columns for statistical analysis
        numeric_cols = table_df.select_dtypes(include=['number']).columns
        return len(numeric_cols) > 0

    # Structural inference methods
    def _infer_lineage_from_structure(self, table_name: str,
                                      db: Any) -> torch.Tensor:
        """Generate lineage labels using table metadata."""
        table_df = db.table_dict[table_name].df

        # Count foreign key columns
        fk_count = len([
            col for col in table_df.columns
            if col.endswith('_id') or col.endswith('Id')
        ])

        # Check for aggregated columns
        has_aggregated = any('total' in col.lower() or 'sum' in col.lower()
                             or 'avg' in col.lower() or 'count' in col.lower()
                             for col in table_df.columns)

        # Check for timestamp columns (indicates processing)
        has_timestamps = any(
            'date' in col.lower() or 'time' in col.lower()
            or 'created' in col.lower() or 'updated' in col.lower()
            for col in table_df.columns)

        num_nodes = len(table_df)

        # Inference logic:
        if fk_count >= 3 or has_aggregated:
            # Likely mart/target table (joins multiple sources)
            return torch.full((num_nodes, ), 2, dtype=torch.long)  # 2 = target
        elif fk_count == 0 and not has_timestamps:
            # Likely source table (reference data)
            return torch.zeros(num_nodes, dtype=torch.long)  # 0 = source
        else:
            # Likely intermediate table
            return torch.ones(num_nodes, dtype=torch.long)  # 1 = intermediate

    def _infer_silo_from_connectivity(self, table_name: str, db: Any,
                                      num_nodes: int) -> torch.Tensor:
        """Generate silo labels using connectivity information."""
        # Count connections to other tables
        connections = 0

        # Check outgoing FK connections
        if hasattr(db, 'fkey_dict') and table_name in db.fkey_dict:
            connections += len(db.fkey_dict[table_name])

        # Check incoming FK connections
        if hasattr(db, 'fkey_dict'):
            for other_table, fkeys in db.fkey_dict.items():
                if other_table != table_name:
                    for fkey in fkeys:
                        if fkey.get('ref_table') == table_name:
                            connections += 1

        # Silo detection logic: isolated if <= 1 connection
        if connections <= 1:
            return torch.ones(num_nodes, dtype=torch.long)  # 1 = isolated/silo
        else:
            return torch.zeros(num_nodes, dtype=torch.long)  # 0 = connected

    def _infer_anomalies_from_statistics(self, table_name: str,
                                         db: Any) -> torch.Tensor:
        """Generate anomaly labels using statistical methods."""
        table_df = db.table_dict[table_name].df
        num_nodes = len(table_df)

        anomaly_labels = torch.zeros(num_nodes, dtype=torch.long)

        # Find statistical outliers in numeric columns
        numeric_cols = table_df.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            values = table_df[col].dropna()
            if len(values) > 0:
                # Use IQR method for outlier detection
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1

                if IQR > 0:  # Avoid division by zero
                    # Mark outliers as anomalies
                    outlier_mask = (values < (Q1 - 1.5 * IQR)) | (values > (
                        Q3 + 1.5 * IQR))

                    # Update anomaly labels for outlier rows
                    outlier_indices = table_df[col].index[table_df[col].isin(
                        values[outlier_mask])]
                    anomaly_labels[outlier_indices] = 1

        return anomaly_labels

    def _infer_record_lineage(self, table_name: str, db: Any,
                              num_records: int) -> torch.Tensor:
        """Generate lineage labels for individual records."""
        table_df = db.table_dict[table_name].df

        # Count foreign keys in this table
        fk_count = 0
        if hasattr(db, 'fkey_dict') and table_name in db.fkey_dict:
            fk_count = len(db.fkey_dict[table_name])

        # Check for aggregated columns (sum, count, avg patterns)
        has_aggregated = any(col.lower().startswith(('sum_', 'count_', 'avg_',
                                                     'total_'))
                             for col in table_df.columns)

        # Inference logic applied to all records in table
        if fk_count >= 3 or has_aggregated:
            # Target/mart records
            return torch.full((num_records, ), 2, dtype=torch.long)
        elif fk_count == 0:
            # Source records
            return torch.zeros(num_records, dtype=torch.long)
        else:
            # Intermediate records
            return torch.ones(num_records, dtype=torch.long)

    def _infer_record_silo(self, table_name: str, db: Any,
                           num_records: int) -> torch.Tensor:
        """Generate silo labels for individual records."""
        # Check table connectivity
        has_connections = False
        if hasattr(db, 'fkey_dict'):
            has_connections = (table_name in db.fkey_dict
                               and len(db.fkey_dict[table_name]) > 0)

        # All records in connected tables are connected
        if has_connections:
            return torch.zeros(num_records, dtype=torch.long)  # 0 = connected
        else:
            return torch.ones(num_records, dtype=torch.long)  # 1 = isolated

    def _infer_record_anomaly(self, table_name: str, db: Any,
                              num_records: int) -> torch.Tensor:
        """Generate anomaly labels for individual records."""
        # Use existing statistical inference but return per-record labels
        return self._infer_anomalies_from_statistics(table_name, db)

    def _create_edges(self, hetero_data: HeteroData, db: Any) -> None:
        """Create graph edges between records."""
        if 'record' not in hetero_data.node_types:
            warnings.warn('No record nodes found for edge creation',
                          stacklevel=2)
            return

        # Create FK-based edges between records
        self._add_fk_edges_unified(hetero_data, db)

        # Add value similarity edges for anomaly detection
        self._add_value_similarity_edges(hetero_data)

    def _add_fk_edges_unified(self, hetero_data: HeteroData, db: Any) -> None:
        """Add relationship-based edges between records."""
        if not hasattr(db, 'fkey_dict') or not db.fkey_dict:
            return

        record_store = hetero_data['record']
        table_ids = record_store.table_id
        table_names = record_store.table_names

        src_indices = []
        dst_indices = []

        # Process FK relationships
        for src_table_name, fk_list in db.fkey_dict.items():
            if src_table_name not in table_names:
                continue

            src_table_id = record_store.table_name_to_id[src_table_name]
            src_mask = table_ids == src_table_id
            src_node_indices = torch.where(src_mask)[0]

            for fk_info in fk_list:
                dst_table_name = fk_info.get('table', fk_info.get('dst_table'))
                if dst_table_name not in table_names:
                    continue

                dst_table_id = record_store.table_name_to_id[dst_table_name]
                dst_mask = table_ids == dst_table_id
                dst_node_indices = torch.where(dst_mask)[0]

                # Create edges between records (simplified)
                if len(src_node_indices) > 0 and len(dst_node_indices) > 0:
                    # Sample edges for demonstration
                    num_edges = min(len(src_node_indices),
                                    len(dst_node_indices), 50)
                    edge_src = src_node_indices[:num_edges]
                    edge_dst = dst_node_indices[:num_edges]

                    src_indices.extend(edge_src.tolist())
                    dst_indices.extend(edge_dst.tolist())

        # Add FK edges
        if src_indices and dst_indices:
            fk_edge_index = torch.tensor([src_indices, dst_indices],
                                         dtype=torch.long)
            hetero_data['record', 'fk_relation',
                        'record'].edge_index = fk_edge_index

    def _add_value_similarity_edges(self, hetero_data: HeteroData) -> None:
        """Add similarity-based edges between nodes."""
        if 'record' not in hetero_data.node_types:
            return

        record_store = hetero_data['record']
        embeddings = record_store.x

        if embeddings.shape[0] < 2:
            return

        # Compute cosine similarity between embeddings
        from torch.nn.functional import cosine_similarity

        # For efficiency, only connect most similar pairs
        num_nodes = embeddings.shape[0]
        max_edges = min(num_nodes * 5, 1000)  # Limit edges for performance

        src_indices = []
        dst_indices = []

        # Sample node pairs and compute similarities
        import random

        random.seed(42)  # Deterministic for testing

        for _ in range(max_edges):
            i = random.randint(0, num_nodes - 1)
            j = random.randint(0, num_nodes - 1)

            if i != j:
                # Compute similarity
                sim = cosine_similarity(embeddings[i:i + 1],
                                        embeddings[j:j + 1])

                # Connect if similarity is high (threshold = 0.8)
                if sim.item() > 0.8:
                    src_indices.append(i)
                    dst_indices.append(j)

        # Add similarity edges
        if src_indices and dst_indices:
            sim_edge_index = torch.tensor([src_indices, dst_indices],
                                          dtype=torch.long)
            hetero_data['record', 'similar_to',
                        'record'].edge_index = sim_edge_index

    def _add_sample_edges(self, hetero_data: HeteroData, src: str, rel: str,
                          dst: str) -> None:
        """Add sample edges between node types."""
        import warnings

        warnings.warn(
            f'Using synthetic/dummy edges between {src} and {dst}. '
            'These are random connections for demonstration only and '
            'should not be used for research.',
            UserWarning,
            stacklevel=2,
        )

        src_nodes = hetero_data[src].num_nodes
        dst_nodes = hetero_data[dst].num_nodes

        if src_nodes > 0 and dst_nodes > 0:
            # Create sample edges (in practice, these would be based on actual
            # foreign keys)
            num_edges = min(20, src_nodes, dst_nodes)
            edge_index = torch.randint(0, min(src_nodes, dst_nodes),
                                       (2, num_edges))

            hetero_data[src, rel, dst].edge_index = edge_index
            hetero_data[dst, f'rev_{rel}', src].edge_index = edge_index.flip(0)

    def _discover_real_relationships(
            self, db: Any,
            node_types: List[str]) -> List[Tuple[str, str, str]]:
        """Extract relationships from RelBench metadata."""
        relationships = []

        # Method 1: Try to use edge_df if available
        if (hasattr(db, 'edge_df') and hasattr(db.edge_df, 'empty')
                and not db.edge_df.empty):
            for _, row in db.edge_df.iterrows():
                src = row.get('src_table', row.get('source_table'))
                dst = row.get('dst_table', row.get('target_table'))
                edge_type = row.get('edge_type', 'fk_relation')

                if src in node_types and dst in node_types:
                    relationships.append((src, edge_type, dst))

        # Method 2: Try to use fkey_dict if available
        elif hasattr(db, 'fkey_dict') and db.fkey_dict:
            for table_name, fkeys in db.fkey_dict.items():
                if table_name in node_types:
                    for fkey_info in fkeys:
                        ref_table = fkey_info.get('table',
                                                  fkey_info.get('ref_table'))
                        column = fkey_info.get('column', 'unknown')

                        if ref_table and ref_table in node_types:
                            rel_name = f'fk_{column}'
                            relationships.append(
                                (table_name, rel_name, ref_table))

        # Method 3: Try to infer from column names (basic heuristic)
        else:
            relationships = self._infer_relationships_from_columns(
                db, node_types)

        return relationships

    def _infer_relationships_from_columns(
            self, db: Any,
            node_types: List[str]) -> List[Tuple[str, str, str]]:
        """Extract relationships from column patterns."""
        relationships = []

        for table_name, table_obj in db.table_dict.items():
            if table_name not in node_types:
                continue

            table_df = table_obj.df

            # Look for columns that end with _id or Id
            for col in table_df.columns:
                if col.endswith('_id') or col.endswith('Id'):
                    # Try to find the referenced table
                    potential_table = (col.replace('_id',
                                                   '').replace('Id',
                                                               '').lower())

                    # Check for exact match or plural form
                    for candidate in node_types:
                        if (candidate.lower() == potential_table
                                or candidate.lower() == potential_table + 's'
                                or candidate.lower() + 's' == potential_table):
                            rel_name = f'references_{col}'
                            relationships.append(
                                (table_name, rel_name, candidate))
                            break

        return relationships

    def _add_real_edges_from_fk(
        self,
        hetero_data: HeteroData,
        src_table: str,
        rel_name: str,
        dst_table: str,
        db: Any,
    ) -> None:
        """Create edges using available relationship data."""
        try:
            src_df = db.table_dict[src_table].df
            dst_df = db.table_dict[dst_table].df

            # Extract FK column name from relation name
            fk_column = rel_name.replace('fk_', '').replace('references_', '')

            # Find the FK column in source table
            fk_col = None
            for col in src_df.columns:
                if col.lower() == fk_column.lower() or col.lower().endswith(
                        fk_column.lower()):
                    fk_col = col
                    break

            if fk_col is None:
                print(f"⚠️  FK column '{fk_column}' not found in {src_table}")
                return

            # Create real edges based on FK values
            edge_list = []
            src_fk_values = src_df[fk_col].dropna()

            # Assume destination table has an 'id' column or use index
            if 'id' in dst_df.columns:
                dst_pk_values = dst_df['id']
            elif f'{dst_table}_id' in dst_df.columns:
                dst_pk_values = dst_df[f'{dst_table}_id']
            else:
                # Use index as PK
                dst_pk_values = dst_df.index

            # Match FK values to PK values
            for src_idx, fk_val in enumerate(src_fk_values):
                matching_dst = dst_pk_values[dst_pk_values == fk_val]
                if len(matching_dst) > 0:
                    # Get the first matching destination index
                    dst_idx = matching_dst.index.tolist()[0]
                    edge_list.append([src_idx, dst_idx])

            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                hetero_data[src_table, rel_name,
                            dst_table].edge_index = edge_index
                hetero_data[dst_table, f'rev_{rel_name}',
                            src_table].edge_index = edge_index.flip(0)

        except Exception as e:
            # Fallback to sample edges with warning
            import warnings

            warnings.warn(
                f'Error creating real edges for {src_table} -> '
                f'{dst_table}: {e}. Falling back to sample edges.',
                UserWarning,
                stacklevel=2,
            )
            self._add_sample_edges(hetero_data, src_table, rel_name, dst_table)


def create_relbench_hetero_data(
    dataset_name: str,
    sbert_model: str = 'all-MiniLM-L6-v2',
    sample_size: Optional[int] = None,
    add_warehouse_labels: bool = False,
    create_lineage_labels: bool = False,
    create_silo_labels: bool = False,
    create_anomaly_labels: bool = False,
    use_dummy_fallback: bool = False,
    batch_size: int = 64,
) -> HeteroData:
    """Create HeteroData from RelBench dataset.

    TODO: Add support for custom edge types and weights
    TODO: Implement lineage tracking
    """
    processor = RelBenchProcessor(sbert_model)

    # Handle legacy parameter
    if add_warehouse_labels and not (create_lineage_labels
                                     or create_silo_labels
                                     or create_anomaly_labels):
        create_lineage_labels = create_silo_labels = create_anomaly_labels = (
            True)

    return processor.process_dataset(
        dataset_name,
        sample_size,
        add_warehouse_labels=(create_lineage_labels or create_silo_labels
                              or create_anomaly_labels),
        batch_size=batch_size,
    )


def get_warehouse_task_info() -> Dict[str, Dict[str, Any]]:
    """Get warehouse task metadata for lineage, silo, and anomaly detection."""
    return {
        'lineage': {
            'num_classes':
            3,
            'classes': ['source', 'intermediate', 'target'],
            'description':
            'ETL lineage detection - classify nodes by their '
            'position in data flow',
            'data_availability':
            'structural_inference',
            'notes':
            'Uses structural heuristics based on FK count and '
            'column patterns. Real ETL logs preferred when available.',
        },
        'silo': {
            'num_classes':
            2,
            'classes': ['connected', 'isolated'],
            'description':
            'Data silo detection - identify disconnected '
            'data components',
            'data_availability':
            'real_data',
            'notes':
            'Based on connectivity information '
            'from RelBench metadata.',
        },
        'anomaly': {
            'num_classes': 2,
            'classes': ['normal', 'anomaly'],
            'description': 'Anomaly detection - identify unusual patterns '
            'in data warehouse',
            'data_availability': 'statistical_inference',
            'notes': 'Based on statistical methods '
            'applied to numeric columns.',
        },
    }


class RelBenchDataset(InMemoryDataset):
    """PyG Dataset wrapper for RelBench data with warehouse task labels.

    This dataset follows PyG conventions and provides a standard interface
    for loading RelBench data as HeteroData objects with semantic embeddings.
    """
    def __init__(
        self,
        root: str,
        dataset_name: str = 'rel-trial',
        sample_size: Optional[int] = None,
        add_warehouse_labels: bool = True,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
    ) -> None:
        """Initialize RelBench dataset.

        Args:
            root: Root directory to store the dataset
            dataset_name: RelBench dataset name (default: 'rel-trial')
            sample_size: Limit number of records per table
            add_warehouse_labels: Add lineage/silo/anomaly labels
            transform: Transform to apply to data objects
            pre_transform: Pre-transform to apply during processing
        """
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        self.add_warehouse_labels = add_warehouse_labels

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []  # RelBench downloads data automatically

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def download(self) -> None:
        pass  # RelBench handles downloads automatically

    def process(self) -> None:
        """Process RelBench data into PyG format."""
        # Use the processor to create HeteroData
        data = create_relbench_hetero_data(
            self.dataset_name,
            sample_size=self.sample_size,
            add_warehouse_labels=self.add_warehouse_labels,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save as InMemoryDataset format
        data_list = [data]
        collated_data, slices = self.collate(data_list)
        torch.save((collated_data, slices), self.processed_paths[0])


def prepare_for_gretriever(
        hetero_data: HeteroData) -> Tuple[HeteroData, Dict[str, Any]]:
    """Prepare RelBench HeteroData for G-Retriever training.

    Enhances HeteroData with G-Retriever-specific attributes and metadata.

    Args:
        hetero_data: HeteroData object from RelBench integration

    Returns:
        Tuple of (enhanced_hetero_data, metadata_dict)
    """
    metadata = {
        'embedding_dim': getattr(hetero_data, 'embedding_dim', 384),
        'node_types': list(hetero_data.node_types),
        'edge_types': list(hetero_data.edge_types),
        'warehouse_tasks': ['lineage', 'silo', 'anomaly'],
        'recommended_qa_pairs': get_warehouse_task_info(),
        'conversion_ready': True,
    }

    # Add G-Retriever specific attributes
    hetero_data.gretriever_ready = True
    hetero_data.embedding_type = 'sbert'  # Indicates SBERT embeddings
    hetero_data.warehouse_enhanced = True

    return hetero_data, metadata


# Backward compatibility aliases
RelBenchHeteroDataProcessor = RelBenchProcessor
create_hetero_data_from_relbench = create_relbench_hetero_data
