import re
from abc import ABC
from typing import Any

import torch
from torch import Tensor

from torch_geometric.data.database_graph_store import DatabaseGraphStore
from torch_geometric.sampler.database_sampler import DatabaseSampler

_CYPHER_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Neo4jSampler(DatabaseSampler, ABC):
    r"""Abstract base for Neo4j-backed samplers.

    Handles Neo4j-specific init shared across variants: Cypher identifier
    validation for ``node_label`` / ``rel_type``, and storage of common
    attributes (:attr:`nodeid_property`, :attr:`node_label`, :attr:`rel_type`,
    :attr:`profile`). Subclasses that depend on APOC should call
    :meth:`_probe_apoc` from their own ``__init__``.

    Subclasses must implement the relevant query builder, params builder,
    and decoder hooks defined on :class:`DatabaseSampler`.

    Args:
        graph_store (DatabaseGraphStore): Neo4j-backed graph store.
        node_label (str, optional): Node label filter (e.g. ``"User"``).
            :obj:`None` matches all labels. (default: :obj:`None`)
        rel_type (str, optional): Relationship type filter (e.g. ``"KNOWS"``).
            :obj:`None` matches all types. (default: :obj:`None`)
        profile (bool): Prepend ``PROFILE`` to the Cypher query for
            execution-plan introspection. (default: :obj:`False`)
        **kwargs: Forwarded to :class:`DatabaseSampler` (e.g. ``is_hetero``).
    """
    def __init__(
        self,
        graph_store: DatabaseGraphStore,
        node_label: str | None = None,
        rel_type: str | None = None,
        profile: bool = False,
        **kwargs,
    ) -> None:
        self._validate_cypher_ident("node_label", node_label)
        self._validate_cypher_ident("rel_type", rel_type)

        self.nodeid_property = graph_store.nodeid_property
        self.node_label = node_label
        self.rel_type = rel_type
        self.profile = profile

        super().__init__(graph_store, **kwargs)

    @staticmethod
    def _validate_cypher_ident(name: str, value: str | None) -> None:
        r"""Reject ``value`` if set and not a valid Cypher identifier.

        Blocks Cypher injection via label / relationship type strings.
        Raises :class:`ValueError` on a bad identifier.
        """
        if value is not None and not _CYPHER_IDENT_RE.match(value):
            raise ValueError(
                f"{name} must be a valid Cypher identifier matching "
                f"{_CYPHER_IDENT_RE.pattern!r}; got {value!r}.")

    @staticmethod
    def _probe_apoc(graph_store: DatabaseGraphStore) -> None:
        r"""Verify APOC is installed in the target Neo4j database.

        Call from subclass ``__init__`` when the sampling query uses
        ``apoc.*`` procedures. Raises :class:`RuntimeError` if missing.
        """
        check = getattr(graph_store, "apoc_available", None)
        if check is not None and not check():
            raise RuntimeError(
                f"{__class__.__name__} requires the APOC plugin. "
                "Install APOC in the target Neo4j database and try again.")


class Neo4jGraphSAGESampler(Neo4jSampler):
    r"""Neo4j-backed GraphSAGE neighbor sampler.

    Pushes a pre-compiled Cypher query into Neo4j for multi-hop neighbor
    sampling. Mirrors pyg-lib ``_sample`` semantics with ``replace=False``
    and ``disjoint=False``: if ``k < 0`` or ``k >= |neighborhood|`` all
    neighbors are taken (pyg-lib Case 1), else ``k`` are sampled uniformly
    via ``apoc.coll.randomItems``.

    Requires the Neo4j APOC plugin (``coll.distinct``, ``coll.flatten``,
    ``apoc.coll.randomItems``).

    Args:
        graph_store (DatabaseGraphStore): Neo4j-backed graph store.
        num_neighbors (list[int]): Neighbors to sample per hop. ``-1`` =
            take all neighbors at that hop.
        direction (str): Edge direction from each frontier node. One of
            ``'incoming'``, ``'outgoing'``, ``'undirected'``.
            (default: :obj:`'incoming'`)
        node_label (str, optional): Node label filter (e.g. ``"User"``).
            :obj:`None` matches all labels. (default: :obj:`None`)
        rel_type (str, optional): Relationship type filter (e.g. ``"KNOWS"``).
            :obj:`None` matches all types. (default: :obj:`None`)
        profile (bool): Prepend ``PROFILE`` to the query for execution-plan
            profiling. (default: :obj:`False`)
    """
    def __init__(
        self,
        graph_store: DatabaseGraphStore,
        num_neighbors: list[int],
        direction: str = 'incoming',
        node_label: str | None = None,
        rel_type: str | None = None,
        profile: bool = False,
    ) -> None:
        if direction not in ('incoming', 'outgoing', 'undirected'):
            raise ValueError(
                f"direction must be one of 'incoming', 'outgoing', "
                f"'undirected'; got {direction!r}.")

        self.num_neighbors = num_neighbors
        self.direction = direction

        super().__init__(
            graph_store,
            node_label=node_label,
            rel_type=rel_type,
            profile=profile,
        )

        self._probe_apoc(graph_store)

    def _build_node_query_params(self, seeds, **kwargs) -> dict:
        if seeds.numel() == 0:
            raise ValueError(
                "Neo4jGraphSAGESampler received an empty seed batch; the "
                "Cypher query cannot be evaluated against an empty list.")
        return {"seed_ids": seeds.tolist()}

    def _build_node_sampling_query(self) -> str:
        r"""Compile the multi-hop neighbor-sampling Cypher query.

        Parameterised by ``$seed_ids`` (see :meth:`_build_node_query_params`).
        Returns a single record with:

        * ``edges`` — ``[src_id, dst_id]`` pairs (global node IDs) for every
          sampled edge across all hops.
        * ``nodes`` — visited global node IDs in BFS order (seeds first,
          then each hop's new frontier), deduplicated.
        """
        rel = "" if self.rel_type is None else f":{self.rel_type}"
        seed_label = "" if self.node_label is None else f":{self.node_label}"
        nbr_label = "" if self.node_label is None else f":{self.node_label}"

        if self.direction == 'incoming':
            # (src)<-[r]-(neighbor): startNode(r)=neighbor, endNode(r)=src
            edge_pat = f"<-[r{rel}]-"
            nbr_expr = "startNode(rel)"
        elif self.direction == 'outgoing':
            # (src)-[r]->(neighbor): startNode(r)=src, endNode(r)=neighbor
            edge_pat = f"-[r{rel}]->"
            nbr_expr = "endNode(rel)"
        else:  # undirected
            edge_pat = f"-[r{rel}]-"
            nbr_expr = (f"CASE WHEN startNode(rel).{self.nodeid_property} = "
                        f"src.{self.nodeid_property} "
                        f"THEN endNode(rel) ELSE startNode(rel) END")

        edge_src_expr = f"startNode(rel).{self.nodeid_property}"
        edge_dst_expr = f"endNode(rel).{self.nodeid_property}"

        profile_prefix = "PROFILE\n        " if self.profile else ""

        q = []

        q.append(f"""
        // 1. initialize the frontier, visited and edges (single batched
        //    index seek; fail-loud if any seed_id is missing in the DB).
        {profile_prefix}MATCH (s{seed_label})
        WHERE s.{self.nodeid_property} IN $seed_ids
        WITH collect(s) AS frontier
        WHERE size(frontier) = size(coll.distinct($seed_ids))
        WITH frontier, frontier AS visited, [] AS edges
        """)

        for k in self.num_neighbors:
            q.append(f"""
            CALL (frontier, visited, edges) {{

            // 2. process frontier nodes in stable index order.
            UNWIND range(0, size(frontier)-1) AS i
            WITH i, frontier[i] AS src, visited, edges

            // 3. match neighbors via edges
            MATCH (src){edge_pat}(neighbor{nbr_label})
            WITH i, src, visited, edges, collect(r) AS cand_rels

            // 4. pyg-lib "take all" rule (Case 1 in _sample).
            WITH i, src, visited, edges,
                CASE
                    WHEN {k} < 0 OR {k} >= size(cand_rels)
                    THEN cand_rels
                    ELSE apoc.coll.randomItems(cand_rels, {k}, false)
                END AS picked_rels

            // 5. build the neighbor list and edge list for this src.
            WITH i, visited, edges,
                [rel IN picked_rels | {nbr_expr}] AS picked_nbrs,
                [rel IN picked_rels |
                 [{edge_src_expr}, {edge_dst_expr}]] AS new_edges
            ORDER BY i

            // 6. aggregate across all src nodes — back to a single row.
            WITH visited, edges,
                coll.flatten(collect(picked_nbrs)) AS picked_nbrs,
                coll.flatten(collect(new_edges)) AS new_edges


            // 7. filter revisited + deduplicate next frontier
            WITH visited, edges, new_edges,
                coll.distinct(
                    [n IN picked_nbrs WHERE NOT n IN visited]
                ) AS next_frontier

            RETURN
                next_frontier,
                visited + next_frontier AS next_visited,
                edges + new_edges AS next_edges
            }}
            WITH next_frontier AS frontier,
                next_visited AS visited,
                next_edges AS edges
            """)

        q.append(f"""
            RETURN
                edges AS edges,
                [n IN visited | n.{self.nodeid_property}] AS nodes
            """)

        return "\n".join(q)

    def _decode_node_sampling_record(
        self,
        record: Any,
        seed_nodes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Convert a raw record into ``(node, row, col)`` COO tensors.

        On an empty result returns ``seed_nodes`` plus zero-length edge
        tensors.
        """
        empty = torch.zeros(0, dtype=torch.long)

        if record is None or not record.get("nodes", None):
            return seed_nodes, empty, empty

        global_ids = torch.tensor(record["nodes"], dtype=torch.long)

        global_to_local = {
            int(gid): i
            for i, gid in enumerate(global_ids.tolist())
        }

        edges = record.get("edges") or []
        if edges:
            row = torch.tensor([global_to_local[e[0]] for e in edges],
                               dtype=torch.long)
            col = torch.tensor([global_to_local[e[1]] for e in edges],
                               dtype=torch.long)
        else:
            row = col = empty

        return global_ids, row, col
