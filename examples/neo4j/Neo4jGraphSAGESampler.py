from typing import Any, List, Optional

from torch_geometric.data.graph_store import RemoteGraphStore
from torch_geometric.sampler.remote_sampler import RemoteSampler


class Neo4jGraphSAGESampler(RemoteSampler):
    """Neo4j neighbor sampler structurally equivalent to pyg-lib
    GraphSAGE sampling.

    Performs multi-hop incoming-edge sampling by pushing a pre-compiled Cypher
    query into Neo4j.  The query mirrors pyg-lib's ``_sample`` semantics:

    * ``replace=False``, ``disjoint=False``
    * **Take-all rule** — when ``k < 0`` or ``k >= |neighbourhood|``, all
      neighbors are taken (pyg-lib Case 1).
    * **Order-preserving frontier deduplication** — new nodes are appended in
      first-encounter order, mirroring pyg-lib's ``Mapper``
      insertion semantics.
    * **All edges recorded** — edges to already-visited nodes are included,
      consistent with pyg-lib's ``add()`` function.

    Args:
        graph_store (RemoteGraphStore): Neo4j-backed graph store.
        num_neighbors (List[int]): Number of neighbors to sample per hop.
            Use ``-1`` to take all neighbors at a given hop.
        rel_type (str, optional): Relationship type filter (e.g. ``"KNOWS"``).
            If ``None``, all relationship types are matched.
            (default: ``None``)
        node_label (str, optional): Node label filter (e.g. ``"User"``).
            If ``None``, all node labels are matched.
            (default: ``None``)
        return_nodes_by_hop (bool): If ``True``, the nodes by hop will
            be tracked.
            (default: ``False``)
        direction (str): The direction of the edges to sample.
            (default: ``'incoming'``)
        profile (bool): Prepend ``PROFILE`` to the Cypher query for execution
            plan introspection in Neo4j Browser. (default: ``False``)
    """
    def __init__(
        self,
        graph_store: RemoteGraphStore,
        num_neighbors: List[int],
        return_nodes_by_hop: bool = False,
        direction: str = 'incoming',  # 'incoming' or 'outgoing' or undirected
        rel_type: Optional[str] = None,
        node_label: Optional[str] = None,
        profile: bool = False,
    ):
        self.return_nodes_by_hop = return_nodes_by_hop
        self.direction = direction
        self.nodeid_property = graph_store.nodeid_property
        self.rel_type = rel_type
        self.node_label = node_label
        self.profile = profile

        super().__init__(graph_store, num_neighbors,
                         track_nodes_by_hop=return_nodes_by_hop)

    def _build_node_fanout_query(self) -> str:
        """Build a Cypher query that performs multi-hop incoming-edge sampling.

        The query returns a single row with:

        * ``edges`` — list of ``[src_id, dst_id]`` (global node IDs) for
          every sampled edge.
        * ``nodes_by_hop`` — list of per-hop node ID lists (seeds first, then
          hop-1 new nodes, …).
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
        // 1. initialise the frontier, visited and edges
        {profile_prefix}UNWIND range(0, size($seed_ids)-1) AS i
        WITH i, $seed_ids[i] AS seed_id
        MATCH (s{seed_label})
        WHERE s.{self.nodeid_property} = seed_id
        WITH i, s
        WITH collect(s) AS frontier, collect(s) AS visited, [] AS edges
        WITH frontier, visited, edges, [frontier] AS nodes_by_hop
        """)

        for k in self.num_neighbors:
            q.append(f"""
            CALL (frontier, visited, edges, nodes_by_hop) {{

            // 2. process frontier nodes in stable index order.
            UNWIND range(0, size(frontier)-1) AS i
            WITH i, frontier[i] AS src, visited, edges

            // 3. match neighbors via incoming edges
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
                apoc.coll.flatten(collect(picked_nbrs)) AS picked_nbrs,
                apoc.coll.flatten(collect(new_edges)) AS new_edges

            // 7. filter revisited + deduplicate next frontier
            WITH visited, edges, new_edges,
                apoc.coll.toSet(
                    [n IN picked_nbrs WHERE NOT n IN visited]
                ) AS next_frontier

            RETURN
                next_frontier,
                visited + next_frontier AS next_visited,
                edges + new_edges AS next_edges,
                nodes_by_hop + [next_frontier] AS next_nodes_by_hop
            }}
            WITH next_frontier AS frontier,
                next_visited AS visited,
                next_edges AS edges,
                next_nodes_by_hop AS nodes_by_hop
            """)

        if self.return_nodes_by_hop:
            return_clause = f"""
            RETURN
                edges AS edges,
                [hop IN nodes_by_hop |
                [n IN hop | n.{self.nodeid_property}]] AS nodes_by_hop
            """
        else:
            return_clause = f"""
            RETURN
                edges AS edges,
                [n IN apoc.coll.flatten(nodes_by_hop) |
                n.{self.nodeid_property}] AS nodes
            """

        q.append(return_clause)

        return "\n".join(q)

    def _build_query_params(self, seeds, **kwargs) -> dict:
        return {"seed_ids": seeds.tolist()}

    def _extract_nodes_by_hop(self, record: Any) -> List[List[int]]:
        """Extract the ``nodes_by_hop`` field from the raw Neo4j record."""
        if not record:
            return []
        return [list(hop) for hop in record.get("nodes_by_hop") or []]
