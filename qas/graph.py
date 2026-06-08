"""
Per-query graph construction (step 4 of the QAS pipeline).

Builds an undirected weighted igraph over passage nodes and
entity nodes. 

Edges:
  passage <-> entity : weight = number of triples in the chunk touching entity
  entity  <-> entity : weight per triple (subject<->object), accumulated
  entity  <-> entity : synonym, weight = cosine if NV-Embed cosine >= threshold
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import igraph as ig

from .utils import text_processing
from .openie import OpenIEExtractor
from .encoder import EmbeddingStore


class GraphBuilder:
    def __init__(self, openie: OpenIEExtractor, store: EmbeddingStore,
                 synonym_threshold: float = 0.80):
        self.openie = openie
        self.store = store
        self.synonym_threshold = synonym_threshold

    def collect_triples_and_entities(self, pool: List[str]
                                     ) -> Tuple[list, dict, set]:
        """Read cached pool triples and derive entities.

        Returns:
          triples:               list of {subject, predicate, object,
                                          subject_n, object_n, from_chunk}
          chunk_triple_entities: dict cid -> set of normalized entities
          all_entities:          set of all normalized entities in the pool

        passage<->entity edge weight = number of triples in the chunk touching
        an entity, so we count occurrences here rather than collapsing to a set.
        """
        triples = []
        chunk_entity_counts = defaultdict(lambda: defaultdict(int))
        chunk_triple_entities = defaultdict(set)
        all_entities = set()
        for cid in pool:
            for t in self.openie.load_chunk_triples(cid):
                if not isinstance(t, (list, tuple)) or len(t) < 3:
                    continue
                s_n = text_processing(t[0])
                o_n = text_processing(t[2])
                if not s_n or not o_n:
                    continue
                triples.append({
                    "subject": t[0], "predicate": t[1], "object": t[2],
                    "subject_n": s_n, "object_n": o_n,
                    "from_chunk": cid,
                })
                # one triple touches both its subject and object entities
                chunk_entity_counts[cid][s_n] += 1
                chunk_entity_counts[cid][o_n] += 1
                chunk_triple_entities[cid].add(s_n)
                chunk_triple_entities[cid].add(o_n)
                all_entities.add(s_n)
                all_entities.add(o_n)
        # stash the per-chunk per-entity touch counts for edge weighting
        self._chunk_entity_counts = chunk_entity_counts
        return triples, chunk_triple_entities, all_entities

    def build(self, pool: List[str], triples: list,
              chunk_triple_entities: dict, all_entities: set
              ) -> Tuple[ig.Graph, Dict[str, int], List[str]]:
        """Build the undirected weighted igraph. Returns (G, name_to_idx, names)."""
        ent_keys = sorted(all_entities)
        ent_texts = ent_keys[:]   # normalized text is the embedding input

        names: List[str] = []
        name_to_idx: Dict[str, int] = {}

        def add(name):
            if name in name_to_idx:
                return
            name_to_idx[name] = len(names)
            names.append(name)

        for cid in pool:
            add(f"doc:{cid}")
        for k in ent_keys:
            add(f"ent:{k}")

        edge_w: Dict[Tuple[int, int], float] = defaultdict(float)
        counts = getattr(self, "_chunk_entity_counts", {})

        # passage <-> entity : weight 1 per (chunk, entity) — set membership,
        # matching the graph construction (NOT a triple-occurrence count).
        for cid, ents in chunk_triple_entities.items():
            for n in ents:
                if n not in all_entities:
                    continue
                a = name_to_idx[f"doc:{cid}"]
                b = name_to_idx[f"ent:{n}"]
                edge_w[(min(a, b), max(a, b))] += 1.0

        # entity <-> entity : per triple (subject<->object)
        for t in triples:
            s_n, o_n = t["subject_n"], t["object_n"]
            if s_n in all_entities and o_n in all_entities and s_n != o_n:
                a = name_to_idx[f"ent:{s_n}"]
                b = name_to_idx[f"ent:{o_n}"]
                edge_w[(min(a, b), max(a, b))] += 1.0

        # entity <-> entity : synonym, weight = cosine if cosine >= threshold
        if len(ent_keys) >= 2:
            embs = self.store.embed_texts(ent_texts, "query_to_passage")
            sim = embs @ embs.T
            iu, ju = np.triu_indices(len(ent_keys), k=1)
            sims = sim[iu, ju]
            mask = sims >= self.synonym_threshold
            for i, j, c in zip(iu[mask], ju[mask], sims[mask]):
                ai = name_to_idx[f"ent:{ent_keys[i]}"]
                bi = name_to_idx[f"ent:{ent_keys[j]}"]
                edge_w[(min(ai, bi), max(ai, bi))] += float(c)

        G = ig.Graph(n=len(names), directed=False)
        G.vs["name"] = names
        if edge_w:
            edges = list(edge_w.keys())
            G.add_edges(edges)
            G.es["weight"] = [edge_w[e] for e in edges]
        return G, name_to_idx, names
