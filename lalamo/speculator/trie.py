from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FlatTrie:
    token_ids: np.ndarray  # (N,) int32 — root at index 0
    seeds: np.ndarray  # (N,) uint64 — per-node Gumbel seed
    depths: np.ndarray  # (N,) int32 — root has depth 0
    parent_indices: np.ndarray  # (N,) int32 — root has parent -1

    @property
    def num_nodes(self) -> int:
        return len(self.token_ids)

    def positions(self, base_position: int) -> np.ndarray:
        return base_position + self.depths

    def accept(
        self,
        sampled_tokens: np.ndarray,
    ) -> tuple[list[int], np.ndarray]:
        """Walk trie: accept path where sampled tokens match draft tokens.

        Args:
            sampled_tokens: (N,) token IDs sampled from LLM logits.
        Returns:
            accepted_tokens: list of accepted token IDs (excluding root).
            accepted_indices: (A,) int32 indices into this FlatTrie.
        """
        if self.num_nodes == 0:
            return [], np.array([], dtype=np.int32)

        children_map: dict[int, dict[int, int]] = {}
        for i in range(self.num_nodes):
            parent = int(self.parent_indices[i])
            children_map.setdefault(parent, {})[int(self.token_ids[i])] = i

        root_sampled = int(sampled_tokens[0])
        root_children = children_map.get(0, {})
        if root_sampled not in root_children:
            return [], np.array([], dtype=np.int32)

        child_idx = root_children[root_sampled]
        accepted_tokens = [root_sampled]
        accepted_indices = [child_idx]

        while child_idx in children_map:
            sampled = int(sampled_tokens[child_idx])
            siblings = children_map[child_idx]
            if sampled not in siblings:
                break
            child_idx = siblings[sampled]
            accepted_tokens.append(sampled)
            accepted_indices.append(child_idx)

        return accepted_tokens, np.array(accepted_indices, dtype=np.int32)


@dataclass
class TrieNode:
    token: int
    seed: int
    depth: int = 0
    children: list["TrieNode"] = field(default_factory=list)

    def add_child(self, token: int, seed: int) -> "TrieNode":
        child = TrieNode(token=token, seed=seed, depth=self.depth + 1)
        self.children.append(child)
        return child

    def get_child(self, token: int) -> "TrieNode | None":
        for child in self.children:
            if child.token == token:
                return child
        return None

    def total_nodes(self) -> int:
        return 1 + sum(child.total_nodes() for child in self.children)

    def max_depth(self) -> int:
        if not self.children:
            return self.depth
        return max(child.max_depth() for child in self.children)

    def linearize(self, *, include_root: bool = True) -> FlatTrie:
        token_ids: list[int] = []
        seeds: list[int] = []
        depths: list[int] = []
        parent_indices: list[int] = []

        stack: list[tuple[TrieNode, int]] = [(self, -1)]

        while stack:
            node, parent_idx = stack.pop()
            cur_idx = len(token_ids)
            token_ids.append(node.token)
            seeds.append(node.seed)
            depths.append(node.depth)
            parent_indices.append(parent_idx)
            stack.extend((child, cur_idx) for child in reversed(node.children))

        if not include_root:
            token_ids = token_ids[1:]
            seeds = seeds[1:]
            depths = [d - 1 for d in depths[1:]]
            parent_indices = [-1 if p == 0 else p - 1 for p in parent_indices[1:]]

        return FlatTrie(
            token_ids=np.array(token_ids, dtype=np.int32),
            seeds=np.array(seeds, dtype=np.uint64),
            depths=np.array(depths, dtype=np.int32),
            parent_indices=np.array(parent_indices, dtype=np.int32),
        )
