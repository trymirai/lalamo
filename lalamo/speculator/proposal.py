from dataclasses import dataclass

__all__ = ["AcceptedProposal", "ProposalNode", "TrieProposal"]


@dataclass(frozen=True)
class AcceptedProposal:
    token_ids: tuple[int, ...]
    node_indices: tuple[int, ...]
    compact_indices: tuple[int, ...]
    num_compact_indices: int
    terminal_node_index: int
    bonus_token_id: int


@dataclass(frozen=True)
class ProposalNode:
    token_id: int
    parent_index: int
    sample_position: int
    depth: int


@dataclass(frozen=True)
class TrieProposal:
    nodes: tuple[ProposalNode, ...]

    @staticmethod
    def create(root_token_id: int, first_sample_position: int = 1) -> "TrieProposal":
        return TrieProposal((ProposalNode(root_token_id, -1, first_sample_position, 0),))

    def add_node(self, parent_index: int, token_id: int) -> tuple["TrieProposal", int]:
        node_index = len(self.nodes)
        sample_position = self.nodes[-1].sample_position + 1
        depth = self.nodes[parent_index].depth + 1
        return TrieProposal((*self.nodes, ProposalNode(token_id, parent_index, sample_position, depth))), node_index

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(node.token_id for node in self.nodes)

    @property
    def parent_indices(self) -> tuple[int, ...]:
        return tuple(node.parent_index for node in self.nodes)

    @property
    def sample_positions(self) -> tuple[int, ...]:
        return tuple(node.sample_position for node in self.nodes)

    @property
    def depths(self) -> tuple[int, ...]:
        return tuple(node.depth for node in self.nodes)

    def verify(self, sampled_token_ids: tuple[int, ...]) -> AcceptedProposal:
        token_ids: tuple[int, ...] = ()
        node_indices: tuple[int, ...] = ()
        terminal_node_index = 0

        while True:
            sampled_token_id = sampled_token_ids[terminal_node_index]
            next_node_index: int | None = None

            for node_index, node in enumerate(self.nodes):
                if node.parent_index == terminal_node_index and node.token_id == sampled_token_id:
                    next_node_index = node_index
                    break

            if next_node_index is None:
                break

            token_ids = (*token_ids, sampled_token_id)
            node_indices = (*node_indices, next_node_index)
            terminal_node_index = next_node_index

        unpadded_compact_indices = (0, *node_indices)
        compact_padding = (0 for _ in range(len(self.nodes) - len(unpadded_compact_indices)))
        compact_indices = (*unpadded_compact_indices, *compact_padding)
        return AcceptedProposal(
            token_ids=token_ids,
            node_indices=node_indices,
            compact_indices=compact_indices,
            num_compact_indices=len(unpadded_compact_indices),
            terminal_node_index=terminal_node_index,
            bonus_token_id=sampled_token_ids[terminal_node_index],
        )
