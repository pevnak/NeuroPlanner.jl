import HierarchicalUtils: NodeType, LeafNode, children

@nospecialize

NodeType(::Type{<:BitVector}) = LeafNode()

children(n::MaskedNode) = (n.data, n.mask)
children(n::MaskedModel) = (n.m,)

@specialize
