def test_imports():
    import retree
    from retree.models.retree import ReTreeModel
    from retree.memory.tree_index import HierarchicalTreeMemory
    from retree.retrieval.hierarchical_retriever import HierarchicalRetriever
    assert ReTreeModel is not None
    assert HierarchicalTreeMemory is not None
    assert HierarchicalRetriever is not None
