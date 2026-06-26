import importlib


def test_vector_store_module_exposes_required_functions():
    module = importlib.import_module("vector_store")

    assert hasattr(module, "build_faiss_index")
    assert hasattr(module, "load_faiss_index")
    assert hasattr(module, "search")
