import pytest

from lalamo.inference.page_allocator import PageAllocator


def test_allocator_reserves_grows_and_reclaims_pages() -> None:
    allocator = PageAllocator(total_pages=6)
    allocator.reserve("first", page_count=2)
    allocator.reserve("second", page_count=1)
    allocator.grow("first", page_count=3)
    allocator.release("first")

    with pytest.raises(KeyError):
        allocator.pages_for("first")
    assert allocator.pages_for("second") == (2,)
    assert tuple(allocator.free_page_ids) == (4, 5, 0, 1, 3)


def test_allocator_reports_exhaustion_without_losing_pages() -> None:
    allocator = PageAllocator(total_pages=2)
    allocator.reserve("first", page_count=2)

    with pytest.raises(RuntimeError, match="only 0 free"):
        allocator.reserve("second", page_count=1)

    assert not allocator.free_page_ids
    assert allocator.pages_for("first") == (0, 1)
