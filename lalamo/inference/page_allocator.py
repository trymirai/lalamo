from collections import deque
from dataclasses import dataclass, field


@dataclass
class PageAllocator:
    total_pages: int
    free_page_ids: deque[int] = field(init=False)
    _pages_by_request: dict[str, list[int]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_pages < 1:
            raise ValueError("total_pages must be at least one.")
        self.free_page_ids = deque(range(self.total_pages))

    def pages_for(self, request_id: str) -> tuple[int, ...]:
        return tuple(self._pages_by_request[request_id])

    def reserve(self, request_id: str, page_count: int) -> None:
        if request_id in self._pages_by_request:
            raise ValueError(f"Request {request_id!r} already has pages.")
        if page_count < 1:
            raise ValueError("page_count must be at least one.")
        self._pages_by_request[request_id] = self._take_free_pages(page_count)

    def grow(self, request_id: str, page_count: int) -> bool:
        page_ids = self._pages_by_request[request_id]
        pages_needed = page_count - len(page_ids)
        if pages_needed > len(self.free_page_ids):
            return False
        page_ids.extend(self.free_page_ids.popleft() for _ in range(max(pages_needed, 0)))
        return True

    def release(self, request_id: str) -> None:
        self.free_page_ids.extend(self._pages_by_request.pop(request_id))

    def _take_free_pages(self, page_count: int) -> list[int]:
        if page_count > len(self.free_page_ids):
            raise RuntimeError(
                f"Requested {page_count} pages with only {len(self.free_page_ids)} free out of {self.total_pages}."
            )
        return [self.free_page_ids.popleft() for _ in range(page_count)]
