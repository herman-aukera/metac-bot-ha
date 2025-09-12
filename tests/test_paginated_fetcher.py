from src.infrastructure.metaculus.paginated_fetcher import fetch_all


class Item:
    def __init__(self, id: int):  # noqa: A003 (simple test helper)
        self.id: int = id


def test_pagination_fetch_accumulates() -> None:
    pages = {
        0: [Item(1), Item(2), Item(3)],
        1: [Item(4)],
    }

    def fetch(page: int, size: int):  # type: ignore[override]
        return pages.get(page, [])  # type: ignore[return-value]

    items = fetch_all(fetch_page=fetch, limit=10, page_size=3)
    assert [i.id for i in items] == [1, 2, 3, 4]
