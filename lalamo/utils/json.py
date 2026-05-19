__all__ = ["JSON"]

type JSON = str | int | float | bool | None | dict[str, JSON] | list[JSON]
