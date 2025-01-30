from dataclasses import dataclass

from fartsovka.modules.common import config_converter, register_config_union


@dataclass
class Abstract:
    pass


@dataclass
class A(Abstract):
    a: int


@dataclass
class B(Abstract):
    b: int


UnionType = A | B


register_config_union(UnionType)


@dataclass
class C:
    c: UnionType


def test_module_config() -> None:
    assert config_converter.unstructure(C(c=A(a=1)), C) == {"c": {"type": "A", "a": 1}}
