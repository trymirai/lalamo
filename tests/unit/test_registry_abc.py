# test_registry_abc.py
import gc
import weakref

from lalamo.registry_abc import RegistryABC


def test_basic_registration_and_multilevel() -> None:
    class AbstractFoo(RegistryABC):
        pass

    class A(AbstractFoo):
        pass

    class B(A):
        pass

    # __get_descendants__ is a class method returning a list
    assert isinstance(AbstractFoo.__descendants__(), tuple)

    # Order is not guaranteed; compare as sets
    assert set(AbstractFoo.__descendants__()) >= {A, B}
    assert set(A.__descendants__()) == {B}
    assert set(B.__descendants__()) == set()


def test_excludes_classes_that_directly_list_registryabc() -> None:
    class AbstractFoo(RegistryABC):
        pass

    class A(AbstractFoo):
        pass

    # This class directly lists RegistryABC, so it must be excluded everywhere.
    class Hybrid(AbstractFoo, RegistryABC):
        pass

    assert A in AbstractFoo.__descendants__()  # tracked
    assert Hybrid not in AbstractFoo.__descendants__()  # excluded


def test_no_cross_pollination_between_roots() -> None:
    class AbstractFoo(RegistryABC):
        pass

    class AbstractBar(RegistryABC):
        pass

    class AFoo(AbstractFoo):
        pass

    class BBar(AbstractBar):
        pass

    assert AFoo in AbstractFoo.__descendants__()  # in its own tree
    assert BBar not in AbstractFoo.__descendants__()  # not leaking across

    assert BBar in AbstractBar.__descendants__()  # in its own tree
    assert AFoo not in AbstractBar.__descendants__()  # not leaking across


def test_mixin_and_indirect_registryabc_paths_are_included() -> None:
    class AbstractFoo(RegistryABC):
        pass

    class OtherRoot(RegistryABC):
        pass

    class Mixin:
        pass

    # Indirect path via a base that itself descends from RegistryABC should *not*
    # exclude the class, because the class does not directly list RegistryABC.
    class Indirect(OtherRoot):
        pass

    class D(Mixin, AbstractFoo, Indirect):
        pass

    assert D in AbstractFoo.__descendants__()


def test_dynamic_updates_on_late_definitions() -> None:
    class AbstractFoo(RegistryABC):
        pass

    assert set(AbstractFoo.__descendants__()) == set()

    class A(AbstractFoo):
        pass

    assert A in AbstractFoo.__descendants__()

    class B(A):
        pass

    # Newly defined class appears without needing to "refresh" anything
    assert B in AbstractFoo.__descendants__()
    # And it should also appear under A
    assert B in A.__descendants__()


def test_weakset_allows_garbage_collection() -> None:
    class AbstractFoo(RegistryABC):
        pass

    def make_temp() -> type:
        class Temp(AbstractFoo):
            pass

        return Temp

    temp_cls = make_temp()
    wr = weakref.ref(temp_cls)

    # It should be registered while alive
    assert temp_cls in AbstractFoo.__descendants__()
    assert wr() is not None

    # Drop the strong ref and collect
    del temp_cls
    gc.collect()

    # Weak ref is cleared, and the registry reflects that
    assert wr() is None
    names = {cls.__name__ for cls in AbstractFoo.__descendants__()}
    assert "Temp" not in names


def test_descendants_method_exists_on_all_registryabc_subclasses() -> None:
    class Root(RegistryABC):
        pass

    class Mid(Root):
        pass

    class Leaf(Mid):
        pass

    # Method exists and returns lists on all levels
    for C in (Root, Mid, Leaf):  # noqa: N806
        assert callable(C.__descendants__)
        assert isinstance(C.__descendants__(), tuple)

    assert Leaf in Mid.__descendants__()
    assert Leaf in Root.__descendants__()
    assert Mid in Root.__descendants__()
