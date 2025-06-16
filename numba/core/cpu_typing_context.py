"""
Extended CPU typing context that includes atomic operations
"""

from numba.core.typing.context import Context


class CPUAtomicTypingContext(Context):
    """
    Extended CPU typing context that includes atomic operations
    """

    def load_additional_registries(self):
        # Load standard CPU registries first
        super().load_additional_registries()

        # Load CPU atomic typing registry
        from numba.core.typing import cpu_atomic_decl

        self.install_registry(cpu_atomic_decl.registry)
