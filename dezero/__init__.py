# =============================================================================
# step23.pyからstep32.pyまではsimple_coreを利用
is_simple_core = False
# =============================================================================

if is_simple_core:
    from dezero.core_simple import Variable  # noqa
    from dezero.core_simple import Function  # noqa
    from dezero.core_simple import using_config  # noqa
    from dezero.core_simple import no_grad  # noqa
    from dezero.core_simple import as_array  # noqa
    from dezero.core_simple import as_variable  # noqa
    from dezero.core_simple import setup_variable  # noqa
    pass

else:
    from dezero.core import Variable  # noqa
    from dezero.core import Parameter  # noqa
    from dezero.core import Function  # noqa
    from dezero.core import using_config  # noqa
    from dezero.core import no_grad  # noqa
    from dezero.core import test_mode  # noqa
    from dezero.core import as_array  # noqa
    from dezero.core import as_variable  # noqa
    from dezero.core import setup_variable  # noqa
    from dezero.core import Config  # noqa
    from dezero.layers import Layer  # noqa
    from dezero.models import Model  # noqa
    from dezero.datasets import Dataset  # noqa
    from dezero.dataloaders import DataLoader  # noqa
    from dezero.dataloaders import SeqDataLoader  # noqa

    from dezero import functions  # noqa
    from dezero import datasets  # noqa
    from dezero import optimizers  # noqa
    from dezero import cuda  # noqa
    pass

setup_variable()  # noqa
__version__ = '0.0.13'
