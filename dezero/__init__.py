# =============================================================================
# step23.pyからstep32.pyまではsimple_coreを利用
is_simple_core = True
# =============================================================================

from dezero.core_simple import Variable  # noqa
from dezero.core_simple import Function  # noqa
from dezero.core_simple import using_config  # noqa
from dezero.core_simple import no_grad  # noqa
from dezero.core_simple import as_array  # noqa
from dezero.core_simple import as_variable  # noqa
from dezero.core_simple import setup_variable  # noqa

setup_variable()  # noqa
__version__ = '0.0.13'
