from typing import Protocol, KeysView
from numpy import float64
from numpy.typing import NDArray


class NetworkOps(Protocol):
    """
    Protocol for communication operations in distributed optimization.

    In our examples, we use 'topolink.NodeHandle' as a concrete implementation of this protocol.
    The source code of 'topolink.NodeHandle' can be found at:
        https://github.com/rui-huang-opt/topolink.

    However, any class that implements these methods and properties can be used as long as it adheres to this protocol.

    Attributes
    ----------
    degree : int
        The number of neighbors connected to this node.

    neighbors : KeysView[str]
        A view of the names of the neighbor nodes.
    """

    @property
    def degree(self) -> int: ...

    @property
    def neighbors(self) -> KeysView[str]: ...

    def exchange_as_array(self, state: NDArray[float64]) -> NDArray[float64]:
        """
        Exchanges the given state with all neighbor nodes and returns their states as a stacked array.

        Args:
            state (NDArray[np.float64]): The state array to exchange with neighbors.

        Returns:
            NDArray[np.float64]: A 2D array where each row corresponds to a neighbor's received state array.
        """
        ...
