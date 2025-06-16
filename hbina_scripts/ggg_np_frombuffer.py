import numpy as np
import numba


@numba.jit(cache=False)
def hello():
    buffer = np.empty(1024, dtype=np.float64)
    buffer2 = np.frombuffer(buffer, dtype=np.float64)
    buffer3 = np.frombuffer(buffer, dtype=np.float64, count=5, offset=1)
    print(buffer)
    print(buffer2)
    print(buffer3)

    assert len(buffer3) == 5
    assert np.all(buffer[1:6] == buffer3)


hello()
