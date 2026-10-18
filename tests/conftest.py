import os

import mlx.core as mx

if os.environ.get("MOSS_MLX_TEST_CPU"):
    mx.set_default_device(mx.cpu)
