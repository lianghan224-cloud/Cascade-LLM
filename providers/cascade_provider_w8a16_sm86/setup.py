from pathlib import Path

from setuptools import Distribution, setup


class BinaryProviderDistribution(Distribution):
    def has_ext_modules(self):
        return True


library = (
    Path(__file__).resolve().parent
    / "cascade_provider/lib/libcascade_cutlass_sm86.so"
)
if not library.is_file():
    raise RuntimeError(
        "refusing to build an unusable provider wheel: missing {}".format(
            library
        )
    )

setup(distclass=BinaryProviderDistribution)
