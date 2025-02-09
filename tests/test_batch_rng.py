from torch import tensor, float64
from torch.testing import assert_close
from torch_deterministic import BatchGenerator
from numpy.random import Generator, PCG64
from pytest import raises

def make_batch_generator(n=2):
    # Avoid using the default generator, because future versions of numpy might 
    # change the default.
    return BatchGenerator([
        Generator(PCG64(i))
        for i in range(n)
    ])


def test_batch_generator_mock():

    class MockGenerator:

        def __init__(self, x):
            self.x = x

        def get(self):
            return self.x

    bg = BatchGenerator([
        MockGenerator(0),
        MockGenerator(1),
        MockGenerator(2),
    ])

    assert_close(
            bg.get(),
            tensor([0, 1, 2]),
    )

def test_batch_generator_pickle():
    bg = make_batch_generator()

    import pickle
    bg_packed = pickle.dumps(bg)
    bg_unpacked = pickle.loads(bg_packed)

    assert_close(
            bg.uniform(),
            bg_unpacked.uniform(),
    )

def test_batch_generator_err_unknown_method():
    bg = make_batch_generator()
    with raises(AttributeError):
        bg.unknown_method()

# I don't know what the results for any of the following tests are supposed to 
# be *a priori*; I just ran the code once and copied the results.  But I do 
# know that the code should be deterministic, so the results should be the same 
# every time.

def test_batch_generator_integers():
    bg = make_batch_generator()
    assert_close(
            bg.integers(10),
            tensor([8, 4]),
    )

def test_batch_generator_uniform():
    bg = make_batch_generator()
    assert_close(
            bg.uniform(),
            tensor([0.63696169, 0.51182162], dtype=float64),
    )

def test_batch_generator_normal():
    bg = make_batch_generator()
    assert_close(
            bg.normal(),
            tensor([0.12573022, 0.34558419], dtype=float64),
    )

def test_batch_generator_2d():
    bg = make_batch_generator()
    assert_close(
            bg.integers(10, size=3),
            tensor([
                [8, 6, 5],
                [4, 5, 7],
            ]),
    )

def test_batch_generator_3d():
    bg = make_batch_generator()
    assert_close(
            bg.integers(10, size=(3, 4)),
            tensor([
                [[8, 6, 5, 2],
                 [3, 0, 0, 0],
                 [1, 8, 6, 9]],

                [[4, 5, 7, 9],
                 [0, 1, 8, 9],
                 [2, 3, 8, 4]],
            ]),
    )

