class FeatureScalingMixin:
    def __init__(self, blocks=None):
        self.blocks = blocks

    def scale(self, x, gamma):
        blocks = self.blocks
        if blocks is None:
            return gamma * x
        else:
            xs = x.copy()
            assert not isinstance(gamma, float) or len(blocks) == 1
            assert isinstance(gamma, float) or len(gamma) == len(blocks)
            for gamma_k, block_k in zip(gamma, blocks):
                xs[:, block_k] *= gamma_k
            return xs