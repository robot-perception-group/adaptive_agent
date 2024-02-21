import torch
from torch import nn


class FTA(nn.Module):
    """
    Fuzzy Tiling Activations (FTA)
    https://github.com/labmlai/annotated_deep_learning_paper_implementations
    Reference: fuzzy tiling activations: a simple approach to learning sparse representations online, Yangchen Pan et al. 2021
    """

    def __init__(self, upper_limit: float = 2.0, delta: float = 0.125):
        """
        :param limit: is the limit $[-u, u]$
        :param delta: is the bin size $\delta$
        """
        super().__init__()

        lower_limit = -upper_limit
        eta = delta

        # Initialize tiling vector
        # $$\mathbf{c} = (l, l + \delta, l + 2 \delta, \dots, u - 2 \delta, u - \delta)$$
        self.c = nn.Parameter(
            torch.arange(lower_limit, upper_limit, delta), requires_grad=False
        )
        # The input vector expands by a factor equal to the number of bins $\frac{u - l}{\delta}$
        self.nbins = len(self.c)
        # $\delta$
        self.delta = delta
        # $\eta$
        self.eta = eta

    def fuzzy_i_plus(self, x: torch.Tensor):
        """
        #### Fuzzy indicator function

        $$I_{\eta,+}(x) = I_+(\eta - x) x + I_+ (x - \eta)$$
        """
        return (x <= self.eta) * x + (x > self.eta)

    def forward(self, z: torch.Tensor):
        # Add another dimension of size $1$.
        # We will expand this into bins.
        # z = z.view(*z.shape, 1)
        z.unsqueeze_(-1)

        # $$\phi_\eta(z) = 1 - I_{\eta,+} \big( \max(\mathbf{c} - z, 0) + \max(z - \delta - \mathbf{c}, 0) \big)$$
        z = 1.0 - self.fuzzy_i_plus(
            torch.clip(self.c - z, min=0.0)
            + torch.clip(z - self.delta - self.c, min=0.0)
        )

        # Reshape back to original number of dimensions.
        # The last dimension size gets expanded by the number of bins, $\frac{u - l}{\delta}$.
        return z.view(-1, z.shape[-2] * z.shape[-1])


if __name__ == "__main__":
    fta = FTA()

    z = torch.rand(10, 100)
    print(fta(z).shape)
