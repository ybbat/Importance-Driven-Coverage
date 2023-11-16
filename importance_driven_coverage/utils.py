import torch


def unravel_index(indices: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Converts a tensor of flat indices into a tensor of coordinate vectors.

    This is a `torch` implementation of `numpy.unravel_index`.

    Credit:
        https://github.com/francois-rozet/torchist

    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.

    Returns:
        The unraveled coordinates, (*, D).
    """

    shape = indices.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode="trunc") % shape[:-1]
