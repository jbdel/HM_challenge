import torch

def _convert_to_one_hot(expected, output):
    # This won't get called in case of multilabel, only multiclass or binary
    # as multilabel will anyways be multi hot vector
    if output.squeeze().dim() != expected.squeeze().dim() and expected.dim() == 1:
        expected = torch.nn.functional.one_hot(
            expected.long(), num_classes=output.size(-1)
        ).float()
    return expected
