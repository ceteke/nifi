import torch


def convert_dtype(dtype):
    if dtype == 'f32':
        return torch.float32
    elif dtype == 'f16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        raise ValueError("Unknown dtype expected to be one of [f32, f16, bf16]")


def pad_to_multiple(x, multiple=8):
    _, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple

    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x