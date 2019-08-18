import torch
from torch import Tensor
import torch.nn as nn
from typing import Iterable
import math
import time


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)    # 在64 x 64 这个输入的model中没有执行resnet类似的结构
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # [1, 16, 1, 1],   # [1, 16, 1, 1],
            [2, 24, 1, 2],  # [4, 24, 2, 2],   # [6, 24, 2, 2],
            [2, 32, 1, 2],  # [4, 32, 2, 2],   # [6, 32, 3, 2],
            [2, 64, 1, 2],  # [4, 64, 2, 2],   # [6, 64, 4, 2],
            [2, 96, 1, 1],  # [4, 96, 3, 1],   # [6, 96, 3, 1],
            [2, 160, 1, 2],  # [4, 160, 3, 2],   # [6, 160, 3, 2],
            [2, 320, 1, 1],  # [4, 320, 1, 1],   # [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2(pretrained=True):
    model = MobileNetV2()
    if pretrained:
        state = torch.load('/home/prototype/Documents/Pytorch-pretrained/mobilenet_v2.pth')
        model.load_state_dict(state)
        return model
    else:
        return model


def children(m: nn.Module):
    "Get children of `m`."
    return list(m.children())


def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))


flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]


def in_channels(m: nn.Module):
    "Return the shape of the first weight layer in `m`."
    for l in flatten_model(m):
        if hasattr(l, 'weight'): return l.weight.shape[1]
    raise Exception('No weight layer')


def one_param(m: nn.Module):
    "Return the first parameter of `m`."
    return next(m.parameters())


def dummy_batch(m: nn.Module, size: tuple = (64, 64)):
    "Create a dummy batch to go through `m` with `size`."
    ch_in = in_channels(m)
    return one_param(m).new(1, ch_in, *size).zero_().requires_grad_(False)


def dummy_eval(m: nn.Module, size: tuple = (64, 64)):
    "Pass a `dummy_batch` in evaluation mode in `m` with `size`."
    return m.eval()(dummy_batch(m, size))


def is_listy(x) -> bool: return isinstance(x, (tuple, list))


class Hook():
    "Create a hook on `m` with `hook_func`."

    def __init__(self, m: nn.Module, hook_func, is_forward: bool = True, detach: bool = True):
        self.hook_func, self.detach, self.stored = hook_func, detach, None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module: nn.Module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input = (o.detach() for o in input) if is_listy(input) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


class Hooks():
    "Create several hooks on the modules in `ms` with `hook_func`."

    def __init__(self, ms, hook_func, is_forward: bool = True, detach: bool = True):
        self.hooks = [Hook(m, hook_func, is_forward, detach) for m in ms]

    def __getitem__(self, i: int) -> Hook: return self.hooks[i]

    def __len__(self) -> int: return len(self.hooks)

    def __iter__(self): return iter(self.hooks)

    @property
    def stored(self): return [o.stored for o in self]

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks: h.remove()

    def __enter__(self, *args): return self

    def __exit__(self, *args): self.remove()


def _hook_inner(m, i, o): return o if isinstance(o, Tensor) else o if is_listy(o) else list(o)


def hook_outputs(modules, detach: bool = True, grad: bool = False) -> Hooks:
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner, detach=detach, is_forward=not grad)


def model_sizes(m: nn.Module, size: tuple = (64, 64)):
    "Pass a dummy input through the model `m` to get the various sizes of activations."
    with hook_outputs(m) as hooks:
        x = dummy_eval(m, size)
        return [o.stored.shape for o in hooks]


def num_features_model(m: nn.Module) -> int:
    "Return the number of output features for `model`."
    return model_sizes(m)[-1][1]


def create_body(model, cut=-1):
    return nn.Sequential(*list(model.children())[:cut])


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    n = q if type(q) == int else len(p) if q is None else len(q)
    if len(p) == 1: p = p * n
    assert len(p) == n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."

    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)


class Flatten(nn.Module):
    def forward(self, input):
        # output = input.view(input.size(0), -1)
        # output = input.view([int(input.size(0)), -1])
        output = input.flatten(1)
        return output


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0., actn=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers


def create_head(nf: int, nc: int, lin_ftrs=None, ps=0.5, bn_final: bool = False):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps) == 1: ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    layers = [AdaptiveConcatPool2d(), Flatten()]
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, True, p, actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)


body = create_body(mobilenetv2(pretrained=False), -1)
nf = num_features_model(body) * 2  # Here we get the output channel from last layer
head = create_head(nf, 3, None, ps=0.5, bn_final=False)
model = nn.Sequential(body, head)


if __name__ == '__main__':

    # state = torch.load('/home/prototype/Documents/gesture/hand-images/models/mobilenetv2-64_SS.pth')
    # model.load_state_dict(state['model'], strict=True)

    example = torch.rand(10, 3, 64, 64)
    print(model)
    out = model(example)

    # torch_out = torch.onnx.export(model,
    #                               example,
    #                               "mobilenetv2-64_SS.onnx",
    #                               verbose=True,
    #                               export_params=True
    #                               )


# model.eval()
# traced_script_module = torch.jit.trace(model, example)
# output = traced_script_module(example)
# print(traced_script_module)
# traced_script_module.save('new-mobilenetv2-128_S.pt')
# print(output)


# with torch.no_grad():
#     model.eval().cuda()
#     since = time.time()
#     for i in range(1000):
#         model(example)
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.
#           format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
