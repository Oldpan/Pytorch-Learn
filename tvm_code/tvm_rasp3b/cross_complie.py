import numpy as np

import tvm
from tvm import rpc
from tvm.contrib import util

n = tvm.convert(1024)
A = tvm.placeholder((n,), name='A')
B = tvm.compute((n,), lambda i: A[i] + 1.0, name='B')
s = tvm.create_schedule(B.op)

local_demo = False

if local_demo:
    target = 'llvm'
else:
    target = 'llvm -target=armv7l-linux-gnueabihf'

func = tvm.build(s, [A, B], target=target, name='add_one')
# save the lib at a local temp folder
temp = util.tempdir()
path = temp.relpath('lib.tar')
func.export_library(path)


if local_demo:
    remote = rpc.LocalSession()
else:
    # The following is my environment, change this to the IP address of your target device
    host = '192.168.1.104'
    port = 9000
    remote = rpc.connect(host, port)


remote.upload(path)
func = remote.load_module('lib.tar')

# create arrays on the remote device
ctx = remote.cpu()
a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
# the function will run on the remote device
func(a, b)
np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

time_f = func.time_evaluator(func.entry_name, ctx, number=10)
cost = time_f(a, b).mean
print('%g secs/op' % cost)