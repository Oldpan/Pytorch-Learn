import tvm
import numpy as np

tgt_host = "llvm"
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt = "llvm"

n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = tvm.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=64)

if tgt == "cuda" or tgt.startswith('opencl'):
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

ctx = tvm.context(tgt, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

# Inspect the Generated Code
if tgt == "cuda" or tgt.startswith('opencl'):
    dev_module = fadd.imported_modules[0]
    print("-----GPU code-----")
    print(dev_module.get_source())
else:
    print("-----CPU code-----")
    print(fadd.get_source())

# Save Compiled Module
from tvm.contrib import cc
from tvm.contrib import util

temp = util.tempdir()
fadd.save(temp.relpath("myadd.o"))
if tgt == "cuda":
    fadd.imported_modules[0].save(temp.relpath("myadd.ptx"))
if tgt.startswith('opencl'):
    fadd.imported_modules[0].save(temp.relpath("myadd.cl"))
cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])
print("temp listdir")
print(temp.listdir())

# Load Compiled Module

fadd1 = tvm.module.load(temp.relpath("myadd.so"))
if tgt == "cuda":
    fadd1_dev = tvm.module.load(temp.relpath("myadd.ptx"))
    fadd1.import_module(fadd1_dev)

if tgt.startswith('opencl'):
    fadd1_dev = tvm.module.load(temp.relpath("myadd.cl"))
    fadd1.import_module(fadd1_dev)

fadd1(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

fadd.export_library(temp.relpath("myadd_pack.so"))
fadd2 = tvm.module.load(temp.relpath("myadd_pack.so"))
fadd2(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
