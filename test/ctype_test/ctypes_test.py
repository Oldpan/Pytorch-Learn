from ctypes import *

dll = CDLL("./add.so")

print(dll.add(10, 30))

dll.addf.restype = c_float
dll.addf.argtypes = (c_float, c_float)

print(dll.addf(4.2, 8.5))


class Point(Structure):
    _fields_ = [("x", c_float), ("y", c_float)]


p = Point(2, 5)
p.y = 4
print(p.x, p.y)


a = c_int(66)
b = pointer(a)
c = POINTER(c_int)(a)
print(b.contents)
print(c.contents)

dll.print_point.argtypes = (POINTER(Point),)
dll.print_point.restype = None

p = Point(32.4, 22.5)
dll.print_point(byref(p))

