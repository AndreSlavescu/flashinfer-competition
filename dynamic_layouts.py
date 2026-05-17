import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def foo(tensor):
    print(tensor.layout)  # Prints (?,?):(?,1) for dynamic layout
    a[0][0] = 67

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint16)
compiled_func = cute.compile(foo, a)
compiled_func(a)
print(a[0][0])

b = torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.uint16)
compiled_func(b)  # Reuse the same compiled function for different shape
