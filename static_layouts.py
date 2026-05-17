import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor):
    print(f"tensor.layout: {tensor.layout}")  # Prints tensor layout at compile time
    cute.printf("tensor: {}", tensor)         # Prints tensor values at runtime
    tensor[0] = 67
    
a = torch.tensor([1, 2, 3])
a_pack = from_dlpack(a)
compiled_func = cute.compile(foo, a_pack)
compiled_func(a_pack)
print(a)