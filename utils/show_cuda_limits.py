import torch
p = torch.cuda.get_device_properties(0)
print(p)
print("name:", p.name)
print("compute capability:", f"{p.major}.{p.minor}")
print("SMs (multiprocessors):", p.multi_processor_count)
print("VRAM GB:", p.total_memory / 1e9)
print("max threads/SM:", getattr(p, "max_threads_per_multi_processor", "n/a"))
print("warp size:", getattr(p, "warp_size", 32))
print("regs per block:", getattr(p, "regs_per_multiprocessor", "n/a"))
print("shared mem/block KB:", getattr(p, "shared_memory_per_block", 0) / 1024)

