from gguf import GGUFReader
import numpy as np

r = GGUFReader('/home/mbeldis/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57')
for k, v in r.fields.items():
    val = v.parts[v.data[0]] if v.data else v.parts
    if isinstance(val, np.ndarray) and val.dtype == np.uint8:
        try:
            val = bytes(val).decode('utf-8')
        except UnicodeDecodeError:
            pass
    print(k, val)

