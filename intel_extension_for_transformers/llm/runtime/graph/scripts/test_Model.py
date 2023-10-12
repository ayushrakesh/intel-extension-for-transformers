import numpy as np
from intel_extension_for_transformers.llm.runtime.graph import Model
model = Model()

weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.int8)
scales = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
dst = np.zeros((12288, 4096), dtype=np.int8)

model.init_from_bin("mpt", "/mnt/disk2/data/zhenweil/codes/ggml/mpt_ne.bin", max_new_tokens=20, num_beams=1, do_sample=True, top_k=40, top_p=0.95)
model.model.numpy_to_float_ptr(weights, scales, dst)

# 打印C++函数返回的指针值
print(dst)

import struct
# num = struct.pack('b', -128)
# 打开一个文件以二进制写入
with open('output.bin', 'wb') as f:
    for i in range(len(dst)):
        f.write(struct.pack('b', dst[i]))
