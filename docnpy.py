import numpy as np
file_path = r"C:\Users\Admin\Desktop\cs106\doannhom\bieudo\td3\TD3_InvertedDoublePendulum-v5_0.npy"

data = np.load(file_path)

print("Shape:", data.shape)
print("Dtype:", data.dtype)
print("Số phần tử:", data.size)
print("10 giá trị đầu:", data[:10])
print("10 giá trị cuối:", data[-10:])