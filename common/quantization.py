import numpy as np

def clip_quantize(data, clip_min, clip_max, bits):
    data = np.clip(data, clip_min, clip_max)
    data = np.round(((data - clip_min) / (clip_max - clip_min)) * (2**bits - 1)).astype(np.uint8)

    return data

def clip_dequantize(data, clip_min, clip_max, bits, shape):
    data = (data.astype(np.float32) / (2**bits - 1)) * (clip_max - clip_min) + clip_min

    return data

def array_to_binary(arr, n_bits):

    original_shape = arr.shape
    
    flattened = arr.flatten()
    
    binary_list = []
    for num in flattened:
        binary = format(int(num) & ((1 << n_bits) - 1), f'0{n_bits}b')
        binary_list.append(binary)

    binary_str = ''.join(binary_list)
    
    return binary_str, original_shape

def binary_to_array(binary_str, n_bits, original_shape, dtype=np.float32):
    chunks = [binary_str[i:i+n_bits] for i in range(0, len(binary_str), n_bits)]
    
    numbers = []
    for chunk in chunks:
        num = int(chunk, 2)
        if num & (1 << (n_bits-1)):
            num -= (1 << n_bits)
        numbers.append(num)

    arr = np.array(numbers, dtype=dtype).reshape(original_shape)
    
    return arr