def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


input_height, input_width = 4, 4  # Input size
kernal_height, kernel_width = 3, 3  # Kernel size
stride_height, stride_width = 1, 1  # Kernel stride
padding_height, padding_width = 1, 1  # Padding size

outsize_height = get_conv_outsize(input_height, kernal_height, stride_height, padding_height)
outsize_width = get_conv_outsize(input_width, kernel_width, stride_width, padding_width)

print(f"outsize_width: {outsize_width}")
print(f"outsize_height: {outsize_height}")
