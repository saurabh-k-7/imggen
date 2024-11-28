import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.device_count())  # Shows the number of GPUs detected
print(torch.cuda.get_device_name(0))  # Displays the name of the GPU (e.g., RTX 3050)
