import torch

print(torch.__version__)                # Check the version number of the pytorch installation
print(torch.cuda.is_available())        # Check whether cuda is available. True is available, i.e. the gpu version pytorch
print(torch.cuda.get_device_name())     # Return GPU model
print(torch.cuda.device_count())        # Returns the number of cuda (GPU) available, where 0 represents one
print(torch.version.cuda)   
# print(torch.cuda.current_device())      # Return the GPU index number