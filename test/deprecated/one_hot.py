# import numpy as np
# import math
# from Layers.layer import layer
# from Layers.utils import is_valid_input_feature
#
#
# class one_hot(layer):
#     def __init__(self, input_features=None):
#         super(one_hot, self).__init__()
#
#         self.input_features = input_features
#         self.is_compiled = False
#
#         if is_valid_input_feature(input_features):
#             self.compile(input_features)
#
#     def compile(self, input_features):
#         if is_valid_input_feature(input_features):
#             self.input_features = input_features
#         else:
#             raise ValueError
#
#         self.input_dim = input_features
#         self.output_features = input_features
#         self.output_dim = input_features
#
#
#         self.output_gpu = cuda.device_array(shape=self.output_dim, dtype=np.float32)
#         self.max_index = cuda.device_array(shape=1, dtype=np.int32)
#
#         self.block_dim = 256
#         self.grid_dim = math.ceil(self.input_features / 256)
#         self._preprocess = _find_max_index
#         self._compute = _do_one_hot
#
#         self.is_compiled = True
#
#         self.description = {'type': 'layer',
#                             'class': 'one_hot',
#                             'trainable': False,
#                             'input_features': self.input_features}
#
#     def forward(self, input_gpu):
#         assert isinstance(input_gpu, cuda.cudadrv.devicearray.DeviceNDArray)
#         assert len(input_gpu.shape) == 1
#
#         self._preprocess[1, 1](input_gpu, self.max_index)
#         cuda.synchronize()
#         # tmp = self.max_index.copy_to_host()
#         # print(tmp)
#         self._compute[self.grid_dim, self.block_dim](self.output_gpu, self.max_index)
#         return self.output_gpu
