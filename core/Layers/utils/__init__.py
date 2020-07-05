# def is_valid_3d_input_dim(input_dim):
#     if input_dim is None:
#         return False
#     else:
#         if not isinstance(input_dim, tuple):
#             raise TypeError
#         if len(input_dim) != 3:
#             raise ValueError
#         if input_dim[1] != input_dim[2]:
#             raise ValueError
#
#         if min(input_dim) > 0:
#             return True
#         else:
#             return False
#
#
# def is_valid_input_feature(input_features):
#     if input_features is None:
#         return False
#     else:
#         if not isinstance(input_features, int):
#             raise TypeError
#         if input_features <= 0:
#             raise ValueError
#         else:
#             return True
