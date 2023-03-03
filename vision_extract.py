from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch

torch.set_grad_enabled(False)
feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-dc5')
detection_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')
vf_extractor = detection_model.model


# def inspect(a, prefix=''):
#     if not (isinstance(a, list) or isinstance(a, tuple) or len(prefix)<=2):
#         print(f"{prefix} {type(a)} {a.shape if type(a)==torch.Tensor else ''}")
#         return
#     print(f'{prefix} {type(a)}: {len(a)}')
#     for item in a:
#         inspect(item, prefix=prefix+'-')

# inspect(output)
"""
backbone outputs:

 <class 'tuple'>: 2

- <class 'list'>: 4   ### features
-- <class 'tuple'>: 2
--- <class 'torch.Tensor'> torch.Size([1, 256, 200, 267])
--- <class 'torch.Tensor'> torch.Size([1, 200, 267])
-- <class 'tuple'>: 2
--- <class 'torch.Tensor'> torch.Size([1, 512, 100, 134])
--- <class 'torch.Tensor'> torch.Size([1, 100, 134])
-- <class 'tuple'>: 2
--- <class 'torch.Tensor'> torch.Size([1, 1024, 50, 67])
--- <class 'torch.Tensor'> torch.Size([1, 50, 67])
-- <class 'tuple'>: 2
--- <class 'torch.Tensor'> torch.Size([1, 2048, 50, 67])
--- <class 'torch.Tensor'> torch.Size([1, 50, 67])

- <class 'list'>: 4   ### position_embeddings_list
-- <class 'torch.Tensor'>: 1
--- <class 'torch.Tensor'> torch.Size([256, 200, 267])
-- <class 'torch.Tensor'>: 1
--- <class 'torch.Tensor'> torch.Size([256, 100, 134])
-- <class 'torch.Tensor'>: 1
--- <class 'torch.Tensor'> torch.Size([256, 50, 67])
-- <class 'torch.Tensor'>: 1
--- <class 'torch.Tensor'> torch.Size([256, 50, 67])
"""