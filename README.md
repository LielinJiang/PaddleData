
English | [简体中文](./README_cn.md)

# PaddleData

## Document Tutorial

#### **Installation**

* Environment dependence:
   - PaddlePaddle >= 2.3.0
   - Python >= 3.6
   - CUDA >= 10.1

##### install with pip
pip
```
pip install paddledata
```
from source
```
python setup.py install
```
## Examples

```python
# load imagenet dataset
import sys
import time
import paddle
import paddledata

data_root = '/path/to/imagenet/ILSVRC2012/train/'

def imagenet_pipeline():
    image, label = paddle.vision.reader.file_label_reader(data_root, 
                                                          batch_size=64, 
                                                          shuffle=True, 
                                                          drop_last=True)

    def decode(image):
        image = paddledata.ops.decode_random_crop(image)
        return image

    def resize(image):
        image = paddle.vision.ops.image_resize(image, size=224, data_format='NHWC')
        return image
    def transpose(image):
        image = paddle.transpose(image, [0, 3, 1, 2])
        return image

    image = paddle.io.map(decode, image)

    image = paddle.io.map(resize, image)

    return {'image': image, 'label': label}


dataloader = paddle.io.DataLoader(imagenet_pipeline)

for i, data in enumerate(dataloader):
    print('index:', i, data['image'].shape, data['image'].place, data['image'].dtype)
```

## Changelog

- v0.1.0 (2023.04.15)
  - Release first version

## Contributing

Contributions and suggestions are highly welcomed. Most contributions require you to agree to a [Contributor License Agreement (CLA)](https://cla-assistant.io/PaddlePaddle/PaddleGAN) declaring.
When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA. Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.
For more, please reference [contribution guidelines](docs/en_US/contribute.md).

## License
PaddleGAN is released under the [Apache 2.0 license](LICENSE).
