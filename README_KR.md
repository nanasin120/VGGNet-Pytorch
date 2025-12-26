# VGGNet

[English Version](./README.md)

이 프로젝트는 2014년 Oxford 대학교의 Karen Simonyan과 Andrew Zisserman이 발표한 논문 

*"Very Deep Convolutional Networks for Large-Scale Image Recognition"* 에 명시된 VGGNet 모델을 PyTorch로 재현한 프로젝트입니다.

## 프로젝트 결과
<img width="5367" height="1468" alt="image" src="https://github.com/user-attachments/assets/4a858864-8677-463a-b7f3-fc6eecba72ae" />
배치 정규화와 learning_rate 낮추기를 통해 그래프의 변동성을 줄이고 학습을 원활하게 했습니다.

Epochs를 10으로 정해 그리 많이 학습을 한것은 아니지만, 모델의 깊이가 깊을수록 정확도에서는 보다 더 높은 결과를 얻을 수 있었습니다.


## 프로젝트 구성

*"Very Deep Convolutional Networks for Large-Scale Image Recognition"* 에 명시된 **VGGNet** 아키텍처를 PyTorch로 최대한 그대로 구현한 프로젝트입니다.

## 프로젝트 준비

파이썬 환경은 아나콘다를 이용해 torch, torchvision, tqdm 패키지를 설치해 사용했습니다.
```
conda install torch torchvision tqdm
```

## 설치 및 실행 방법

conda를 이용할 경우 제공된 environment.yml 파일을 사용하여 제가 사용한 개발 환경을 그대로 복제할 수 있습니다.
```
# 환경 생성 (환경 이름은 environment.yml 내부에 정의된 이름을 따릅니다)
conda env create -f environment.yml

# 가상환경 활성화
conda activate [가상환경_이름]
```

혹은 pip를 이용할수도있습니다.
```
pip install -r requirements.txt
```

실행은 아래 코드로 가능합니다.
```
python train.py
```

결과는 matplotlib를 통해 그래프로 나오게 됩니다.

## 모델 구성

<img width="618" height="630" alt="image" src="https://github.com/user-attachments/assets/b32c8cf0-68fe-484b-a0c8-9d1c770c1659" />
<p align="center"><em>Table 1: Architectural configurations of VGGNet (A-E). (Source: Simonyan & Zisserman, 2014)</em></p>

VGGNet에는 A, B, C, D, E, 그래고 A-LRN 총 여섯개의 구성이 존재합니다. 이번 프로젝트에서는 A-LRN을 제외한 다섯개의 모델을 구현했습니다.

```
vggnet_a_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vggnet_b_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vggnet_c_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, (1, 256), 'M', 512, 512, (1, 512), 'M', 512, 512, (1, 512), 'M']
vggnet_d_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
vggnet_e_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
```
그림에 나와있는 채널 그대로 준비한 리스트입니다. 

모든 conv는 3x3으로 통일되었지만 중간에 있는 (1, 256)에서는 1x1 conv를 사용했습니다.

```
def build_layer(cfg):
    layers = []
    in_channels = 3

    for x in cfg:
        if x == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif isinstance(x, tuple):
            k, c = x
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=c, kernel_size=k, padding=0))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            in_channels = c
        else:
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(x))
            layers.append(nn.ReLU(inplace=True))
            in_channels = x

    return nn.Sequential(*layers)
```
해당 함수를 통해 각자에 맞는 hidden layer를 넣을 수 있었습니다.

```
class VGGNet(nn.Module):
    def __init__(self, mode):
        super(VGGNet, self).__init__()

        configs = {
            'A': vggnet_a_cfg,
            'B': vggnet_b_cfg,
            'C': vggnet_c_cfg,
            'D': vggnet_d_cfg,
            'E': vggnet_e_cfg
        }

        self.feature_layer = build_layer(configs[mode])

        self.classifier_layer = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )
```
train에서 사용하는 데이터가 cifar10이기 때문에 마지막 Linear에는 output을 10으로 줬습니다.

또 Classifier의 초반 Linear도 512 * 7 * 7이 아닌 512 * 1 * 1로 적용되었습니다.

## 학습결과
<img width="5367" height="1468" alt="image" src="https://github.com/user-attachments/assets/85e15a0f-4fb0-4260-ae1f-1dec2919ef23" />

맨 처음 훈련했을때는 위의 그래프처럼 D와 E가 전혀 학습이 되지 않은 모습을 볼 수 있습니다.

그래서 배치 정규화를 적용 시켰습니다.

<img width="5367" height="1468" alt="image" src="https://github.com/user-attachments/assets/60906732-d7a4-4d63-8602-3bbcea9361e8" />
그러자 위처럼 D와 E도 제대로 학습이 된 모습을 볼 수 있습니다.

하지만 그래프의 요동침이 마음에 들지 않아 0.001이었던 learning_rate를 0.0001로 낮추었습니다.

<img width="5367" height="1468" alt="image" src="https://github.com/user-attachments/assets/effba762-c043-4fa9-aa1c-0f5be827acf9" />

그러자 학습 손실에서는 굉장히 완만한 곡선을 갖게 되었습니다.


