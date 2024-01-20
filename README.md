# AI Tech 6기 Team 아웃라이어

## Members
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/kangshwan">
        <img src="https://imgur.com/ozd1yor.jpg" width="100" height="100" /><br>
        강승환
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/viitamin">
        <img src="https://imgur.com/GXteBDS.jpg" width="100" height="100" /><br>
        김승민
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/tjfgns6043">
        <img src="https://imgur.com/aMVcwCF.jpg" width="100" height="100" /><br>
        설훈
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/leedohyeong">
        <img src="https://imgur.com/F6ZfcEl.jpg" width="100" height="100" /><br>
        이도형
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/wjsqudrhks">
        <img src="https://imgur.com/ZSVCV82.jpg" width="100" height="100" /><br>
        전병관
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/seonghyeokcho">
        <img src="https://imgur.com/GBdY0k4.jpg" width="100" height="100" /><br>
        조성혁
      </a>
    </td>
  </tr>
</table>

## 재활용 품목 분류를 위한 Object Detection

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

![쓰레기 재활용품 사진](https://imgur.com/ldk2oSk.jpg)


분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

이번 프로젝트는 부스트캠프 AI Tech CV 트랙내에서 진행된 대회이며 mAP50으로 최종평가를 진행하게 됩니다.

## Ground Rules
### [Conventional Commits 1.0.0](https://www.conventionalcommits.org/ko/v1.0.0/)
```bash
<타입>[적용 범위(선택 사항)]: <설명>

[본문(선택 사항)]

[꼬리말(선택 사항)]
```

#### Types
- fix | feat | BREAKING CHANGE | build | chore | ci | docs | style | refactor | test | release
  - fix : 기능에 대한 버그 수정
  - feat : 새로운 기능 추가, 기존의 기능을 요구 사항에 맞추어 수정
  - build : 빌드 관련 수정
  - chore : 패키지 매니저 수정, 그 외 기타 수정 ex) .gitignore
  - ci : CI 관련 설정 수정
  - docs : 문서(주석) 수정
  - style : 코드 스타일, 포맷팅에 대한 수정
  - refactor : 기능의 변화가 아닌 코드 리팩터링 ex) 변수 이름 변경
  - test : 테스트 코드 추가/수정
  - release : 버전 릴리즈

## Requirements
* Python >= 3.10.13
* PyTorch >= 1.12.1
* mmcv-full >= 1.6.2

## Folder Structure
  ```
baseline
├── EDA
│   └── eda.ipynb
│   
├── mmdetection
│   ├── configs
|   │   └── Outliers
|   ├── work_dirs                                     #체크포인트가 저장되는 폴더입니다.
|   ├── experient_mmdetection_v3.ipynb
|   ├── gradcam-faster-rcnn-C4-proposal.ipynb
│   ├── metrics.py
│   ├── supervisly_to_coco.py
│   ├── StratifiedGroupKFold.py
│   └── main.py
│
├── requirements.txt
  ```
## Dataset
- Total Images : 9754장 (train : 4883, test : 4871)
- 10 Class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- Image Size : (1024, 1024)
- COCO Format

## Main Script('main.py')
- main.py에서는 제공된 config를 바탕으로 훈련 환경을 설정하고, 모델, 데이터셋, 그리고 train/test 절차를 초기화합니다.

### 주요 특징
- '--amp' : 혼합 정밀도 훈련을 설정하는 config로 default값은 0으로 0을 제외한 값을 사용시 amp를 적용합니다.
- '--root', '--annotation', '--valid_annotation' : 데이터셋 경로와 어노테이션 경로를 설정합니다.
- '--output' : 실험 결과를 저장할 디렉토리를 지정합니다.
- '--load_from' : 사전 훈련된 모델이나 특정 checkpoint를 불러옵니다.
- '--wandb_name' : wandb사용시 name을 지정해줍니다.

## Using Shell Script
```
ROOT="path/to/dataset/"
ANNOTATION="your_annotation.json"
VALID_ANNOTATION="your_validation_annotation.json"
OUTPUT="path/to/output_directory"
AMP=0  # 혼합 정밀도 훈련을 위해 1로 설정
LOAD_FROM="path/to/pretrained_model_or_checkpoint"
TRAIN=1  # 추론을 위해 0으로 설정
CONFIG_DIR="path/to/your_config_file.py"
WANDB_NAME="YourExperimentName"

python main.py \
--root ${ROOT} \
--annotation ${ANNOTATION} \
--valid_annotation ${VALID_ANNOTATION} \
--output ${OUTPUT} \
--amp ${AMP} \
--load_from ${LOAD_FROM} \
--train ${TRAIN} \
--config_dir ${CONFIG_DIR} \
--wandb_name ${WANDB_NAME}
```
다음과 같은 쉘 스크립트를 사용하여 실험을 진행합니다.

### Wandb Visualization
This template supports Wandb visualization by using [Wandb](https://github.com/wandb/wandb) library.

#### Quickstart

Get started with W&B in four steps:

1. First, sign up for a [free W&B account](https://wandb.ai/login?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=quickstart).

2. Second, install the W&B SDK with [pip](https://pip.pypa.io/en/stable/). Navigate to your terminal and type the following command:

```bash
pip install wandb
```

3. Third, log into W&B:

```bash
wandb init
```

4. Setting WANDB_NAME of Shell Script and enjoy
```train.sh
WANDB_NAME="YourExperimentName"
```



That's it! Navigate to the W&B App to view a dashboard of your first W&B Experiment. Use the W&B App to compare multiple experiments in a unified place, dive into the results of a single run, and much more!

<p align='center'>
<img src="https://github.com/wandb/wandb/blob/main/docs/README_images/wandb_demo_experiments.gif?raw=true" width="100%">
</p>
<p align = "center">
Example W&B Dashboard that shows Runs from an Experiment.
</p>

&nbsp;

