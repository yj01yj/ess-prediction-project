# ESS 배터리 수명 예측 
목적 작성 


## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12)
- 평가 데이터 : Batch 2 (2018-02-20)
- 태스크 : Regression (Cycle Life 예측)


## 파일 구조 (sample) 
```
├── data/
│   └── README.md          
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── train.py
├── results/
│   └── model_performance.csv
├── requirements.txt
└── README.md
```


## 환경 설정 (sample) 
```bash
git clone https://github.com/팀명/ess-battery-project
cd ess-prediction-project
pip install -r requirements.txt
```


## EDA 

- Cycle Life 분포
	- 분포 형태 및 장단수명 비율 요약:

        - Batch 1: 534~1227 사이클 범위를 가지며, 평균 844.7로 중수명 중심의 분포를 보입니다. 장수명(>1,000) 비율은 약 21.7%입니다. 

        - Batch 2: 392~1186 사이클 범위로 세 배치 중 수명이 가장 짧으며, 평균 565.7의 단수명 중심 분포입니다. 특히 **단수명(<500) 비율이 71.8%**로 매우 높습니다. 
        
        - Batch 3: 541~1935 사이클로 가장 넓고 긴 수명 범위를 보이며, 평균 1059.7로 장수명 쪽에 치우쳐 있습니다. 장수명 비율이 52.3%에 달합니다. 
    
    - 핵심 발견:
    
        세 배치는 동일 분포가 아니며, 명확한 **Batch Shift(배치 간 차이)**가 존재하므로 모델링 시 이를 고려한 편향 확인이 필수적입니다. 

- 열화 곡선 분석
	- 장수명 vs 단수명 셀의 열화 속도 차이:Batch 2(단수명 중심)는 더 이른 사이클에서 용량 감소가 시작되지만, Batch 3(장수명 중심)는 훨씬 오랫동안 높은 용량을 유지합니다. 대부분의 셀이 초기에는 완만하게 감소하다가 특정 시점 이후 열화가 가속되는 비선형 패턴을 보입니다. Knee point 존재 여부 및 발생 시점:열화가 급격히 빨라지는 전환점인 Knee point가 탐지되며, 모든 배치에서 가장 이른 Knee cycle은 약 80 사이클 부근에서 나타납니다. Batch 1(37개)과 Batch 3(44개)에서는 Knee가 많은 셀에서 검출되지만, Batch 2(5개)는 검출되는 셀 수가 적어 전환 시점이 덜 뚜렷합니다. 핵심 발견: 열화는 선형적이지 않으며, **Knee point 이후의 가속 지표(Fade acceleration)**를 feature로 활용하는 것이 수명 예측에 더 적절합니다. 

- ΔQ(V) 곡선 분석
	- Cycle 100 - Cycle 10 차이 곡선 형태
	- 장단수명 셀 간 ΔQ 형태 비교
	- 핵심 발견 :

- 충전 속도(C-rate)와 수명의 관계
	- 충전 프로토콜별 평균 수명 비교 결과
	- 핵심 발견 :

- (추가 확인한 내용 작성) 


## Modeling 

### 피처 엔지니어링 전략
EDA 결과를 바탕으로 선택한 피처와 그 근거를 기술


### 모델 선택 및 근거
- 후보 모델 : 
- 최종 모델 :
- 선택 이유 :


## 성능 결과
Format에 맞춰 작성


## 오류 분석
- 모델이 가장 크게 틀린 셀의 공통점
- 원인 가설 및 개선 방향


## ESS 도메인 해석
분석 결과를 실제 ESS 운영 관점에서 해석

- 이 모델을 실제 BESS에 적용한다면 어떤 의사결정에 활용 가능한가?
- 어떤 한계가 있으며, 실 배포를 위해 추가로 필요한 것은 무엇인가?


## 참고문헌
- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.


## 팀 구성
- 김영희 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch2)
- 박철수 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch3)