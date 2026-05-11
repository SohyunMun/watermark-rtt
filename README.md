# Watermark RTT Remover

Round-Trip Translation(RTT)을 활용하여 LLM 생성 텍스트에 삽입된 워터마크를 제거하는 도구입니다.

---

## 개요

LLM 워터마킹 기법(KGW, AWT, SynthID, Unigram)으로 삽입된 워터마크를 RTT를 통해 제거합니다. 워터마크가 삽입된 영어 텍스트를 입력하면, 기법별 최적 번역 경로로 왕복 번역을 수행한 뒤 워터마크 훼손 성능과 의미 보존 성능을 평가하여 출력합니다.

---

## 워크플로우

```
입력 텍스트 (워터마크가 삽입된 영어 텍스트)
        ↓
워터마크 기법 선택 (KGW / AWT / SynthID / Unigram)
        ↓
문장 단위 분절 (NLLB 번역 품질 한계 - 200토큰 기준)
        ↓
기법별 최적 경로로 RTT 수행 (NLLB-200)
  EN → 중간 언어 → EN
        ↓
평가 지표 계산 (Z-Score / PPL / SBERT / NLI)
        ↓
RTT 결과 텍스트 + 지표 출력 + 그래프 저장
```

---

## 기법별 RTT 최적 경로

| 기법 | 경로 |
|------|------|
| KGW | EN → JA → EN |
| AWT | EN → JA → EN |
| SynthID | EN → ZH → JA → EN |
| Unigram | EN → JA → EN |

경로는 자체 실험을 통해 의미 보존율과 워터마크 훼손 성능의 파레토 최적 기준으로 선정하였습니다.

---

## 평가 지표

| 지표 | 설명 | 방향 |
|------|------|------|
| **Z-Score** | 기법별 워터마크 감지 강도 | ↓ 낮을수록 워터마크 제거 |
| **PPL** | `facebook/opt-125m` 기반 텍스트 자연스러움 | ↓ 낮을수록 자연스러움 |
| **SBERT** | `all-MiniLM-L6-v2` 기반 의미 유사도 | ↑ 높을수록 의미 보존 |
| **NLI** | `cross-encoder/nli-deberta-v3-small` 기반 함의 확률 | ↑ 높을수록 의미 보존 |

> **Z-Score 계산 방식**
> - KGW: 이전 토큰 id를 seed로 green list 생성 후 z-score 계산 (`gamma=0.25`)
> - Unigram: 고정 seed(42)로 전체 vocab 50:50 분할 후 z-score 계산
> - SynthID: n-gram(5) + 20개 key 기반 hash 마스크 방식
> - AWT: 고정 seed(42)로 green set 생성 후 비율(wm_score, 0~1) 반환

---

## 파일 구조

```
watermark-rtt/
├── main.py          # CLI 진입점
├── app.py           # Gradio 웹 UI
├── config.py        # 모델명, 언어코드, 설정값
├── routes.py        # 기법별 RTT 경로
├── translator.py    # NLLB-200 번역 모듈 (분절/병합 포함)
├── segmenter.py     # 문장 단위 분절 (단어 단위 fallback 포함)
├── evaluator.py     # 평가 지표 계산
├── visualizer.py    # 결과 그래프 생성 및 저장
└── requirements.txt
```

---

## 설치 및 실행

### 환경 설정

```bash
git clone https://github.com/SohyunMun/watermark-rtt.git
cd watermark-rtt

python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
```

### CLI 실행

```bash
python main.py
```

```
Available methods: KGW, AWT, SynthID, Unigram
Select watermark method: KGW
Enter watermarked English text: (텍스트 입력)
```

### 웹 UI 실행

```bash
python app.py
```

브라우저에서 `http://localhost:7860` 접속

---

## 출력

- **콘솔**: RTT 결과 텍스트 + 평가 지표 수치
- **result.png**: Before/After 비교 그래프 (SBERT, NLI, PPL, Z-Score)
- **result.txt**: 실행 일시, 기법, 경로, 원본/RTT 텍스트, 지표 기록

---

## 의존 패키지

```
transformers
torch
sentencepiece
sentence-transformers
matplotlib
gradio
nltk
numpy
```

---

## 번역 모델

[facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (CC-BY-NC 4.0)

본 프로젝트는 연구 목적으로 제작되었습니다.

---

## 저자

Sohyun Mun
