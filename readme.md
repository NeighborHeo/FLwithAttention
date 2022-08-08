# Federated Learning with attention (A working title)

- 다기관에서, 각 기관별 모델에 대한 예측변수의 기여도가 이질적인 상황을 가정
- 전체 기관의 Attention Aggregation으로 종합된 예측 변수의 변수 기여도를 추출하고 평가 

## Setup

1. Clone this repo
2. "python -m venv .venv"
3. "source .venv/bin/activate"
4. "pip install -r requirements.txt"

## Results

To reproduce main results:

split terminal

```posh
cd 4_FL
```

terminal 1 : create server
```posh
python manage.py runserver
```

terminal 2 : create edge 1
```posh
python client.py --edge 0
```

terminal 3 : create edge 2
```posh
python client.py --edge 1
```

terminal 4 : create edge 3
```posh
python client.py --edge 2
```

result will be saved to `/result/eicu`

run `plot.ipynb` To plot a feather file as a heatmap.

Figures will be saved to `/results/eicu`. 

## Comparison of the results of learning with the entire data and the results of federated learning

Learning with the entire data (central).
```posh
python client.py --edge -1 --local true
```

## Citation

If you use any of this code in your work, please reference us:

```latex
@misc{ ,
      title={Federated Learning with attention}, 
      author={},
      year={},
      eprint={},
      archivePrefix={},
      primaryClass={}
}
```

---

### Notes


### Stack
