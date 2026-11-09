# api-sentinel-ml

## Treinar o modelo

Com o ambiente do projeto instalado, execute na raiz do repositório:

```bash
uv run python -m src.train
```

O treino carrega `data/dataset_1_train.json`, aplica as features do notebook e
avalia uma árvore de decisão com busca em grade e validação cruzada estratificada.
Para salvar o pipeline treinado (pré-processamento + modelo), informe um caminho:

```bash
uv run python -m src.train --model-output models/decision_tree.joblib
```
