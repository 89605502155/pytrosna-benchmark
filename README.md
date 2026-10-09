# pytrosna-benchmark

Проверка корректности и измерение производительности библиотеки
[pytrosna](https://pypi.org/project/pytrosna/) (файлы временных рядов
Trosna, `.trosna`). Материалы к статье А.О. Ферубко, О.Д. Казакова
«Библиотека языка Python для чтения, записи и редактирования файлов
временных рядов формата Trosna и ее сопряжение с библиотеками анализа данных».

Correctness checks and benchmarks of the [pytrosna](https://pypi.org/project/pytrosna/)
library (Trosna time-series files), accompanying the paper by A.O. Ferubko,
O.D. Kazakov “A Python library for reading, writing, and editing Trosna time
series files and its integration with data analysis libraries”.

## Содержимое / Contents

| Файл | Назначение |
|---|---|
| `benchmark.ipynb` | ноутбук с проверками и измерениями (с результатами выполнения) |
| `make_notebook.py` | исходный текст ячеек ноутбука |
| `results/benchmark.csv` | результаты измерений |
| `figures/` | рисунки, PNG и TIFF, 600 dpi |

## Запуск / Running

```sh
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace benchmark.ipynb
# или интерактивно / or interactively:
uv run jupyter lab benchmark.ipynb
```

Нужен Python 3.11 или новее; зависимости (pytrosna 0.1.1, pandas, Polars,
PyArrow, matplotlib, Jupyter) устанавливает `uv sync`.

## Лицензия / License

MIT
