# ExLlamaV2 CUDA Optimization Guide

Этот документ описывает настройки для оптимизации ExLlamaV2 с использованием CUDA.

## Настройки

### 1. FlashAttention ON
FlashAttention включен по умолчанию, если установлен пакет `flash-attn`. Для проверки:

```bash
pip install flash-attn
```

Если FlashAttention установлен, ExLlamaV2 автоматически использует его для ускорения вычислений внимания.

### 2. GPU Memory Utilization: 0.90 (90%)
Используется параметр `reserve_vram` в `load_autosplit()` для контроля использования памяти GPU.

- **90% utilization** означает, что модель будет использовать 90% доступной памяти GPU
- Остальные 10% резервируются для системных нужд и предотвращения OOM ошибок

### 3. No CPU Offload
По умолчанию ExLlamaV2 загружает модель полностью на GPU. CPU offload не используется, что обеспечивает максимальную производительность.

### 4. FP16 Compute
KV-cache использует FP16 (half precision) по умолчанию через `ExLlamaV2Cache`, который создает тензоры с `dtype=torch.half`.

### 5. KV-cache Pinned
KV-cache тензоры размещаются непосредственно на GPU, что оптимально для инференса. Pinned memory обычно используется только для CPU-GPU трансферов, которые не требуются при GPU-only инференсе.

## Использование

### Базовый пример

```python
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator
import torch

# Получить информацию о GPU памяти
def get_gpu_memory():
    gpu_info = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory
        reserved = torch.cuda.memory_reserved(i)
        free = total - reserved
        gpu_info[i] = {
            'total': total,
            'free': free
        }
    return gpu_info

# Вычислить reserve_vram для 90% использования
gpu_info = get_gpu_memory()
gpu_utilization = 0.90
reserve_vram = [
    int(gpu_info[i]['total'] * (1.0 - gpu_utilization))
    for i in sorted(gpu_info.keys())
]

# Инициализация модели
config = ExLlamaV2Config("path/to/model")
config.prepare()

model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy=True)

# Загрузка с autosplit и 90% использованием памяти
model.load_autosplit(
    cache,
    reserve_vram=reserve_vram,
    progress=True
)

tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)
generator.warmup()

# Генерация
output = generator.generate(
    prompt="Hello, how are you?",
    max_new_tokens=200
)
```

### Использование готового скрипта

```bash
python examples/inference_cuda_optimized.py \
    -m /path/to/model \
    -u 0.90 \
    -p "Your prompt here" \
    -t 200
```

Параметры:
- `-m, --model_dir`: Путь к директории модели (обязательно)
- `-u, --gpu_utilization`: Использование GPU памяти (0.0-1.0, по умолчанию 0.90)
- `-p, --prompt`: Промпт для генерации
- `-t, --tokens`: Максимальное количество токенов для генерации

## Проверка настроек

### Проверка FlashAttention

```python
from exllamav2.attn import has_flash_attn
print(f"FlashAttention available: {has_flash_attn}")
```

### Проверка использования памяти

```python
import torch

for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1024**3
    reserved = torch.cuda.memory_reserved(i) / 1024**3
    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
    utilization = (reserved / total) * 100

    print(f"GPU {i}: {utilization:.1f}% utilized")
```

## Требования

1. **PyTorch с CUDA**: Установите PyTorch с поддержкой CUDA
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **FlashAttention** (опционально, но рекомендуется):
   ```bash
   pip install flash-attn
   ```

3. **CUDA Toolkit**: Убедитесь, что установлен CUDA Toolkit и переменная окружения `CUDA_HOME` настроена

## Примечания

- **GPU Memory Utilization**: Значение 0.90 означает использование 90% памяти. Оставшиеся 10% резервируются для предотвращения OOM ошибок
- **FP16**: Используется по умолчанию для KV-cache, что экономит память и ускоряет вычисления
- **Pinned Memory**: Для GPU-only инференса pinned memory не требуется, так как все тензоры находятся на GPU
- **CPU Offload**: Отключен по умолчанию для максимальной производительности

## Troubleshooting

### Ошибка: CUDA out of memory
- Уменьшите `gpu_utilization` (например, до 0.85)
- Используйте меньший `max_seq_len` в конфигурации
- Используйте квантованные модели (EXL2, GPTQ)

### FlashAttention не работает
- Убедитесь, что установлен: `pip install flash-attn`
- Проверьте совместимость версии FlashAttention с вашей версией PyTorch
- Убедитесь, что GPU поддерживает FlashAttention (Ampere или новее)

### Низкая производительность
- Убедитесь, что FlashAttention включен
- Проверьте, что модель полностью загружена на GPU (нет CPU offload)
- Используйте `generator.warmup()` перед генерацией для оптимизации ядер
