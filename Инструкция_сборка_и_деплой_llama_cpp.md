# Сборка llama.cpp (master) для RTX 5060 Ti / sm_120 / CUDA 13 и деплой в Ollama

Проверенный пайплайн: `git pull origin master` → сборка dlopen-архитектуры с явными
CUDA 13 библиотеками → атомарная замена `/usr/local/lib/ollama`.

## 0. Требования системы
- NVIDIA RTX 5060 Ti (sm_120), драйвер ≥ 580
- CUDA Toolkit 13.x в `/usr/local/cuda-13.1`
- cmake, ninja, gcc, patchelf
- Ollama: бинарь `/usr/local/bin/ollama`, библиотеки в `/usr/local/lib/ollama/` (root)

## 1. Ключевые архитектурные решения (и почему)

### 1.1 `GGML_BACKEND_DL=ON` — обязательно для Ollama
- **OFF** (по умолчанию): бэкенды компилируются hard-link'ом в `libggml.so`.
  Ollama делает `dlopen("libggml-cuda.so")` из своего каталога — такого файла нет,
  CUDA 12/13 SONAME из build-dir не резолвятся → crash exit 127.
- **ON**: `libggml.so` становится «хабом», который dlopen'ит плагины
  (`libggml-cuda.so`, `libllama-*-impl.so`). Именно это ожидает Ollama.

### 1.2 `GGML_NATIVE=OFF` — обязательно для чужой машины / переноса
Автооптимизация под текущий CPU даёт бинарий, несовместимый с другими процессорами.
Для деплоя в системный каталог всегда OFF.

### 1.3 CUDA 12 vs 13 — главная ловушка сборки
**Симптом:** сборка «успешна», но `libggml-cuda.so` линкуется на SONAME из
`/usr/lib/x86_64-linux-gnu/` (CUDA 12, ставший системным пакетом), а не из
`/usr/local/cuda-13.1`.

**Причина:** CMake `FindCUDAToolkit` ищет библиотеки сначала в
`CUDAToolkit_IMPLICIT_LIBRARY_DIRECTORIES` — это пути GCC/host-linker, куда попадают
системные CUDA 12. Даже `-DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT` не переопределяет
найденные `*_LIBRARY` переменные кэша.

**Решение:** явный `-D` для каждой динамической библиотеки (полный список в §2).

### 1.4 Имя плагина
Региستر плагинов (`ggml-backend-reg.cpp`) ищет паттерн `libggml-<name>-*.so`, но есть
fallback на прямой `libggml-cuda.so`. Ollama передаёт свой `GGML_BACKEND_PATH` —
файл должен лежать **точно как `cuda_v13/libggml-cuda.so`** (без версионного суффикса).

## 2. Команда сборки (проверенная, build #3)

```bash
cd <fork> && git pull origin master
CUDA=/usr/local/cuda-13.1
cmake -S . -B build-master-cuda13fix \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="120" \
  -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc \
  -DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT=$CUDA \
  -DCUDA_cudart_LIBRARY=$CUDA/lib64/libcudart.so \
  ... и так далее для ВСЕХ библиотек из таблицы ниже
```

**Полный рабочий список `-D` для всех динамических CUDA-библиотек**
(имя переменной CMake → файл в `$CUDA/lib64/`):

| Переменная | Библиотека |
|---|---|
| `CUDA_cudart_LIBRARY`, `CUDART_LIBRARY`, `CUDA_CUDART` | libcudart.so |
| `CUDA_cublas_LIBRARY` | libcublas.so |
| `CUDA_cublasLt_LIBRARY` | libcublasLt.so |
| `CUDA_cufft_LIBRARY` | libcufft.so |
| `CUDA_cufftw_LIBRARY` | libcufftw.so |
| `CUDA_curand_LIBRARY` | libcurand.so |
| `CUDA_cusolver_LIBRARY` | libcusolver.so |
| `CUDA_cusparse_LIBRARY` | libcusparse.so |
| `CUDA_cupti_LIBRARY` | libcupti.so |
| `CUDA_nvrtc_LIBRARY` | libnvrtc.so |
| `CUDA_nvrtc_builtins_LIBRARY` | libnvrtc-builtins.so |
| `CUDA_nvjpeg_LIBRARY` | libnvjpeg.so |
| `CUDA_cufile_LIBRARY` | libcufile.so |
| `CUDA_nppc/nppial/nppicc/nppidei/nppif/nppig/nppim/nppist/nppisu/nppitc/npps_LIBRARY` | NPP (libnppX.so) |

> Если в вашем CMakeCache появились другие `CUDA_*_LIBRARY`, переопределите их
> аналогично. Проверка после сборки: `readelf -d libggml-cuda.so | grep NEEDED` —
> все SONAME должны быть `.so.13`, никаких `/usr/lib/x86_64-linux-gnu/` в ldd.

```bash
cmake --build build-master-cuda13fix -j$(nproc)
```

## 3. Контроль собранного бинаря (до деплоя!)

```bash
readelf -d build-*/bin/libggml-cuda.so | grep NEEDED   # SONAME .so.13
ldd     build-*/bin/cuda_v13/libggml-cuda.so           # 0 × "not found" (кроме libggml-base — см. §4)
```

Примечание: в изолированном `ldd` плагина из подкаталога `libggml-base.so.0 => not found` —
это **нормально**: при запуске её уже длопенул хаб `libggml.so`, динамический линкер
находит по SONAME в кэше загруженных библиотек.

## 4. Атомарный деплой в /usr/local/lib/ollama

```bash
TS=$(date +%s); STAGE=/usr/local/lib/ollama.new-$TS; echo $TS > /tmp/deploy-ts.txt
sudo cp -a /usr/local/lib/ollama "$STAGE"

# 4.1 Удалить старые артефакты build #N из стейджа (только верхний уровень + плагины)
cd "$STAGE"
sudo find . -maxdepth 1 \( -name 'libggml*' -o -name 'libllama*' -o -name 'libmtmd*' \
     -o -name 'llama-*' -o -name '*.so*' \) -exec rm -f {} +
# ⚠️ урок build #2: в список копирования ОБЯЗАТЕЛЬНО входят llama-server и
#    llama-quantize (а не только .so). Если забыть — бинари останутся старые.

# 4.2 Скопировать свежие артефакты ВСЕГО верхнего уровня build/bin + libs
B=build-master-cuda13fix/bin
sudo cp -a $B/lib*.so* "$STAGE"/        # все .so (хаб, impl-плагины)
sudo cp -a $B/llama-server $B/llama-quantize "$STAGE"/

# 4.3 Обновить CUDA-плагин в подкаталоге (CUDA runtime из cuda_v13 НЕ трогать — они свои .so.13)
sudo cp -a $B/libggml-cuda.so "$STAGE/cuda_v13/libggml-cuda.so"

# 4.4 patchelf: все RUNPATH → $ORIGIN (иначе линкер ищет build-dir на этой машине!)
for f in "$STAGE"/*.so* "$STAGE"/llama-server "$STAGE"/llama-quantize \
         "$STAGE/cuda_v13/libggml-cuda.so"; do
    sudo patchelf --set-rpath '$ORIGIN' "$f"
done

# 4.5 Атомарная свап-пара mv (одна ФС — атомарно)
sudo mv /usr/local/lib/ollama "/usr/local/lib/ollama.pre-swap-$TS"
sudo mv "$STAGE" /usr/local/lib/ollama
sudo chown -R root:root /usr/local/lib/ollama

# 4.6 Перезапуск и проверка
sudo systemctl restart ollama
journalctl -u ollama --since "1 min ago" | grep -E 'compute=12|CUDA0'   # sm_120 загружен
```

## 5. Финальная верификация

```bash
ldd /usr/local/lib/ollama/llama-server | grep -c 'not found'        # → 0
readelf -d /usr/local/lib/ollama/cuda_v13/libggml-cuda.so | grep NEEDED   # SONAME .so.13
# GPU-смоук: любой generate-запрос через API; в journald видно
# "inference compute ... library=CUDA compute=12.0 name=CUDA0"
```

## 6. Откат

```bash
TS=$(cat /tmp/deploy-ts.txt)
sudo mv /usr/local/lib/ollama "/usr/local/lib/ollama.failed-$(date +%s)"
sudo mv "/usr/local/lib/ollama.pre-swap-$TS" /usr/local/lib/ollama
sudo systemctl restart ollama
```

Резервные каталоги после деплоя: `ollama.old-*` (до первого деплоя),
`ollama.pre-swap-*` (сразу перед свапом). Удалять только после недели стабильной работы.

## 7. Частые ошибки → причины

| Симптом | Причина | Лечение |
|---|---|---|
| `libggml-cuda.so: cannot open shared object`, exit 127 | hard-link сборка (без DL) | §1.1, пересобрать с `GGML_BACKEND_DL=ON` |
| Плагины линкуются на CUDA 12 SONAME | FindCUDAToolkit нашёл системные пути GCC | явные `-D` (§1.3), проверить CMakeCache |
| После свапа binaрии старые / не те | в копировании забыт `llama-server`/`llama-quantize` или путь `$SRC/` | §4.2, всегда проверять по списку |
| «not found» на RUNPATH build-dir после деплоя | не прогнан patchelf | §4.4 |
