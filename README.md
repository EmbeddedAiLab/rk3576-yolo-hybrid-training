
```
g++ -O3 -std=c++17 -fopenmp train_main_improved.cpp -o train_main_improved $(pkg-config --cflags --libs opencv4) -lrknnrt -lpthread
```


sudo OMP_NUM_THREADS=4 taskset -c 4-7 ./train_main_v2 dataset_big 30 0.005 | tee train_v2.txt

sudo HYB_RESUME=1 OMP_NUM_THREADS=4 taskset -c 4-7 ./train_main_v2 dataset_big 30 0.005 | tee -a train_v2.txt

HYB_CPU=1 taskset -c 0-3 ./detect snap.bin ./image.png 0.4 r.jpg
HYB_CPU=1 taskset -c 0-3 ./detect snap.bin dataset_big/images/val/000000002229.jpg 0.4 r.jpg

HYB_CPU=1 taskset -c 0-3 ./detect snap.bin dataset_big/images/val/000000001401.jpg 0.33 r.jpg



## Diagram 

```

       RK3576 ROCK 4D — HYBRID AI TRAINING
  ┌───────────────────────────────────────────┐
  │             DATASET INPUT                  │
  │  Images + YOLO Bounding-Box Labels         │
  │  Keyboard | Mouse | Bottle | Cup | Person  │
  └─────────────────────┬─────────────────────┘
                        │
                        ▼
  ┌───────────────────────────────────────────┐
  │          CPU: DATA PREPROCESSING           │
  │  OpenCV Resize: 192 × 192                  │
  │  RGB Conversion + Normalization            │
  │  Random Zoom / Shift / Flip / Color Jitter │
  │  Bounding-Box Augmentation                 │
  └─────────────────────┬─────────────────────┘
                        │
                        ▼
  ┌───────────────────────────────────────────┐
  │       YOLO-LIKE CNN: FORWARD PASS          │
  │                                           │
  │  Conv 1:  3  →  16 channels                │
  │  Conv 2: 16  →  32 channels                │
  │  Conv 3: 32  →  64 channels                │
  │  Conv 4: 64  → 128 channels                │
  │  Conv 5:128  → 128 channels                │
  │  Conv 6:128  → 128 channels                │
  │  Conv 7:128  → 128 channels                │
  │  Conv 8:128  →  10 channels*               │
  │                                           │
  │  Output: 12 × 12 grid                      │
  │  Each cell: Objectness + Box + Class       │
  └─────────────────────┬─────────────────────┘
                        │
                        ▼
       ┌────────────────────────────────┐
       │      RK3576 NPU ACCELERATOR    │
       │                                │
       │  INT8 Matrix Multiplication    │
       │  im2col → Quantization          │
       │  RKNN MatMul API                │
       │  INT32 Accumulation             │
       │  Dequantization + Bias          │
       └────────────────┬───────────────┘
                        │
                        ▼
       ┌────────────────────────────────┐
       │        CPU: TRAINING            │
       │                                │
       │  Detection Loss                 │
       │  Backpropagation (Float32)      │
       │  Straight-Through Estimator     │
       │  Gradient Clipping              │
       │  SGD + Momentum + Weight Decay  │
       └────────────────┬───────────────┘
                        │
                        ▼
       ┌────────────────────────────────┐
       │      UPDATE MODEL WEIGHTS       │
       │                                │
       │  Quantize Updated Weights       │
       │  Upload Weights to NPU          │
       │  Repeat for Next Training Step  │
       └────────────────┬───────────────┘
                        │
                        ▼
  ┌───────────────────────────────────────────┐
  │         VALIDATION ON CPU + NPU            │
  │  Validation Loss | Precision | Recall      │
  │  F1 Score | Confidence Filtering           │
  │  Non-Maximum Suppression (NMS)             │
  └─────────────────────┬─────────────────────┘
                        │
                        ▼
  ┌───────────────────────────────────────────┐
  │              MODEL CHECKPOINTS             │
  │  yolo_hybrid_last.bin                      │
  │  yolo_hybrid_best.bin                      │
  │  yolo_hybrid_ckpt.bin                      │
  └───────────────────────────────────────────┘

  ```