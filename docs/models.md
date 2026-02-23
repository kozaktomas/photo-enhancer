# Model Guide

This document covers each model used in the Photo AI Processing Service: what it does, available variants, configuration, and weight sources.

## DDColor (Colorization)

**Endpoint:** `POST /colorize`
**Paper:** [DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders](https://arxiv.org/abs/2212.11613)
**Repository:** [github.com/piddnad/DDColor](https://github.com/piddnad/DDColor)

DDColor is an automatic image colorization model using dual decoders — a pixel decoder and a color decoder with learned color queries. It produces natural, vivid colorization without requiring user hints.

### Variants

| Variant | Env value | Model size | Description |
|---|---|---|---|
| `modelscope` | `MODEL_COLORIZE=modelscope` | large | **Default.** Best overall quality. Trained on ModelScope dataset. |
| `paper_tiny` | `MODEL_COLORIZE=paper_tiny` | tiny | Smaller and faster. From the original paper. |
| `artistic` | `MODEL_COLORIZE=artistic` | large | Artistic style with more saturated colors. |

### Parameters

- **`render_factor`** (1-100, default 35): Controls the internal processing resolution. Higher values process at higher resolution, producing more detailed colorization but using more memory.

### Weight Sources

All DDColor weights are hosted on HuggingFace:

| Variant | URL |
|---|---|
| `paper_tiny` | `huggingface.co/piddnad/DDColor-models/.../ddcolor_paper_tiny.pth` |
| `modelscope` | `huggingface.co/piddnad/DDColor-models/.../ddcolor_modelscope.pth` |
| `artistic` | `huggingface.co/piddnad/DDColor-models/.../ddcolor_artistic.pth` |

### Technical Notes

- Uses the `ddcolor` pip package with `ColorizationPipeline` for inference
- Input size is fixed at 512×512 internally (render_factor controls this)
- The model variant maps to model size: `paper_tiny` → `tiny`, `modelscope`/`artistic` → `large`

---

## NAFNet (Restoration)

**Endpoint:** `POST /restore`
**Paper:** [Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676)
**Repository:** [github.com/megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)

NAFNet (Nonlinear Activation Free Network) is an image restoration model that achieves state-of-the-art results with a simplified architecture. It removes the need for nonlinear activation functions in the main network body.

### Variants

| Variant | Env value | Trained on | Use case |
|---|---|---|---|
| `denoise` | `MODEL_RESTORE=denoise` | SIDD dataset | **Default.** Remove sensor noise from photos. Best for photos taken in low light or with high ISO. |
| `deblur` | `MODEL_RESTORE=deblur` | GoPro dataset | Remove motion blur. Best for photos with camera shake or subject motion. |

Both variants use `width=64` architecture.

### Parameters

- **`tile_size`** (>=0, default 0): Tile size for processing. `0` means no tiling (whole image at once). Set to 256 or 512 to reduce VRAM usage on large images.

### Weight Sources

NAFNet weights are hosted on Google Drive:

| Variant | Filename | Source |
|---|---|---|
| `denoise` | `NAFNet-SIDD-width64.pth` | Google Drive |
| `deblur` | `NAFNet-GoPro-width64.pth` | Google Drive |

Google Drive URLs are resolved via `drive.usercontent.google.com` to bypass the virus-scan interstitial that Google shows for large files.

### Technical Notes

- Architecture (block counts, width) is inferred from checkpoint keys — no hard-coded config per variant
- The wrapper counts encoder/decoder/middle blocks from key patterns like `encoders.0.1.conv1.weight`
- Width is read from `intro.weight` shape

---

## CodeFormer (Face Restoration)

**Endpoint:** `POST /face-restore`
**Paper:** [Towards Robust Blind Face Restoration with Codebook Lookup Transformer](https://arxiv.org/abs/2206.11253)
**Repository:** [github.com/sczhou/CodeFormer](https://github.com/sczhou/CodeFormer)

CodeFormer is a face restoration model that uses a learned codebook (VQ-GAN) combined with a transformer to restore degraded faces. It handles severe degradation: blur, noise, compression artifacts, low resolution, and old photo damage.

### Variants

| Variant | Env value | Description |
|---|---|---|
| `v0.1` | `MODEL_FACE=v0.1` | **Default.** Only available variant. The v0.1.0 release weights. |

### Parameters

- **`fidelity`** (0.0-1.0, default 0.5): Controls the balance between restoration quality and input fidelity. The `w` parameter passed to the CodeFormer model.
  - `0.0` = Maximum quality restoration (may change facial features)
  - `0.5` = Balanced (default)
  - `1.0` = Maximum fidelity to input (less restoration)
- **`upscale`** (1-4, default 2): Upscale factor for the output image.

### Face Detection Pipeline

CodeFormer uses `facexlib` (RetinaFace) for face handling:

1. **Detect** all faces using `retinaface_resnet50` (downloads ~100MB detection model on first use)
2. **Align** each face to a canonical 512×512 crop using affine transform based on 5 facial landmarks
3. **Restore** each face through the CodeFormer model
4. **Preserve color**: Keep luminance from restored face, chrominance from original (prevents color artifacts on B&W/sepia inputs)
5. **Paste back** into the original image using inverse affine transform

If no faces are detected, the image is returned with a simple bicubic upscale.

### Weight Sources

| Variant | URL |
|---|---|
| `v0.1` | `github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth` |

### Technical Notes

- Architecture: `dim_embd=512`, `codebook_size=1024`, `n_head=8`, `n_layers=9`
- Uses a VQ-GAN autoencoder (vendored in `models/archs/vqgan_arch.py`)
- Face detection model (`retinaface_resnet50`) is downloaded separately by `facexlib` into `TORCH_HOME`

---

## Real-ESRGAN (Upscaling)

**Endpoint:** `POST /upscale`
**Paper:** [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
**Repository:** [github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

Real-ESRGAN is a super-resolution model that upscales images while adding realistic details. It uses the RRDBNet (Residual-in-Residual Dense Block Network) architecture and is trained on synthetic degradation to handle real-world low-quality images.

### Variants

| Variant | Env value | Scale | Description |
|---|---|---|---|
| `x4plus` | `MODEL_UPSCALE=x4plus` | 4× | **Default.** Best for real-world photos. |
| `x4anime` | `MODEL_UPSCALE=x4anime` | 4× | Optimized for anime/illustration style images. Uses 6 RRDB blocks (vs 23 for x4plus). |
| `x2plus` | `MODEL_UPSCALE=x2plus` | 2× | 2× upscale for photos. Uses pixel unshuffle preprocessing. |

### Parameters

- **`scale`** (1-8, default 4): Desired upscale factor. Note: the model has a native scale (2× or 4× depending on variant). If `scale` differs from the native scale, the output is the model's native scale.
- **`tile_size`** (>=0, default 512): Tile size for tiled processing. `0` = no tiling. Default is 512 to prevent OOM on typical inputs.

### Tiling

Real-ESRGAN supports tiled processing for large images via `_tile_process()`:

- Image is divided into a grid of `tile_size × tile_size` tiles
- Each tile gets 10px overlap padding for seamless stitching
- Tiles are processed independently, keeping VRAM bounded
- Default tile size is 512 (set at the endpoint level)

### Weight Sources

All Real-ESRGAN weights are hosted on GitHub Releases:

| Variant | URL |
|---|---|
| `x4plus` | `github.com/xinntao/Real-ESRGAN/releases/.../RealESRGAN_x4plus.pth` |
| `x4anime` | `github.com/xinntao/Real-ESRGAN/releases/.../RealESRGAN_x4plus_anime_6B.pth` |
| `x2plus` | `github.com/xinntao/Real-ESRGAN/releases/.../RealESRGAN_x2plus.pth` |

### Technical Notes

- Architecture (num_feat, num_block, num_grow_ch, scale) is inferred from checkpoint keys
- Scale detection: `conv_first` input channels reveal the scale — 3 channels → 4× (raw input), 12 channels → 2× (pixel unshuffle ×2)
- The `x4anime` variant uses 6 RRDB blocks; `x4plus` and `x2plus` use 23 blocks

---

## Old Photo Restore (Old Photo Restoration)

**Endpoint:** `POST /old-photo-restore`
**Paper:** [Bringing Old Photos Back to Life](https://arxiv.org/abs/2004.09484) (CVPR 2020)
**Repository:** [github.com/microsoft/Bringing-Old-Photos-Back-to-Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)
**License:** MIT

A multi-stage pipeline for restoring old and damaged photographs. Automatically detects scratches, performs global quality restoration, and enhances faces. Three sub-networks work together: a UNet for scratch detection, a VAE + mapping network for global restoration, and a SPADE generator for face enhancement.

### Variants

| Variant | Env value | Description |
|---|---|---|
| `v1` | `MODEL_OLD_PHOTO=v1` | **Default.** Original paper weights from HuggingFace. |

### Parameters

- **`with_scratch`** (bool, default `true`): Enable automatic scratch detection and repair. Set to `false` for photos without physical damage (fading, color loss only).
- **`with_face`** (bool, default `true`): Enable face enhancement via SPADE generator. Detected faces are aligned, enhanced, and blended back.
- **`scratch_threshold`** (0.0-1.0, default `0.4`): Sensitivity for scratch detection. Lower values detect more scratches (more aggressive repair).

### Pipeline Stages

1. **Scratch Detection**: Input is converted to grayscale, resized to 256x256, and fed through a UNet. The output probability map is thresholded and dilated to produce a binary scratch mask.
2. **Global Restoration**: The image and scratch mask are resized to 256x256. VAE_A encodes the image, the mapping network translates features with scratch-mask awareness (non-local attention blocks zero out attention to scratched regions), and VAE_B decodes the translated features.
3. **Face Enhancement**: dlib detects faces and extracts 68-point landmarks. Each face is affine-aligned to 256x256, enhanced by the SPADE generator, inverse-warped back, and blended into the result with feathered edges.

### Weight Sources

This is a multi-file model (6 weight files). All hosted on HuggingFace:

| File | Description |
|---|---|
| `scratch_detection.pt` | UNet scratch detection network |
| `vae_a_encoder.pth` | VAE A (quality) encoder/decoder |
| `vae_b_decoder.pth` | VAE B (scratch) encoder/decoder |
| `mapping_net.pth` | Feature mapping network with mask attention |
| `face_enhance_gen.pth` | SPADE face enhancement generator |
| `shape_predictor_68_face_landmarks.dat` | dlib 68-point face landmark predictor |

### Technical Notes

- Uses `dlib` for face detection and landmark extraction (requires `cmake` at build time)
- Architecture files are vendored in `models/archs/old_photo_detect_arch.py`, `old_photo_global_arch.py`, and `old_photo_face_arch.py`
- `SynchronizedBatchNorm2d` from the original repo is replaced with `nn.BatchNorm2d` (identical parameter names, checkpoint keys match with `strict=True`)
- Face alignment uses OpenCV `cv2.estimateAffinePartial2D()` instead of scikit-image to avoid an extra dependency
- Unlike single-file models, weights are downloaded via `ensure_model_files_exist()` which returns a directory path

---

## LaMa (Inpainting)

**Endpoint:** `POST /inpaint`
**Paper:** [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161) (WACV 2022)
**Repository:** [github.com/advimman/lama](https://github.com/advimman/lama)

LaMa (Large Mask Inpainting) fills in arbitrary masked regions of an image with realistic content. It uses Fast Fourier Convolutions (FFC) to capture global image context, enabling it to fill large missing areas convincingly — unlike patch-based methods that struggle with large holes.

### Variants

| Variant | Env value | Description |
|---|---|---|
| `big` | `MODEL_INPAINT=big` | **Default.** The "big-lama" checkpoint (~206 MB). |

### Parameters

The inpaint endpoint accepts a **`points`** query parameter (JSON array of `[x,y]` polygon vertices, at least 3). The polygon interior is rasterized into a binary mask where white (255) = inpaint, black (0) = keep.

### Weight Sources

| Variant | URL |
|---|---|
| `big` | `huggingface.co/JosephCatrambone/big-lama-torchscript/.../lama.pt` |

### Technical Notes

- **TorchScript JIT model**: Loaded with `torch.jit.load()` — no architecture vendoring needed, no new pip dependencies
- **FFC architecture**: Uses Fast Fourier Convolutions for global receptive field, enabling coherent filling of large masked regions
- **Pad-to-8**: Input is padded to a multiple of 8 pixels (reflected padding) before inference, then cropped back
- **Input format**: 4-channel tensor — image (3ch RGB) + binary mask (1ch) concatenated along channel dimension
- **Blending**: Output is blended with the original image — only masked regions use the model output, non-masked regions keep original pixels

---

## Weight Download & Caching

### Download Process

1. On startup, `ensure_model_exists(category, variant)` checks if the weight file exists at `/app/weights/<category>/<filename>`. For multi-file models, `ensure_model_files_exist(category, variant)` checks all files in the directory.
2. If missing, downloads from the URL in `MODEL_URLS` (single-file) or `MODEL_URLS_MULTI` (multi-file)
3. Downloads use `.part` file extension during transfer — if interrupted, the partial file is cleaned up
4. After download, the file header is validated (rejects HTML responses that indicate failed downloads)
5. Existing files are also validated on startup to catch previously corrupted downloads

### Storage Layout

```
/app/weights/
├── colorize/
│   └── ddcolor_modelscope.pth
├── restore/
│   └── NAFNet-SIDD-width64.pth
├── face/
│   └── codeformer.pth
├── upscale/
│   └── RealESRGAN_x4plus.pth
├── inpaint/
│   └── big-lama.pt
├── old_photo_restore/         # Multi-file model (6 weight files)
│   ├── scratch_detection.pt
│   ├── vae_a_encoder.pth
│   ├── vae_b_decoder.pth
│   ├── mapping_net.pth
│   ├── face_enhance_gen.pth
│   └── shape_predictor_68_face_landmarks.dat
├── .torch/          # TORCH_HOME — facexlib detection model cache
└── .huggingface/    # HF_HOME — HuggingFace cache
```

### Filename Resolution

Most weight files use the filename from the download URL. For Google Drive URLs (which don't have meaningful filenames), `FILENAME_OVERRIDES` in `downloader.py` provides explicit names:

- `restore/denoise` → `NAFNet-SIDD-width64.pth`
- `restore/deblur` → `NAFNet-GoPro-width64.pth`
