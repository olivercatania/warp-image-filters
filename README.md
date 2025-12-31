# Image Sharpening + Noise Removal (Warp + Python)

A small command-line image processing tool that supports:

- **Sharpening (Unsharp Mask)** (`-s`)
- **Noise removal (Median Filter)** (`-n`)

Uses **warp-lang** for compute kernels and **Pillow** for image I/O.

---

## Folder structure

```
.
├── main.py
├── pyproject.toml
├── uv.lock
├── all_images/
│   ├── in_images/     # put input images here
│   └── out_images/    # outputs are written here
└── .venv/             # local venv (do not commit)
```

---

## Requirements

- Python **3.11+**
- Dependencies are defined in `pyproject.toml` (managed with `uv`)

---

## Setup

From the repo root:

```bash
uv sync
```

---

## Usage

```bash
python main.py <algType> <kernSize> <param> <inFileName> <outFileName>
```

### Arguments

- `algType`
  - `-s` = sharpen (unsharp mask)
  - `-n` = noise removal (median filter)
- `kernSize`
  - positive **odd** integer (e.g., `3`, `5`, `7`)
- `param`
  - for `-s`: sharpening strength `k` (float)
  - for `-n`: not used, but still required (use `0`)
- `inFileName`
  - input image filename (place it in `all_images/in_images/`)
- `outFileName`
  - output image filename (saved to `all_images/out_images/`)

---

## Examples

### Sharpen an image

```bash
uv run python main.py -s 5 1.5 example.jpg sharpened.jpg
```

### Denoise an image

```bash
uv run python main.py -n 5 0 noisy.png denoised.png
```

Outputs will be saved to:

```
all_images/out_images/<outFileName>
```

---