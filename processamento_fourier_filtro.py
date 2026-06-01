import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    return img

def compute_dft(img: np.ndarray):
    img_float = np.float32(img)
    dft = np.fft.fft2(img_float)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(np.abs(dft_shift) + 1)
    return dft_shift, magnitude


def idft(dft_shift: np.ndarray) -> np.ndarray:
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    return img_back

def make_circular_mask(shape: tuple, radius: float, inside: bool = True) -> np.ndarray:
    rows, cols = shape
    cy, cx = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    mask = (dist <= radius).astype(np.float32)
    return mask if inside else (1 - mask)


def lowpass_filter(shape: tuple, cutoff: float) -> np.ndarray:
    return make_circular_mask(shape, cutoff, inside=True)


def highpass_filter(shape: tuple, cutoff: float) -> np.ndarray:
    return make_circular_mask(shape, cutoff, inside=False)


def bandpass_filter(shape: tuple, low_cut: float, high_cut: float) -> np.ndarray:
    inner = make_circular_mask(shape, low_cut,  inside=True)
    outer = make_circular_mask(shape, high_cut, inside=True)
    return (outer - inner).clip(0, 1)


def bandreject_filter(shape: tuple, low_cut: float, high_cut: float) -> np.ndarray:
    return 1 - bandpass_filter(shape, low_cut, high_cut)

def apply_filter(dft_shift: np.ndarray, mask: np.ndarray):
    dft_filtered = dft_shift * mask
    magnitude_map = 20 * np.log(np.abs(dft_filtered) + 1)
    img_result = idft(dft_filtered)
    return magnitude_map, img_result

def plot_all_results(img_original, magnitude_original, filters: list,
                     save_path: str = "resultados/comparativo_completo.png"):
    n = len(filters)
    fig, axes = plt.subplots(n + 1, 3, figsize=(15, 4 * (n + 1)),
                             facecolor="#0d0d0d")
    fig.suptitle("Transformada de Fourier e Filtros no Domínio da Frequência",
                 fontsize=15, color="white", fontweight="bold", y=0.99)

    def _show(ax, img, title, cmap="gray"):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="white", fontsize=9, pad=5)
        ax.axis("off")
        ax.set_facecolor("#1a1a1a")

    _show(axes[0, 0], img_original,       "Imagem Original")
    _show(axes[0, 1], magnitude_original, "Espectro de Fourier", cmap="inferno")
    axes[0, 2].axis("off")
    axes[0, 2].text(0.5, 0.5,
                    "Cada linha abaixo mostra:\n"
                    "① Máscara do filtro\n"
                    "② Espectro após filtragem\n"
                    "③ Imagem reconstruída",
                    ha="center", va="center", color="#aaaaaa",
                    fontsize=9, linespacing=1.9,
                    transform=axes[0, 2].transAxes)
    axes[0, 2].set_facecolor("#0d0d0d")

    for i, f in enumerate(filters):
        row = i + 1
        _show(axes[row, 0], f["mask"],               f"Máscara — {f['name']}")
        _show(axes[row, 1], f["magnitude_filtered"],  f"Espectro — {f['name']}", cmap="inferno")
        _show(axes[row, 2], f["img_result"],           f"Resultado — {f['name']}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Figura salva em: {save_path}")
    plt.show()


def plot_filter_effect(img_original, filter_info: dict, save_path: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d0d0d")
    fig.suptitle(f"Filtro: {filter_info['name']}", color="white", fontsize=13)

    for ax, title, img in zip(axes,
                               ["Imagem Original", f"Após {filter_info['name']}"],
                               [img_original, filter_info["img_result"]]):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, color="#cccccc", fontsize=10)
        ax.axis("off")
        ax.set_facecolor("#1a1a1a")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Salvo em: {save_path}")
    plt.show()

def plot_fourier_transform(img_original, magnitude, save_path: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d0d0d")
    fig.suptitle("Transformada de Fourier 2D", color="white", fontsize=13)

    for ax, title, img, cmap in zip(
        axes,
        ["Imagem Original", "Espectro de Fourier (magnitude log)"],
        [img_original, magnitude],
        ["gray", "inferno"]
    ):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="#cccccc", fontsize=10)
        ax.axis("off")
        ax.set_facecolor("#1a1a1a")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Salvo em: {save_path}")
    plt.show()

def main():
    os.makedirs("resultados", exist_ok=True)

    img_path = "Imagens/lena.png"

    img = load_image(img_path)
    print(f"Imagem carregada: {img_path}  |  shape: {img.shape}")

    dft_shift, magnitude_orig = compute_dft(img)
    shape = img.shape

    plot_fourier_transform(img, magnitude_orig,
                           save_path="resultados/transformada_fourier.png")

    r_low   = 30
    r_band1 = 20
    r_band2 = 60

    masks = {
        "Passa-baixa (D0=30px)":     lowpass_filter(shape, r_low),
        "Passa-alta (D0=30px)":      highpass_filter(shape, r_low),
        "Passa-banda (20px-60px)":   bandpass_filter(shape, r_band1, r_band2),
        "Rejeita-banda (20px-60px)": bandreject_filter(shape, r_band1, r_band2),
    }

    filters_data = []
    for name, mask in masks.items():
        mag_filt, img_result = apply_filter(dft_shift, mask)
        filters_data.append({
            "name":               name,
            "mask":               mask,
            "magnitude_filtered": mag_filt,
            "img_result":         img_result,
        })
        safe = name.split("(")[0].strip().lower().replace(" ", "_").replace("-", "")
        plot_filter_effect(img, {"name": name, "img_result": img_result},
                           save_path=f"resultados/resultado_{safe}.png")

    print("\n✔ Pipeline concluído. Resultados em ./resultados/")


if __name__ == "__main__":
    main()