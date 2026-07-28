from pathlib import Path
import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib import animation

# Allow running from repo root without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from nsf_fmrg_data import extract_final_thermal_frames

THERMAL_CMAP = "jet"
THERMAL_VMIN = 1000.0
THERMAL_VMAX = 2500.0
THERMAL_PIXEL_SIZE_MM = 0.014


def save_video(project_dir: Path, track_id: int, fps: int = 10) -> Path:
    thermal_dir = project_dir / "data" / "raw" / "thermal"
    result = extract_final_thermal_frames(thermal_dir, track_id)

    out_dir = project_dir / "processed_data" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"Track_{track_id}_thermal_20to100mm.mp4"
    frames = result["frames"]
    x_mm = result["x_mm_center"]

    extent = [
        0,
        frames.shape[2] * THERMAL_PIXEL_SIZE_MM,
        frames.shape[1] * THERMAL_PIXEL_SIZE_MM,
        0,
    ]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(
        frames[0],
        cmap=THERMAL_CMAP,
        vmin=THERMAL_VMIN,
        vmax=THERMAL_VMAX,
        extent=extent,
    )
    title = ax.set_title(f"Track {track_id} | x ≈ {x_mm[0]:.1f} mm")
    ax.set_xlabel("thermal local x (mm)")
    ax.set_ylabel("thermal local y (mm)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("temperature / intensity")

    def update(i):
        im.set_data(frames[i])
        title.set_text(f"Track {track_id} | x ≈ {x_mm[i]:.1f} mm")
        return [im, title]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / max(fps, 1),
        blit=False,
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=8000)
    ani.save(str(out_path), writer=writer)
    plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export thermal videos for one NSF FMRG track.")
    parser.add_argument("--project_dir", type=Path, default=Path("."), help="Repository/project root.")
    parser.add_argument("--track_id", type=int, default=8, help="Track ID, e.g., 8, 10, 14, or 21.")
    parser.add_argument("--fps", type=int, default=10, help="Output video frame rate.")
    args = parser.parse_args()

    out_path = save_video(args.project_dir.resolve(), args.track_id, args.fps)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
