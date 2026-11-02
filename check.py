"""Run every chapter script and compare its output with outputs/chNN.txt."""
import glob, subprocess, sys
from pathlib import Path

root = Path(__file__).parent
bad = 0
for f in sorted(glob.glob(str(root / "code" / "ch*.py")) + glob.glob(str(root / "code" / "appD_*.py"))):
    name = Path(f).name.split("_")[0]
    out = subprocess.run([sys.executable, Path(f).name], cwd=root / "code",
                         capture_output=True, text=True).stdout
    ref = (root / "outputs" / f"{name}.txt").read_text()
    norm = lambda s: "\n".join(l.rstrip() for l in s.strip().splitlines())
    ok = norm(out) == norm(ref)
    bad += not ok
    print(f"{name}: {'ok' if ok else 'DIFFERS'}")
print("CLEAN" if bad == 0 else f"{bad} scripts differ")
sys.exit(1 if bad else 0)
