"""Download the verified ONNX bundle from official Hugging Face; no package installs."""
from pathlib import Path
import argparse
import hashlib
import os
import tempfile
import urllib.request

REVISION = 'caf70d3af46478614209d86881a16e601d6bbf61'
FILES = {
    'relateanything.onnx': 'b8b6a047c5e0771a897a5015c2ffb09d8fe5e3ffa0651e1e61af0e8436a3617a',
    'relateanything.json': 'a5aa12ccd217337d6fdfff068943348e010ffec44035ee1b2a99d27085e34b14',
    'predicate_bank.npz': '708f812d6579eab1be85b3378b79e376453c2d4f65b4fad9bcebb9f5bf05da06',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Local model directory')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = output / name
        if target.exists():
            if digest(target) != expected:
                raise SystemExit(f'Refusing to overwrite existing unverified file: {target}')
            print(f'Already verified: {name}', flush=True)
            continue
        url = f'https://huggingface.co/maelic/relsgg-vits16plus/resolve/{REVISION}/{name}'
        print(f'Downloading {name} from official Hugging Face...', flush=True)
        fd, temporary = tempfile.mkstemp(prefix=name + '.', suffix='.part', dir=output)
        tmp = Path(temporary)
        try:
            h = hashlib.sha256()
            with os.fdopen(fd, 'wb') as dest, urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    dest.write(chunk)
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise RuntimeError(f'SHA256 mismatch for {name}; refusing to install downloaded file')
            # Hard-link atomically: fail if destination appeared while downloading, never replace it.
            os.link(tmp, target)
            print(f'Verified: {target}', flush=True)
        finally:
            tmp.unlink(missing_ok=True)
    print('All three model files are verified. Select relateanything.onnx in RA Load Local ONNX.')


if __name__ == '__main__':
    main()
