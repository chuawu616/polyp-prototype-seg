"""Download third-party code that this repository may not redistribute.

EMCAD (Rahman et al., CVPR 2024) is released under the UT Austin Research License, which permits academic use
but not redistribution, so its decoder is fetched from the official repository at a pinned commit:

  python tools/fetch_third_party.py
"""
import hashlib
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [
    dict(url='https://raw.githubusercontent.com/SLDGroup/EMCAD/26c9c31f73/lib/decoders.py',
         dst='protoseg_polyp/models/third_party/emcad_decoders.py',
         sha256='4b36981117376ea005aed8844f4ea9c846f432286695b85c1530797a20157a18',
         license='https://github.com/SLDGroup/EMCAD/blob/main/LICENSE'),
]


def main():
    for f in FILES:
        dst = os.path.join(ROOT, f['dst'])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        data = urllib.request.urlopen(f['url'], timeout=60).read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != f['sha256']:
            raise RuntimeError(f"checksum mismatch for {f['url']}: {digest}")
        with open(dst, 'wb') as fh:
            fh.write(data)
        print(f"{f['dst']}  <-  {f['url']}  (license: {f['license']})")


if __name__ == '__main__':
    main()
