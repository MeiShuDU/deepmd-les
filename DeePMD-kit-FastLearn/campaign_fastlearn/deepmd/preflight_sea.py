"""Check that every generated config is one deepmd will actually accept and build.

Everything downstream assumes these two things and neither is free: the schema has to
normalise (a `les_params` key the les version on THIS host does not read is silently
ignored rather than rejected, so this also reports the keys that survived), and the
model has to construct (the descriptor, the fitting net and the LES charge head are
all built from the file). Running it on both hosts, before spending GPU time, is how
a pod/local divergence in the les tree gets caught while it is still cheap.

Prints one line per config with the parameter count, which is also the cheapest check
that the two arms differ only where they should: every block of an arm must report
the same count, and the LES arm must exceed the SR arm by exactly the charge head.

Usage:
    python preflight_sea.py [--root runs] [--pattern 'runs/*/s*/input.yaml']
"""
import argparse
import glob
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='.')
    ap.add_argument('--pattern', default=None,
                    help="glob, relative to --root; default every chained config")
    args = ap.parse_args()

    from deepmd.common import j_loader
    from deepmd.pt.model.model import get_model
    from deepmd.utils.argcheck import normalize

    pat = args.pattern or 'runs/*sched*/s*/input.yaml'
    paths = sorted(glob.glob(os.path.join(args.root, pat)))
    if not paths:
        raise SystemExit(f'no configs match {pat!r} under {args.root}')

    by_arm = {}
    bad = 0
    for p in paths:
        arm = os.path.basename(os.path.dirname(os.path.dirname(p)))
        try:
            cfg = normalize(j_loader(p))
            model = get_model(cfg['model'])
            nparam = sum(x.numel() for x in model.parameters())
            les = sorted(cfg['model'].get('les_params', {}))
            by_arm.setdefault(arm, []).append(nparam)
            print(f'OK   {p:56s} {cfg["model"]["type"]:12s} params {nparam:>7d} '
                  f'les_params {les}')
        except Exception as e:  # noqa: BLE001 - the point is to report any failure
            bad += 1
            print(f'FAIL {p}: {type(e).__name__}: {str(e)[:300]}')
    print()
    for arm, counts in sorted(by_arm.items()):
        same = len(set(counts)) == 1
        print(f'{arm:28s} {len(counts)} configs, params '
              f'{sorted(set(counts))}{"" if same else "  *** NOT CONSTANT"}')
        bad += 0 if same else 1
    print(f'\n{len(paths)} configs, {bad} problem(s)')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
