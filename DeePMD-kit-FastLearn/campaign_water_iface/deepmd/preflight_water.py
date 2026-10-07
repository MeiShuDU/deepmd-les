"""Check that every generated water-interface config is one deepmd will accept.

Everything downstream assumes two things and neither is free: the schema has to
normalise (a ``les_params`` key the les version on THIS host does not read is
silently ignored rather than rejected, so this also reports the keys that
survived), and the model has to construct (the descriptor, the fitting net and
the LES charge head are all built from the file). Running it on both hosts,
before spending GPU time, is how a pod/local divergence in the les tree gets
caught while it is still cheap.

Prints one line per config with the parameter count, which is also the cheapest
check that the arms differ only where they should:

* every block of an arm must report the same count (the schedule changes the LR
  and the loss weights, never the architecture);
* the LES arm must exceed the SR arm by exactly the LES module's own parameter
  count - not merely by "some positive number". If ``n_hidden``/``n_layers`` in
  ``les_params`` were being ignored, the charge head would silently be les's
  [32, 16] default and this is where it shows.

Usage:
    python preflight_water.py [--root runs] [--pattern 'runs/*_s[AB]/s*/input.yaml']
"""
import argparse
import glob
import os


def les_params_of(model):
    """The les submodule, wherever the hybrid model keeps it."""
    for attr in ('les_model',):
        am = getattr(model, 'atomic_model', None)
        if am is not None and hasattr(am, attr):
            return getattr(am, attr)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='.')
    ap.add_argument('--pattern', default=None,
                    help="glob, relative to --root; default every chained config")
    args = ap.parse_args()

    from deepmd.common import j_loader
    from deepmd.pt.model.model import get_model
    from deepmd.utils.argcheck import normalize

    pat = args.pattern or 'runs/*_s[AB]/s*/input.yaml'
    paths = sorted(glob.glob(os.path.join(args.root, pat)))
    if not paths:
        raise SystemExit(f'no configs match {pat!r} under {args.root}')

    by_arm = {}
    les_module_params = {}
    bad = 0
    for p in paths:
        arm = os.path.basename(os.path.dirname(os.path.dirname(p)))
        try:
            cfg = normalize(j_loader(p))
            model = get_model(cfg['model'])
            nparam = sum(x.numel() for x in model.parameters())
            les = sorted(cfg['model'].get('les_params', {}))
            by_arm.setdefault(arm, []).append(nparam)
            lesmod = les_params_of(model)
            if lesmod is not None:
                les_module_params[arm] = sum(x.numel() for x in lesmod.parameters())
            print(f'OK   {p:56s} {cfg["model"]["type"]:12s} params {nparam:>8d} '
                  f'les_params {les}')
        except Exception as e:  # noqa: BLE001 - the point is to report any failure
            bad += 1
            print(f'FAIL {p}: {type(e).__name__}: {str(e)[:300]}')
    print()
    for arm, counts in sorted(by_arm.items()):
        same = len(set(counts)) == 1
        print(f'{arm:12s} {len(counts)} configs, params '
              f'{sorted(set(counts))}{"" if same else "  *** NOT CONSTANT"}')
        bad += 0 if same else 1

    # the LES arm must exceed the SR arm by the LES module's own count
    sr_arm = next((a for a in by_arm if 'les' not in a), None)
    lr_arm = next((a for a in by_arm if 'les' in a), None)
    if sr_arm and lr_arm:
        d = sorted(set(by_arm[lr_arm]))[0] - sorted(set(by_arm[sr_arm]))[0]
        lesmod = les_module_params.get(lr_arm)
        print(f'\nLES arm exceeds SR arm by {d} params; '
              f'the les submodule itself has {lesmod} params')
        if lesmod is None:
            print('  *** could not locate the les submodule to cross-check')
            bad += 1
        elif d != lesmod:
            print('  *** NOT the les module alone: the SR part differs too')
            bad += 1
        elif d <= 0:
            print('  *** the les arm is not larger; les_params did not take effect')
            bad += 1
        else:
            print('  OK: the two arms differ by the long-range module and nothing else')
    print(f'\n{len(paths)} configs, {bad} problem(s)')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())