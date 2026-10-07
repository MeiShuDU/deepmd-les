#!/root/venv-compshare/bin/python
# coding: utf-8
"""Provision the compshare GPU pod through the platform API and log in to it.

This drives the same `ucompshare` service the web console uses, via
ucloud-sdk-python3. The SDK lives in its own venv (/root/venv-compshare, see
the shebang) so it cannot disturb the deepmd/cace environment.

Credentials - secrets, never committed
--------------------------------------
A public/private API key pair from https://console.compshare.cn/uaccount/api_manage
Resolution order:

    1. --creds PATH
    2. $COMPSHARE_CREDS
    3. ~/.compshare/credentials.json
    4. $COMPSHARE_PUBLIC_KEY / $COMPSHARE_PRIVATE_KEY

The file is JSON and normally only needs the key pair:

    {"public_key": "...", "private_key": "..."}

The create call takes no SSH key (only `LoginMode` + `Password`), so a password
is needed once to install our public key - but the API returns it
base64-encoded, so a pod created here needs nothing stored. `instance_password`
is an optional fallback for a pod whose password the API will not hand back.

Instance spec - no secrets, safe to commit
------------------------------------------
--spec PATH, default <this directory>/compshare_spec.json

Commands
--------
    list                       instances in the configured region
    show <ref>                 one instance, full detail (id, state, ssh command)
    images [--filter TEXT]     platform/app images, to pick a CompShareImageId
    types                      instance types sellable in the zone
    gpu [--zone Z]             remaining GPU cards per zone, by billing mode
    price                      price of the configured spec
    create --yes               create the spec'd instance
    ensure [--yes] [--retry M] create if absent, start if stopped, wait until
                               reachable, install our key, rewrite ssh config
    ssh-config [<ref>]         rewrite the "Host <alias>" block from live state
    wait [<ref>]               poll until the instance is running and ssh is up
    start | stop [--retry M]   power on / off
    terminate --yes <ref>      destroy (irreversible)

`--retry MINUTES` keeps re-issuing a start that fails because the GPU type is
out of capacity (check `gpu` to see whether waiting is likely to help).

<ref> is an instance id or a name. With no ref, the spec's UHostId is used
(falling back to its Name), so the common case is just `ensure`, `ssh-config`
or `stop`.
"""
import argparse
import base64
import json
import logging
import os
import re
import socket
import stat
import subprocess
import sys
import tempfile
import time

from ucloud.client import Client
from ucloud.core.exc import RetCodeException

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPEC = os.path.join(HERE, "compshare_spec.json")
DEFAULT_CREDS = os.path.expanduser("~/.compshare/credentials.json")
SSH_CONFIG = os.path.expanduser("~/.ssh/config")
PUBKEY = os.path.expanduser("~/.ssh/id_rsa.pub")

# The WSL2 side of this link has a path-MTU blackhole (see env-wsl2-mtu-blackhole
# in memory): the default PQ KEX sends a ~1216-byte key share that is silently
# dropped, hanging the handshake at "expecting SSH2_MSG_KEX_ECDH_REPLY".
# curve25519-sha256 sends 32 bytes, so force it on every ssh we spawn.
SSH_OPTS = ["-o", "KexAlgorithms=curve25519-sha256",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10"]

# CompShareInstanceSet.State, per the API docs.
RUNNING = {"running"}
PENDING = {"initializing", "starting", "stopping", "rebooting", "resizing"}
BROKEN = {"install fail", "terminated", "terminating", "deleted"}

# Seconds between attempts when waiting for a GPU to free up.
RETRY_WAIT = 60

# Optional keys of CreateCompShareInstanceRequestSchema we forward verbatim when
# the spec sets them. Required keys (Region, Zone, CPU, Memory, GPU, GpuType,
# MachineType, CompShareImageId) are assembled explicitly below.
OPTIONAL_CREATE_KEYS = (
    "ChargeType", "Disks", "EnableUS3", "LoginMode", "MinimalCpuPlatform",
    "ProjectId", "Quantity", "SecurityGroupId",
)


def die(msg, code=2):
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


# --- config -----------------------------------------------------------------

def load_creds(path):
    path = path or os.environ.get("COMPSHARE_CREDS") or DEFAULT_CREDS
    creds = {}
    if os.path.exists(path):
        with open(path) as f:
            creds = json.load(f)
    for env, key in (("COMPSHARE_PUBLIC_KEY", "public_key"),
                     ("COMPSHARE_PRIVATE_KEY", "private_key")):
        if os.environ.get(env):
            creds[key] = os.environ[env]
    if not (creds.get("public_key") and creds.get("private_key")):
        die(
            f"no API key pair found (looked in {path} and the environment).\n"
            "Create one at https://console.compshare.cn/uaccount/api_manage and write:\n"
            f'  mkdir -p {os.path.dirname(DEFAULT_CREDS)} && '
            f"chmod 700 {os.path.dirname(DEFAULT_CREDS)}\n"
            f'  cat > {DEFAULT_CREDS} <<\'EOF\'\n'
            '  {"public_key": "<public key>", "private_key": "<private key>"}\n'
            "  EOF\n"
            f"  chmod 600 {DEFAULT_CREDS}"
        )
    creds["_path"] = path
    return creds


def save_creds(creds):
    path = creds["_path"]
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    body = {k: v for k, v in creds.items() if not k.startswith("_")}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(body, f, indent=2, sort_keys=True)
        f.write("\n")


def load_spec(path):
    path = path or DEFAULT_SPEC
    if not os.path.exists(path):
        die(f"no spec at {path}; copy the template and fill in the image id")
    with open(path) as f:
        spec = json.load(f)
    for req in ("Region", "Zone", "CompShareImageId", "GPU", "GpuType",
                "MachineType", "CPU", "Memory"):
        if not spec.get(req):
            die(f"spec {path} is missing the required field {req!r}")
    spec.setdefault("Name", "les-pod")
    spec.setdefault("ssh_alias", "hpc")
    spec.setdefault("ssh_user", "root")
    spec["_path"] = path
    return spec


def make_client(creds, spec):
    # The SDK logs every request and full response at INFO, which buries our
    # own output (and echoes the base64 password back).
    logging.getLogger("ucloud").setLevel(logging.WARNING)
    return Client({
        "region": spec["Region"],
        "public_key": creds["public_key"],
        "private_key": creds["private_key"],
        "base_url": "https://api.compshare.cn",
        "project_id": spec.get("ProjectId"),
    })


# --- instance lookup --------------------------------------------------------

def describe_all(cli, spec):
    resp = cli.ucompshare().describe_comp_share_instance({"Region": spec["Region"]})
    return resp.get("UHostSet", [])


def resolve(cli, spec, ref):
    """Accept an instance id or a name; a None ref means the pinned pod.

    With no ref, prefer the spec's UHostId (pins one concrete pod, immune to a
    rename) and fall back to its Name (which is what `create` would use).
    """
    ref = ref or spec.get("UHostId") or spec.get("Name")
    insts = describe_all(cli, spec)
    for i in insts:
        if i.get("UHostId") == ref:
            return i
    named = [i for i in insts if i.get("Name") == ref]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        die(f"{len(named)} instances are named {ref!r}; use an id instead")
    return None


def is_running(inst):
    return (inst.get("State") or "").lower() in RUNNING


def ssh_endpoint(inst, spec=None):
    """(host, port) for the pod.

    The API returns SshLoginCommand only once the instance is up, and that is
    the authoritative source (the pod is behind NAT on a per-instance port).
    Before that, prefer the Bgp (public) IPSet entry over the Private one and
    take the port from the spec, which is what the console shows.
    """
    cmd = (inst.get("SshLoginCommand") or "").strip()
    host = port = None
    if cmd:
        m = re.search(r"@([A-Za-z0-9._-]+)", cmd)
        host = m.group(1) if m else None
        m = re.search(r"(?:^|\s)-p\s*(\d+)", cmd)
        port = int(m.group(1)) if m else None
    if not host:
        for want in ("Bgp", "Internation", "Private"):
            for e in (inst.get("IPSet") or []):
                if e.get("Type") == want and e.get("IP"):
                    host = e["IP"]
                    break
            if host:
                break
    if not host:
        return None, None
    return host, port or (spec or {}).get("ssh_port")


def zone_of(inst, spec):
    return inst.get("Zone") or spec["Zone"]


def pod_password(inst, creds):
    """The pod's root password, or None.

    The API hands it back base64-encoded, so a freshly created pod can be
    logged into without storing a secret anywhere.
    """
    pw = inst.get("Password")
    if pw:
        try:
            return base64.b64decode(pw).decode()
        except (ValueError, UnicodeDecodeError):
            return pw
    return creds.get("instance_password")


# --- ssh --------------------------------------------------------------------

def ssh_works(host, port, user):
    r = subprocess.run(
        ["ssh", *SSH_OPTS, "-o", "BatchMode=yes", "-p", str(port),
         f"{user}@{host}", "true"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=25,
    )
    return r.returncode == 0


def install_key(host, port, user, password):
    """Append our public key to the pod's authorized_keys, using the password once.

    sshpass is not assumed to be installed, so the password reaches ssh through
    SSH_ASKPASS; setsid detaches ssh from the tty, which is what makes ssh
    actually consult the helper.
    """
    if not os.path.exists(PUBKEY):
        die(f"{PUBKEY} does not exist; run ssh-keygen first")
    with open(PUBKEY) as f:
        key = f.read()

    helper = tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh")
    try:
        helper.write('#!/bin/sh\nprintf \'%s\\n\' "$COMPSHARE_SSH_PASSWORD"\n')
        helper.close()
        os.chmod(helper.name, stat.S_IRWXU)
        env = dict(os.environ,
                   SSH_ASKPASS=helper.name,
                   SSH_ASKPASS_REQUIRE="force",
                   COMPSHARE_SSH_PASSWORD=password,
                   DISPLAY=":0")
        # The key arrives on stdin, so it can only be consumed once: append it,
        # then dedupe, rather than grepping for it first.
        remote = ("mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
                  "touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && "
                  "cat >> ~/.ssh/authorized_keys && "
                  "sort -u -o ~/.ssh/authorized_keys ~/.ssh/authorized_keys")
        r = subprocess.run(
            ["setsid", "ssh", *SSH_OPTS, "-p", str(port), f"{user}@{host}", remote],
            input=key, text=True, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        )
    finally:
        os.unlink(helper.name)
    if r.returncode != 0:
        die(f"could not install the public key:\n{r.stderr.strip()}")


def write_ssh_config(alias, user, host, port):
    block = [
        f"Host {alias}",
        f"    User {user}",
        f"    Hostname {host}",
        f"    Port {port}",
        "    IdentityFile ~/.ssh/id_rsa",
        "    # WSL2 path-MTU blackhole. The default PQ KEX (sntrup761x25519-sha512) puts a",
        "    # ~1216-byte key share in SSH2_MSG_KEX_ECDH_INIT; that packet is silently dropped",
        "    # and the client hangs forever at \"expecting SSH2_MSG_KEX_ECDH_REPLY\". Smaller",
        "    # packets (banner, KEXINIT) pass fine. curve25519-sha256 has a 32-byte share.",
        "    # PowerShell works only because Windows OpenSSH does not offer sntrup761 at all.",
        "    KexAlgorithms curve25519-sha256",
        "    ServerAliveInterval 30",
        "    ServerAliveCountMax 4",
        "    StrictHostKeyChecking accept-new",
    ]
    old = ""
    if os.path.exists(SSH_CONFIG):
        with open(SSH_CONFIG) as f:
            old = f.read()
    lines, out, i = old.splitlines(), [], 0
    replaced = False
    while i < len(lines):
        if re.match(rf"^\s*Host\s+{re.escape(alias)}\s*$", lines[i]):
            out.extend(block)
            replaced = True
            i += 1
            # Consume the old block, but carry its trailing blank lines over so
            # the separator before the next Host survives and repeated runs do
            # not accumulate or eat blank lines.
            tail_blank = 0
            while i < len(lines) and not re.match(r"^\s*Host\s+", lines[i]):
                tail_blank = tail_blank + 1 if not lines[i].strip() else 0
                i += 1
            out.extend([""] * tail_blank)
            continue
        out.append(lines[i])
        i += 1
    if not replaced:
        if out and out[-1].strip():
            out.append("")
        out.extend(block)
    os.makedirs(os.path.dirname(SSH_CONFIG), mode=0o700, exist_ok=True)
    with open(SSH_CONFIG, "w") as f:
        f.write("\n".join(out).rstrip() + "\n")
    os.chmod(SSH_CONFIG, 0o600)
    return replaced


# --- commands ---------------------------------------------------------------

def cmd_list(cli, spec, args):
    insts = describe_all(cli, spec)
    if not insts:
        print(f"no instances in {spec['Region']}")
        return
    print(f"{'name':<22} {'id':<22} {'state':<12} {'gpu':<10} {'endpoint':<45}")
    for i in insts:
        host, port = ssh_endpoint(i, spec)
        gpu = f"{i.get('GPU') or 0}x{i.get('GpuType') or '?'}"
        ep = f"{host}:{port}" if host and port else "-"
        print(f"{str(i.get('Name')):<22} {str(i.get('UHostId')):<22} "
              f"{str(i.get('State')):<12} {gpu:<10} {ep:<45}")
    print(f"\n{len(insts)} instance(s)")


def cmd_show(cli, spec, args):
    inst = resolve(cli, spec, args.ref)
    if not inst:
        die(f"no instance named {args.ref or spec['Name']!r}")
    print(json.dumps(inst, indent=2, sort_keys=True, ensure_ascii=False))


def cmd_images(cli, spec, args):
    req = {"Region": spec["Region"]}
    if args.filter:
        req["Name"] = args.filter
    resp = cli.ucompshare().describe_comp_share_images(req)
    for img in resp.get("ImageSet", []):
        print(f"{img.get('CompShareImageId'):<32} {img.get('Name')}")
        if img.get("VersionName") or img.get("Description"):
            print(f"    {img.get('VersionName') or ''} {img.get('Description') or ''}".rstrip())


def cmd_types(cli, spec, args):
    resp = cli.ucompshare().describe_available_comp_share_instance_types(
        {"Region": spec["Region"], "Zone": spec["Zone"]})
    for t in resp.get("AvailableInstanceTypes", []):
        print(json.dumps(t, sort_keys=True, ensure_ascii=False))


def cmd_price(cli, spec, args):
    req = {"Region": spec["Region"], "Zone": spec["Zone"],
           "Cpu": str(spec["CPU"]), "Memory": str(spec["Memory"]),
           "Gpu": str(spec["GPU"]), "GpuType": spec["GpuType"],
           "CompShareImageId": spec["CompShareImageId"]}
    if spec.get("ChargeType"):
        req["ChargeType"] = spec["ChargeType"]
    if spec.get("Disks"):
        req["Disks"] = spec["Disks"]
    print(json.dumps(cli.ucompshare().get_comp_share_instance_price(req),
                     indent=2, sort_keys=True, ensure_ascii=False))


def build_create_request(spec):
    req = {
        "Region": spec["Region"], "Zone": spec["Zone"],
        "CompShareImageId": spec["CompShareImageId"],
        "GPU": spec["GPU"], "GpuType": spec["GpuType"],
        "MachineType": spec["MachineType"],
        "CPU": spec["CPU"], "Memory": spec["Memory"],
        "Name": spec["Name"],
    }
    for k in OPTIONAL_CREATE_KEYS:
        if spec.get(k) is not None:
            req[k] = spec[k]
    req.setdefault("LoginMode", "Password")
    return req


def do_create(cli, creds, spec):
    req = build_create_request(spec)
    if not req.get("Password"):
        pw = creds.get("instance_password")
        if not pw:
            pw = subprocess.run(["openssl", "rand", "-hex", "12"],
                                capture_output=True, text=True).stdout.strip()
            creds["instance_password"] = pw
            save_creds(creds)
            print(f"generated a pod password, stored in {creds['_path']} (mode 600)")
        req["Password"] = pw
    resp = cli.ucompshare().create_comp_share_instance(req)
    ids = resp.get("UHostIds") or []
    if not ids:
        die(f"create returned no UHostIds: {resp}")
    print(f"created {ids[0]} ({spec['Name']}) in {spec['Zone']}")
    return ids[0]


def call_retrying_capacity(call, req, retry_min):
    """Call an API, retrying while the platform says a card is unavailable.

    A start can fail with e.g. 226604 "This GPU type is currently out of
    resources. Please refresh or try again later." The API's own convention
    marks every code above 2000 as retryable, so that is the gate, and
    `gpu` shows whether waiting is likely to help.
    """
    deadline = time.time() + retry_min * 60
    while True:
        try:
            return call(req)
        except RetCodeException as e:
            if not (retry_min and e.retryable) or time.time() + RETRY_WAIT > deadline:
                raise
            print(f"  {e.message.strip()}; retrying in {RETRY_WAIT}s")
            time.sleep(RETRY_WAIT)


def do_start(cli, spec, inst, retry_min=0):
    call_retrying_capacity(
        cli.ucompshare().start_comp_share_instance,
        {"Region": spec["Region"], "Zone": zone_of(inst, spec),
         "UHostId": inst["UHostId"]},
        retry_min)
    print(f"start requested for {inst['UHostId']}")


def wait_running(cli, spec, inst, timeout=600):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        inst = resolve(cli, spec, inst["UHostId"]) or inst
        state = inst.get("State")
        if state != last:
            print(f"  state: {state}")
            last = state
        if is_running(inst):
            host, port = ssh_endpoint(inst, spec)
            if host and port:
                return inst
        if (state or "").lower() in BROKEN:
            die(f"instance entered {state}")
        time.sleep(5)
    die(f"instance did not become reachable within {timeout}s")


def wait_ssh(host, port, user, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(5)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(5)
    return False


def cmd_wait(cli, spec, args):
    inst = resolve(cli, spec, args.ref)
    if not inst:
        die(f"no instance named {args.ref or spec['Name']!r}")
    inst = wait_running(cli, spec, inst)
    host, port = ssh_endpoint(inst, spec)
    print(f"running at {host}:{port}; waiting for sshd...")
    print("sshd up" if wait_ssh(host, port, spec["ssh_user"]) else "sshd not up yet")


def cmd_ssh_config(cli, spec, args):
    inst = resolve(cli, spec, args.ref)
    if not inst:
        die(f"no instance named {args.ref or spec['Name']!r}")
    host, port = ssh_endpoint(inst, spec)
    if not (host and port):
        die("the API reported no host/port for this instance yet; the pod is "
            "behind NAT, so wait until it is running (`wait`) or set ssh_port "
            "in the spec")
    replaced = write_ssh_config(spec["ssh_alias"], spec["ssh_user"], host, port)
    print(f"{'updated' if replaced else 'added'} 'Host {spec['ssh_alias']}' -> "
          f"{spec['ssh_user']}@{host}:{port} in {SSH_CONFIG}")
    print(f"try: ssh {spec['ssh_alias']}")


def raw_call(svc, api_name, req):
    """Call an API and return its full response dict.

    The generated response schemas are sometimes older than the live API and
    silently drop fields they do not declare - DescribeCompShareGpuInventory
    really returns GpuInventory where the schema says GpuInventoryByZone - so
    the calls that need those fields read the raw payload instead.
    """
    from ucloud.services.ucompshare.schemas import apis
    schema = getattr(apis, api_name + "RequestSchema")
    d = {"ProjectId": svc.config.project_id, "Region": svc.config.region}
    d.update(req)
    return svc.invoke(api_name, schema().dumps(d))


def zone_names(cli, region):
    """{zone id string: zone name}, to label the inventory."""
    resp = cli.ucompshare().describe_comp_share_support_zone({"Region": region})
    return {str(z.get("ZoneId")): z.get("Zone")
            for z in resp.get("ZoneInfo", [])}


def cmd_gpu(cli, spec, args):
    """Remaining GPU cards per zone, by billing mode.

    This is the call to make when a start fails with "out of resources": it
    says whether the card is gone fleet-wide or only in this zone.
    """
    region = args.region or spec["Region"]
    req = {"Region": region}
    if args.zone:
        req["Zone"] = args.zone
    resp = raw_call(cli.ucompshare(), "DescribeCompShareGpuInventory", req)
    inv = resp.get("GpuInventory") or resp.get("GpuInventoryByZone")
    if isinstance(inv, str):
        try:
            inv = json.loads(inv)
        except ValueError:
            pass
    if not isinstance(inv, dict):
        print(inv)
    else:
        names = zone_names(cli, region)
        for mode in sorted(inv):
            print(f"{mode}:")
            zones = inv[mode] or {}
            for zid in sorted(zones, key=lambda z: (len(z), z)):
                free = {g: n for g, n in (zones[zid] or {}).items() if n}
                label = f"{zid} ({names.get(zid, '?')})"
                print(f"  {label:32} {free if free else 'nothing free'}")
    if resp.get("UpdateTime"):
        print("\nupdated: " + time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime(resp["UpdateTime"])))
    if resp.get("SpotUnsupportedGpuTypes"):
        print("no spot instances for: " + ", ".join(resp["SpotUnsupportedGpuTypes"]))


def cmd_create(cli, spec, args, creds):
    if not args.yes:
        die(f"create provisions a billable instance ({spec['GPU']}x{spec['GpuType']}); "
            "re-run with --yes")
    if resolve(cli, spec, spec["Name"]):
        die(f"an instance named {spec['Name']!r} already exists; use `ensure`")
    do_create(cli, creds, spec)


def cmd_ensure(cli, spec, args, creds):
    inst = resolve(cli, spec, spec["Name"])
    if inst is None:
        if not args.yes:
            die(f"no instance named {spec['Name']!r}; create one with "
                "`ensure --yes` (billable) or `create --yes`")
        inst = wait_running(cli, spec, {"UHostId": do_create(cli, creds, spec)})
        inst = resolve(cli, spec, spec["Name"]) or inst
    elif not is_running(inst):
        if (inst.get("State") or "").lower() in PENDING:
            inst = wait_running(cli, spec, inst)
        else:
            do_start(cli, spec, inst, getattr(args, "retry", 0))
            inst = wait_running(cli, spec, inst)

    host, port = ssh_endpoint(inst, spec)
    if not (host and port):
        die("instance is running but the API reported no host/port for it")
    user = spec["ssh_user"]
    if not ssh_works(host, port, user):
        wait_ssh(host, port, user)
        pw = pod_password(inst, creds)
        if not pw:
            die(f"no key access to {host} and the API returned no password to "
                f"install one; put the pod's password in {creds['_path']}")
        print(f"installing {PUBKEY} on the pod")
        install_key(host, port, user, pw)
        if not ssh_works(host, port, user):
            die("key install appeared to succeed but key auth still fails")
    write_ssh_config(spec["ssh_alias"], user, host, port)
    print(f"ready: ssh {spec['ssh_alias']}  ({user}@{host}:{port})")


def cmd_power(cli, spec, args, action, retry_min=0):
    inst = resolve(cli, spec, args.ref)
    if not inst:
        die(f"no instance named {args.ref or spec['Name']!r}")
    state = inst.get("State")
    call_retrying_capacity(
        getattr(cli.ucompshare(), action),
        {"Region": spec["Region"], "Zone": zone_of(inst, spec),
         "UHostId": inst["UHostId"]},
        retry_min)
    print(f"{action} requested for {inst['UHostId']} (was {state})")


# --- entry point ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--creds", help=f"credentials JSON (default {DEFAULT_CREDS})")
    ap.add_argument("--spec", help=f"instance spec JSON (default {DEFAULT_SPEC})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list instances in the region")
    sp = sub.add_parser("show", help="dump one instance as JSON")
    sp.add_argument("ref", nargs="?")
    sp = sub.add_parser("images", help="list available images")
    sp.add_argument("--filter", help="substring match on the image name")
    sub.add_parser("types", help="instance types sellable in the zone")
    sp = sub.add_parser("gpu", help="remaining GPU cards per zone")
    sp.add_argument("--region")
    sp.add_argument("--zone")
    sub.add_parser("price", help="price of the configured spec")
    sp = sub.add_parser("create", help="create the spec'd instance (billable)")
    sp.add_argument("--yes", action="store_true")
    sp = sub.add_parser("ensure", help="create/start/wait/key/ssh-config")
    sp.add_argument("--yes", action="store_true")
    sp.add_argument("--retry", type=int, default=0, metavar="MINUTES",
                    help="keep retrying a start that fails for lack of GPU "
                         "capacity, for up to MINUTES")
    sp = sub.add_parser("ssh-config", help="refresh ~/.ssh/config from live state")
    sp.add_argument("ref", nargs="?")
    sp = sub.add_parser("wait", help="poll until running and reachable")
    sp.add_argument("ref", nargs="?")
    for name in ("start", "stop"):
        sp = sub.add_parser(name, help=f"{name} an instance")
        sp.add_argument("ref", nargs="?")
        sp.add_argument("--retry", type=int, default=0, metavar="MINUTES",
                        help="keep retrying a start that fails for lack of GPU "
                             "capacity, for up to MINUTES")
    sp = sub.add_parser("terminate", help="destroy an instance (irreversible)")
    sp.add_argument("ref", nargs="?")
    sp.add_argument("--yes", action="store_true")

    args = ap.parse_args()
    try:
        return run(args)
    except RetCodeException as e:
        # e.g. 226604: This GPU type is currently out of resources.
        die(str(e), 1)


def run(args):
    creds = load_creds(args.creds)
    spec = load_spec(args.spec)
    cli = make_client(creds, spec)

    if args.cmd == "list":
        return cmd_list(cli, spec, args)
    if args.cmd == "show":
        return cmd_show(cli, spec, args)
    if args.cmd == "images":
        return cmd_images(cli, spec, args)
    if args.cmd == "types":
        return cmd_types(cli, spec, args)
    if args.cmd == "gpu":
        return cmd_gpu(cli, spec, args)
    if args.cmd == "price":
        return cmd_price(cli, spec, args)
    if args.cmd == "create":
        return cmd_create(cli, spec, args, creds)
    if args.cmd == "ensure":
        return cmd_ensure(cli, spec, args, creds)
    if args.cmd == "ssh-config":
        return cmd_ssh_config(cli, spec, args)
    if args.cmd == "wait":
        return cmd_wait(cli, spec, args)
    if args.cmd == "start":
        return cmd_power(cli, spec, args, "start_comp_share_instance", args.retry)
    if args.cmd == "stop":
        return cmd_power(cli, spec, args, "stop_comp_share_instance")
    if args.cmd == "terminate":
        inst = resolve(cli, spec, args.ref)
        if not inst:
            die(f"no instance named {args.ref or spec['Name']!r}")
        if not args.yes:
            die(f"terminate destroys {inst['UHostId']} ({inst.get('Name')}) "
                "irreversibly; re-run with --yes")
        return cmd_power(cli, spec, args, "terminate_comp_share_instance")


if __name__ == "__main__":
    main()
