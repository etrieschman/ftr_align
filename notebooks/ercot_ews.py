#!/usr/bin/env python3
"""
Retrieve ERCOT reports over External Web Services (EWS).

WHAT THIS TALKS TO
------------------
ERCOT exposes two entirely separate APIs:

  * The Public Data API (api.ercot.com/api/public-reports) — REST/JSON,
    authenticated with a subscription key plus an OAuth bearer token.
    It serves ONLY products classified securityClassification="Public".
  * EWS (misapi.ercot.com/NodalAPI/EWS/) — SOAP, authenticated with an
    ERCOT-issued API certificate. This is the channel for Secure and
    ECEII-classified products.

The CRR Network Model is Secure/ECEII, so it is not present in the public
API at any endpoint. EWS is the only programmatic route to it.

CREDENTIALS
-----------
EWS requires an *API* certificate, not the personal certificate used to log
into MIS in a browser. The server checks the employee ID against LDAP and
rejects anything not prefixed "API_":

    SECU1073 ... Employee Number: <id> does not start with prefix-API_

A USA creates one in MPIM via Create with the "API Certificate" box ticked,
which auto-generates the API_-prefixed ID, then picks it up through the
normal enrollment flow.

Two independent layers of authentication both apply to every call:

  1. Mutual TLS — the certificate is presented during the TLS handshake.
  2. WS-Security — the SOAP Body is signed with the same certificate and
     the signature travels in a wsse:Security header. Transport security
     alone yields "SECU1096: Could not find a WS-Security Header".

ERCOT's WS-Security stack predates SHA-2 and rejects modern algorithms:

    SECU3518: Invalid digest algorithm ... Expecting ... #sha1
    SECU3517: The signature algorithm ... Expecting ... #rsa-sha1

Hence the sha1 / rsa-sha1 defaults below. They are not a security choice on
our part; the server refuses anything stronger.

SETUP
-----
Convert the API .pfx into the PEM pair Python's TLS stack needs. The
-legacy flag is required because ERCOT wraps the bundle with RC2-40-CBC,
which OpenSSL 3 moved to the legacy provider:

    mkdir -p ~/.ercot
    openssl pkcs12 -legacy -in API_CERT.pfx -clcerts -nokeys -out ~/.ercot/api.crt
    openssl pkcs12 -legacy -in API_CERT.pfx -nocerts -nodes  -out ~/.ercot/api.key
    chmod 600 ~/.ercot/api.key

api.key is an UNENCRYPTED private key. Keep it outside the repository and
out of any synced folder.

Identity comes from the environment so it stays out of version control:

    export ERCOT_DUNS=...
    export ERCOT_API_USER=API_...

or put them in a gitignored .env and run with `uv run --env-file .env`.

Dependencies:

    uv add requests lxml "zeep[xmlsec]"

zeep is used only as a signing library — the SOAP envelope is built by hand
from ERCOT's Message.xsd. The shipped WSDL declares a placeholder address
(https://dummyhost:0000/) and uses xsd:any with processContents="skip" for
the payload, so a generated client buys little and obscures the wire format
when something fails.

USAGE
-----
    uv run ercot_ews.py                      # list last 365 days
    uv run ercot_ews.py --days 30            # narrower window
    uv run ercot_ews.py --all                # no time filter at all
    uv run ercot_ews.py --download           # fetch the files
    uv run ercot_ews.py --option 11204       # a different report type
    uv run ercot_ews.py --debug              # dump signed request + response

REPORT TYPE IDs
---------------
--option takes an EMIL "Report Type ID", not the EMIL product ID. Look it
up on the product's page at ercot.com/mp/data-products.

    11204 = np7-801-m, CRR Network Model. Returns report groups such as
            "CRR Network Model (Long-Term)". MIS shows a 365-day display
            window, but archiveDuration on comparable products is 2555 days
            (7 years), so try a wider --days before assuming a hard limit.
"""

import argparse
import base64
import os
import re
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from lxml import etree

# ---------------------------------------------------------------- config

ENDPOINT = "https://misapi.ercot.com/NodalAPI/EWS/"

# All three EWS operations (MarketInfo, MarketTransactions, Alerts) accept
# the same RequestMessage; the Noun inside the message selects the resource.
# Reports are served by MarketInfo. The SOAPAction is a path, not a URN —
# sending a URN yields "RUNTIME0031: Failed to locate the operation".
SOAP_ACTION = "/BusinessService/NodalService.serviceagent/HttpEndPoint/MarketInfo"

CERT = Path(os.environ.get("ERCOT_CERT", Path.home() / ".ercot" / "api.crt"))
KEY = Path(os.environ.get("ERCOT_KEY", Path.home() / ".ercot" / "api.key"))

DEFAULT_OPTION = "11204"  # np7-801-m, CRR Network Model
OUTDIR = Path(os.environ.get("ERCOT_OUTDIR", "ercot_data"))

NS = {
    "soap": "http://schemas.xmlsoap.org/soap/envelope/",
    "msg": "http://www.ercot.com/schema/2007-06/nodal/ews/message",
    # ERCOT's Message.xsd imports a "www.docs" variant of the WSS
    # namespaces for its ReplayDetection types. The Security header itself
    # uses the standard OASIS namespaces, which zeep supplies. Do not
    # unify these — the server validates both.
    "wsse_msg": "http://www.docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-wssecurity-secext-1.0.xsd",
    "wsu_msg": "http://www.docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-wssecurity-utility-1.0.xsd",
}


def load_identity():
    """
    Read DUNS and API user ID from the environment.

    Kept out of the source so this file is safe to commit. Neither value is
    a secret in the way a password is, but both identify the market
    participant and belong in local config rather than a public repo.
    """
    duns = os.environ.get("ERCOT_DUNS")
    user = os.environ.get("ERCOT_API_USER")
    if not duns or not user:
        raise SystemExit(
            "Set ERCOT_DUNS and ERCOT_API_USER.\n\n"
            "  export ERCOT_DUNS=1234567890000\n"
            "  export ERCOT_API_USER=API_YOURID\n\n"
            "Or put them in a gitignored .env and use:\n"
            "  uv run --env-file .env ercot_ews.py"
        )
    if not user.startswith("API_"):
        print(
            f"Warning: ERCOT_API_USER={user!r} does not start with 'API_'. "
            "EWS will reject a personal certificate.",
            file=sys.stderr,
        )
    return duns, user


# ------------------------------------------------------------ soap build


def build_request(duns, user, option, start=None, end=None):
    """
    Construct an unsigned RequestMessage envelope for GetReports.

    Element order is load-bearing. Message.xsd defines HeaderType and
    RequestType as xsd:sequence, so children must appear in schema order:

      Header:  Verb, Noun, ReplayDetection, Revision, Source, UserID, ...
      Request: MarketType, OperatingDate, TradingDate, StartTime, EndTime,
               Zone, ASType, Option, SortBy, ID

    Per ERCOT's GetReports spec, Option (the report type ID) is required
    and the time bounds are optional. Valid combinations:

      Option + StartTime + EndTime  — reports in the window
      Option + StartTime            — from StartTime to now
      Option + EndTime              — everything up to EndTime
      Option alone                  — everything available

    ReplayDetection is mandatory (not minOccurs="0"). The Nonce is fresh
    random bytes per call; reusing one invites replay rejection.

    Args:
        duns:   market participant DUNS, sent as Header/Source
        user:   API_-prefixed employee ID, sent as Header/UserID
        option: report type ID as a string
        start:  timezone-aware datetime, or None to omit StartTime
        end:    timezone-aware datetime, or None to omit EndTime

    Returns:
        The envelope as a str, ready to sign.
    """
    nonce = base64.b64encode(secrets.token_bytes(16)).decode()
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def iso(dt):
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    window = ""
    if start:
        window += f"        <msg:StartTime>{iso(start)}</msg:StartTime>\n"
    if end:
        window += f"        <msg:EndTime>{iso(end)}</msg:EndTime>\n"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="{NS["soap"]}">
  <soap:Header/>
  <soap:Body>
    <msg:RequestMessage xmlns:msg="{NS["msg"]}"
                        xmlns:wsse="{NS["wsse_msg"]}"
                        xmlns:wsu="{NS["wsu_msg"]}">
      <msg:Header>
        <msg:Verb>get</msg:Verb>
        <msg:Noun>Reports</msg:Noun>
        <msg:ReplayDetection>
          <msg:Nonce>{nonce}</msg:Nonce>
          <msg:Created>{created}</msg:Created>
        </msg:ReplayDetection>
        <msg:Revision>1</msg:Revision>
        <msg:Source>{duns}</msg:Source>
        <msg:UserID>{user}</msg:UserID>
      </msg:Header>
      <msg:Request>
{window}        <msg:Option>{option}</msg:Option>
      </msg:Request>
    </msg:RequestMessage>
  </soap:Body>
</soap:Envelope>
"""


# --------------------------------------------------------- ws-security


def sign(envelope_xml, style="binary", digest="sha1", sig_method="rsa-sha1"):
    """
    Attach a WS-Security header containing an XML signature over the Body.

    zeep's signature classes handle exclusive canonicalisation and the
    xmldsig construction; doing this by hand is a reliable way to produce
    signatures that verify locally and fail server-side.

    Args:
        envelope_xml: unsigned envelope from build_request()
        style: "binary" embeds the certificate as a BinarySecurityToken and
            references it by URI — the X.509 Token Profile ERCOT's spec
            cites, and what works in practice. "plain" inlines the
            certificate as X509Data instead; kept as a fallback in case a
            future endpoint prefers it.
        digest: "sha1" or "sha256". ERCOT requires sha1.
        sig_method: "rsa-sha1" or "rsa-sha256". ERCOT requires rsa-sha1.

    Returns:
        The signed envelope as bytes.

    Note that the server validates the digest algorithm before the
    signature algorithm, so a wrong digest masks a wrong signature method.
    Change one at a time when debugging.
    """
    try:
        import xmlsec
        from zeep.wsse.signature import BinarySignature, Signature
    except ImportError:
        raise SystemExit(
            "WS-Security signing needs xmlsec.\n\n"
            '    uv add "zeep[xmlsec]"\n\n'
            "If the wheel fails to build, it needs the C library:\n"
            "    brew install libxmlsec1 pkg-config"
        )

    digests = {"sha1": xmlsec.Transform.SHA1, "sha256": xmlsec.Transform.SHA256}
    sigs = {
        "rsa-sha1": xmlsec.Transform.RSA_SHA1,
        "rsa-sha256": xmlsec.Transform.RSA_SHA256,
    }

    cls = BinarySignature if style == "binary" else Signature
    signer = cls(
        str(KEY),
        str(CERT),
        password=None,
        signature_method=sigs[sig_method],
        digest_method=digests[digest],
    )

    envelope = etree.fromstring(envelope_xml.encode("utf-8"))
    envelope, _ = signer.apply(envelope, {})
    return etree.tostring(envelope, xml_declaration=True, encoding="UTF-8")


def check_certs():
    """Fail early with a usable message rather than a TLS traceback."""
    missing = [str(p) for p in (CERT, KEY) if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing certificate files:\n  "
            + "\n  ".join(missing)
            + "\n\nSee the module docstring for the openssl conversion."
        )


def call_ews(signed_bytes, debug=False):
    """
    POST a signed envelope and return the raw requests.Response.

    The certificate is supplied twice by design: once by requests for the
    TLS handshake, and once inside the message by sign(). Both are checked.
    """
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": SOAP_ACTION,
    }
    if debug:
        print("--- signed request ---")
        print(signed_bytes.decode("utf-8", "replace")[:6000])
        print("--- end request ---\n")

    r = requests.post(
        ENDPOINT,
        data=signed_bytes,
        headers=headers,
        cert=(str(CERT), str(KEY)),
        timeout=120,
    )
    print(f"HTTP {r.status_code}")
    if debug:
        print("--- response ---")
        print(r.text[:6000])
        print("--- end response ---\n")
    return r


# ---------------------------------------------------------- response parse


def parse_reports(xml_bytes):
    """
    Extract Report entries from a ResponseMessage.

    A successful reply carries Reply/ReplyCode == "OK" and a Payload with
    zero or more ns1:Report elements, each describing one posted file:

        operatingDate, reportGroup, fileName, created, size, format, URL

    The payload sits in .../nodal/ews while the message envelope is in
    .../nodal/ews/message, and ERCOT has changed namespaces across
    revisions before. Matching on local name rather than a fixed namespace
    means a future change surfaces as a parse difference rather than a
    silent zero-result.

    Raises SystemExit on a SOAP Fault or a non-OK ReplyCode, since neither
    is recoverable here and both carry a diagnostic worth reading.
    """
    root = etree.fromstring(xml_bytes)

    fault = root.find(".//soap:Fault", NS)
    if fault is not None:
        msg = fault.findtext("faultstring") or "?"
        detail = fault.findtext("detail") or ""
        raise SystemExit(f"SOAP Fault: {msg}\n{detail}")

    code = root.findtext(".//msg:ReplyCode", namespaces=NS)
    if code and code.upper() != "OK":
        errs = [e.text for e in root.findall(".//msg:Error", NS)]
        raise SystemExit(f"ReplyCode={code}. Errors: {errs}")

    reports = []
    for node in root.iter():
        if etree.QName(node).localname != "Report":
            continue
        entry = {etree.QName(c).localname: (c.text or "").strip() for c in node}
        if entry:
            reports.append(entry)
    return reports


def safe_name(name):
    """Strip anything path-like out of a server-supplied filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "report.bin"


def download(reports):
    """
    Fetch each report's URL into OUTDIR, skipping files already present.

    The URLs point at misdownload/servlets/mirDownload?doclookupId=... and
    need the same client certificate as the SOAP call. Files are large —
    CRR network model archives run tens of megabytes each — so check the
    'size' field from the listing before pulling a wide date range.
    """
    OUTDIR.mkdir(exist_ok=True)
    for i, rep in enumerate(reports, 1):
        url = rep.get("URL")
        if not url:
            print(f"  [{i}] no URL, skipping")
            continue
        dest = OUTDIR / safe_name(rep.get("fileName") or f"report_{i}")
        if dest.exists():
            print(f"  [{i}] {dest.name} already present")
            continue
        try:
            r = requests.get(url, cert=(str(CERT), str(KEY)), timeout=300)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"  [{i}] {dest.name}  ({len(r.content):,} bytes)")
        except Exception as e:
            print(f"  [{i}] FAILED {dest.name}: {e}")


# ------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(
        description="Retrieve ERCOT reports over EWS.",
        epilog="Requires ERCOT_DUNS and ERCOT_API_USER in the environment.",
    )
    ap.add_argument(
        "--option",
        default=DEFAULT_OPTION,
        help="EMIL report type ID (default %(default)s = np7-801-m, CRR Network Model)",
    )
    ap.add_argument(
        "--days",
        type=int,
        default=365,
        help="look back this many days (default %(default)s)",
    )
    ap.add_argument(
        "--all", action="store_true", help="omit the time window — everything available"
    )
    ap.add_argument(
        "--download", action="store_true", help="download the files, not just list them"
    )
    ap.add_argument(
        "--debug", action="store_true", help="print the signed request and raw response"
    )
    ap.add_argument(
        "--sig",
        choices=["binary", "plain"],
        default="binary",
        help="X.509 token style (default %(default)s)",
    )
    ap.add_argument(
        "--digest",
        choices=["sha1", "sha256"],
        default="sha1",
        help="ERCOT requires sha1 (default)",
    )
    ap.add_argument(
        "--sig-method",
        choices=["rsa-sha1", "rsa-sha256"],
        default="rsa-sha1",
        help="ERCOT requires rsa-sha1 (default)",
    )
    args = ap.parse_args()

    duns, user = load_identity()
    check_certs()

    if args.all:
        start = end = None
        print(f"Requesting report {args.option}, no time filter")
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
        print(f"Requesting report {args.option}, {start:%Y-%m-%d} to {end:%Y-%m-%d}")

    envelope = build_request(duns, user, args.option, start, end)
    signed = sign(
        envelope, style=args.sig, digest=args.digest, sig_method=args.sig_method
    )
    resp = call_ews(signed, debug=args.debug)

    if resp.status_code != 200:
        print("\nNon-200 response:\n")
        print(resp.text[:3000])
        sys.exit(1)

    reports = parse_reports(resp.content)
    total = sum(int(r.get("size", 0) or 0) for r in reports)
    print(f"\n{len(reports)} report(s) returned, {total:,} bytes total\n")

    for i, rep in enumerate(reports, 1):
        print(f"  [{i}] {rep.get('fileName', '?')}")
        print(
            f"       {rep.get('reportGroup', '?')}  |  "
            f"operating date {rep.get('operatingDate', '?')}"
        )
        print(
            f"       created {rep.get('created', '?')}  "
            f"{int(rep.get('size', 0) or 0):,} bytes  "
            f"{rep.get('format', '?')}"
        )

    if not reports:
        print(
            "Nothing returned. Try --all to drop the time window, "
            "or --debug to inspect the raw response."
        )
    elif args.download:
        print(f"\nDownloading into {OUTDIR}/")
        download(reports)
    else:
        print("\nRe-run with --download to fetch these.")


if __name__ == "__main__":
    main()
