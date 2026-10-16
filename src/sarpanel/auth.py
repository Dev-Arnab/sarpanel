"""
Earthdata credential resolution.

Credentials are never accepted as command-line arguments. Anything passed on a
command line is visible in the process table to every other user on a shared
node and is captured in shell history and in scheduler job records, which on a
multi-tenant HPC system means an exposed password.

Resolution order, first match wins:

1. ``EARTHDATA_TOKEN`` environment variable (preferred; bearer token)
2. ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` environment variables
3. ``~/.netrc`` entry for ``urs.earthdata.nasa.gov``

Tokens are preferred because they expire and can be revoked without changing
the account password.
"""

from __future__ import annotations

import logging
import netrc as netrc_module
import os
import stat
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

EARTHDATA_HOST = "urs.earthdata.nasa.gov"

# A token on disk, rather than in the environment. Batch jobs do not reliably
# inherit exported variables, and a login shell may re-source a profile that
# sets stale credentials, so an environment-only token works interactively and
# then fails under the scheduler. A file works in both.
TOKEN_FILE = Path(os.environ.get("EARTHDATA_TOKEN_FILE", Path.home() / ".earthdata_token"))


@dataclass(frozen=True)
class Credentials:
    token: str | None = None
    username: str | None = None
    password: str | None = None
    source: str = "unknown"

    def __repr__(self) -> str:  # never leak secrets into logs or tracebacks
        who = self.username or "<token>"
        return f"Credentials(source={self.source!r}, user={who!r}, secret=<redacted>)"


def _read_token_file(path: Path) -> str | None:
    if not path.exists():
        return None
    mode = path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        raise PermissionError(
            f"{path} is readable by group or others. Run: chmod 600 {path}"
        )
    text = path.read_text().strip()
    # Tolerate a KEY=value line so the same file can also be sourced by a shell.
    if "=" in text.split("\n")[0] and text.upper().startswith("EARTHDATA_TOKEN"):
        text = text.split("=", 1)[1].strip().strip("'\"")
    return text or None


def resolve_credentials() -> Credentials:
    token = os.environ.get("EARTHDATA_TOKEN") or os.environ.get("ASF_TOKEN")
    if token and token.strip():
        return Credentials(token=token.strip(), source="environment token")

    file_token = _read_token_file(TOKEN_FILE)
    if file_token:
        return Credentials(token=file_token, source=f"token file ({TOKEN_FILE})")

    user = os.environ.get("EARTHDATA_USERNAME") or os.environ.get("ASF_USERNAME")
    pwd = os.environ.get("EARTHDATA_PASSWORD") or os.environ.get("ASF_PASSWORD")
    if user and pwd:
        # Values sourced from a file, or set with $(cat ...), routinely carry a
        # trailing newline. URS rejects those without saying why, which reads as
        # a wrong password.
        return Credentials(
            username=user.strip(),
            password=pwd.strip(),
            source="environment password",
        )

    netrc_path = Path(os.environ.get("NETRC", Path.home() / ".netrc"))
    if netrc_path.exists():
        _check_netrc_permissions(netrc_path)
        try:
            auth = netrc_module.netrc(str(netrc_path)).authenticators(EARTHDATA_HOST)
        except Exception as exc:  # malformed file
            raise RuntimeError(f"Could not parse {netrc_path}: {exc}") from exc
        if auth:
            login, _, password = auth
            if login and password:
                return Credentials(
                    username=login.strip(),
                    password=password.strip(),
                    source=f"netrc ({netrc_path})",
                )

    raise RuntimeError(
        "No Earthdata credentials found. In order of preference:\n"
        f"  1. A token in {TOKEN_FILE} (mode 600). Survives job submission.\n"
        "  2. EARTHDATA_TOKEN in the environment.\n"
        f"  3. A machine entry for {EARTHDATA_HOST} in ~/.netrc (mode 600).\n"
        "  4. EARTHDATA_USERNAME and EARTHDATA_PASSWORD.\n\n"
        "Generate a token at https://urs.earthdata.nasa.gov/profile"
    )


def _check_netrc_permissions(path: Path) -> None:
    mode = path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        raise PermissionError(
            f"{path} is readable by group or others. "
            f"Run: chmod 600 {path}"
        )


def build_session(creds: Credentials | None = None):
    """
    Return an authenticated ``asf_search`` session.

    Authentication is deferred until download, since the catalog query itself
    is unauthenticated. That means a successful ``plan`` says nothing about
    whether credentials work, and the first failure appears at the start of a
    processing run.
    """
    import asf_search as asf

    creds = creds or resolve_credentials()
    session = asf.ASFSession()
    try:
        if creds.token:
            session.auth_with_token(creds.token)
        else:
            session.auth_with_creds(creds.username, creds.password)
    except Exception as exc:
        raise RuntimeError(
            f"Earthdata authentication failed using {creds.source}.\n"
            f"  username: {creds.username or '<token>'}\n"
            f"  error:    {exc}\n\n"
            "Common causes, in the order worth checking:\n"
            "  0. A batch job did not inherit EARTHDATA_TOKEN and fell back to "
            "a stale password. If this worked interactively but fails under "
            f"sbatch, write the token to {TOKEN_FILE} (mode 600) instead of "
            "exporting it: a file survives job submission, an environment "
            "variable may not.\n"
            "  1. The password was rotated but the environment still holds the "
            "old one. Environment variables sourced from a file persist in the "
            "shell after the file is edited, and a login shell re-sources the "
            "profile inside the job.\n"
            "  2. A trailing newline or space in the value. Values set with "
            "$(cat file) or sourced from a .env carry one; they are stripped "
            "here, so this is unlikely but possible upstream.\n"
            "  3. The Earthdata account has not authorised the ASF data "
            "application. Log in at https://urs.earthdata.nasa.gov and accept "
            "the EULA once.\n"
            "  4. Special characters in the password mangled by shell quoting.\n\n"
            "The durable fix is a bearer token, which avoids shell quoting "
            "entirely and can be revoked without changing the account password:\n"
            "  https://urs.earthdata.nasa.gov/profile -> Generate Token\n"
            "  export EARTHDATA_TOKEN='...'\n"
            "  unset EARTHDATA_USERNAME EARTHDATA_PASSWORD ASF_USERNAME ASF_PASSWORD"
        ) from exc

    log.info("Authenticated to Earthdata via %s", creds.source)
    return session


def verify_credentials() -> bool:
    """Check that credentials work, without downloading anything."""
    build_session()
    return True
