"""Command-line entry points that do not need SNAP, a catalog, or credentials."""

import pytest

from sarpanel.cli import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "sarpanel" in capsys.readouterr().out


def test_plan_requires_a_config():
    with pytest.raises(SystemExit) as exc:
        main(["plan"])
    assert exc.value.code == 2


def test_doctor_runs_when_credentials_are_missing(monkeypatch, capsys):
    import sarpanel.auth as auth

    def missing():
        raise RuntimeError("no creds in test")

    monkeypatch.setattr(auth, "resolve_credentials", missing)
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "Software versions" in out
    assert "NOT AVAILABLE" in out
    assert "no creds in test" in out


def test_doctor_warns_when_incidence_angle_cannot_be_exported(monkeypatch, capsys):
    import sarpanel.auth as auth
    import sarpanel.processing as processing

    def missing():
        raise RuntimeError("no creds")

    monkeypatch.setattr(auth, "resolve_credentials", missing)

    class Capabilities:
        def report(self):
            return {"export_extra": False, "terrain_flattening": True}

    monkeypatch.setattr(processing, "GeocodeCapabilities", Capabilities)
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "export_extra" in out
    assert "WARNING" in out
    assert "Upgrade pyroSAR" in out
