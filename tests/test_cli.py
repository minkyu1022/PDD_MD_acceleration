from pathlib import Path

from pdd_md import cli


def test_download_data_passes_root_to_downloader(monkeypatch, capsys, tmp_path: Path):
    seen = []

    def fake_download(root):
        seen.append(root)
        return [Path(root) / "AD-3" / "train" / "example.npz"]

    monkeypatch.setattr(cli, "download_ad3", fake_download)
    cli.main(["download-data", "--data-root", str(tmp_path)])
    assert seen == [str(tmp_path)]
    assert "example.npz" in capsys.readouterr().out
