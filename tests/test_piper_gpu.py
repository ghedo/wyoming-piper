from types import SimpleNamespace

from wyoming_piper import handler


def test_rocm_execution_providers(monkeypatch) -> None:
    captured = {}

    class FakeSession:
        def set_providers(self, providers) -> None:
            captured["providers"] = providers

    class FakeVoice:
        session = FakeSession()
        config = SimpleNamespace(speaker_id_map={})

        def synthesize_wav(self, *_args) -> None:
            pass

    def fake_load(*_args, **kwargs):
        captured["use_cuda"] = kwargs["use_cuda"]
        return FakeVoice()

    monkeypatch.setattr(handler, "_VOICE", None)
    monkeypatch.setattr(handler, "_VOICE_NAME", None)
    monkeypatch.setattr(handler, "ensure_voice_exists", lambda *_args: None)
    monkeypatch.setattr(handler, "find_voice", lambda *_args: ("model", "config"))
    monkeypatch.setattr(handler.PiperVoice, "load", staticmethod(fake_load))

    event_handler = handler.PiperEventHandler.__new__(handler.PiperEventHandler)
    event_handler.cli_args = SimpleNamespace(
        use_cuda=False,
        use_rocm=True,
        data_dir=[],
        download_dir="/data",
        length_scale=None,
        noise_scale=None,
        noise_w_scale=None,
    )
    event_handler.voices_info = {}

    event_handler._synthesize_piper("test", object(), "voice", None)

    assert captured["use_cuda"] is False
    assert captured["providers"] == [
        "MIGraphXExecutionProvider",
        "CPUExecutionProvider",
    ]
