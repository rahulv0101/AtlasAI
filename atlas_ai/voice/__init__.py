def run_voice_loop(*args, **kwargs):
    from atlas_ai.voice.wake_loop import run_voice_loop as _run_voice_loop

    return _run_voice_loop(*args, **kwargs)


__all__ = ["run_voice_loop"]
