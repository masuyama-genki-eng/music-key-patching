"""Vendored pieces of salu133445/mmt (MIT, (c) 2022 Hao-Wen Dong).

Two files, copied from the repository at commit 87a8e26168c0407439e5c83e68c16deb5cac8c67, with the SMALLEST possible modification so that provenance stays checkable:

  music_x_transformers.py  the model classes, verbatim; only the CLI entry point and
                           its `import representation` / `import utils` are removed
                           (they were used by nothing else)
  representation_min.py    the encoding constants and the four functions we consume
                           (get_encoding, encode_notes, decode_notes, and the note
                           extraction helpers), verbatim; the module-level imports of
                           muspy/pretty_midi/utils are dropped because none of the
                           extracted code touches them

tests/test_mmt_adapter.py verifies the extraction is verbatim against the upstream
clone, function body by function body, so drift is loud.
"""
