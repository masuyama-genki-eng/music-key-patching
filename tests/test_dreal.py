"""D-REAL kern parser tests (SPEC §1.3). No corpus needed: synthetic kern text."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.datagen.dreal import (kern_duration_to_16ths, kern_pitch_to_midi,
                               parse_chorale, parse_key, parse_meter)
from src.tokenizer.vocab import VOCAB, pitch_of

KERN = """!!!COM: Bach, Johann Sebastian
**kern\t**kern\t**kern\t**kern
*clefF4\t*clefGv2\t*clefG2\t*clefG2
*k[f#]\t*k[f#]\t*k[f#]\t*k[f#]
*G:\t*G:\t*G:\t*G:
*M4/4\t*M4/4\t*M4/4\t*M4/4
4GG\t4B\t4d\t4g
=1\t=1\t=1\t=1
4G\t4B\t4d\t2g
4E\t8cL\t4e\t.
.\t8BJ\t.\t.
*-\t*-\t*-\t*-
"""


def test_kern_pitch_octaves():
    assert kern_pitch_to_midi("c", "") == 60          # middle C
    assert kern_pitch_to_midi("C", "") == 48
    assert kern_pitch_to_midi("CC", "") == 36
    assert kern_pitch_to_midi("cc", "") == 72
    assert kern_pitch_to_midi("GG", "") == 43         # a chorale bass G
    assert kern_pitch_to_midi("f", "#") == 66
    assert kern_pitch_to_midi("b", "-") == 70


def test_kern_durations():
    assert kern_duration_to_16ths(4, 0) == 4          # quarter
    assert kern_duration_to_16ths(8, 0) == 2          # eighth
    assert kern_duration_to_16ths(2, 0) == 8          # half
    assert kern_duration_to_16ths(1, 0) == 16         # whole
    assert kern_duration_to_16ths(4, 1) == 6          # dotted quarter
    assert kern_duration_to_16ths(16, 0) == 1         # sixteenth


def test_key_and_meter_designations():
    assert parse_key("*G:\t*G:") == 7                 # G major
    assert parse_key("*a:\t*a:") == 9 + 12            # A minor
    assert parse_key("*E-:\t*E-:") == 3               # E-flat major
    assert parse_key("*f#:\t*f#:") == 6 + 12          # F-sharp minor
    assert parse_key("*clefG2") is None
    assert parse_meter("*M3/4\t*M3/4") == (3, 4)


def test_parse_chorale_tokens_and_labels(tmp_path):
    p = tmp_path / "t.krn"
    p.write_text(KERN)
    c = parse_chorale(p)
    assert c["key"] == 7 and c["meter"] == (4, 4)
    assert len(c["tokens"]) == len(c["key_labels"])
    assert all(t in VOCAB for t in c["tokens"])       # leak-free vocabulary
    assert all(k == 7 for k in c["key_labels"])
    # first sounding chord is the SATB G triad: G2 B3 D4 G4
    pitches = [pitch_of(t) for t in c["tokens"] if pitch_of(t) is not None]
    assert pitches[:4] == [43, 59, 62, 67]


def test_parse_chorale_rejects_keyless(tmp_path):
    p = tmp_path / "nokey.krn"
    p.write_text(KERN.replace("*G:\t*G:\t*G:\t*G:\n", ""))
    assert parse_chorale(p) is None


# Humdrum time model: a slice advances to the earliest END of any sounding note,
# not by the shortest note starting on the line. With voices of unequal length
# (soprano dotted-quarter + eighth against three quarters) the naive rule
# desynchronises the parts and the bar comes out too long.
UNEQUAL = """**kern\t**kern
*k[f#]\t*k[f#]
*G:\t*G:
*M3/4\t*M3/4
=1\t=1
4G\t4.b
4D\t.
.\t8a
4E\t4g
=2\t=2
4G\t4g
4D\t4b
4E\t4g
=3\t=3
4G\t4g
*-\t*-
"""


def test_timing_model_unequal_voices(tmp_path):
    p = tmp_path / "u.krn"
    p.write_text(UNEQUAL)
    c = parse_chorale(p)
    assert c is not None
    # bar 1 must be exactly 3/4 = 12 sixteenths long
    assert c["bars"][2] - c["bars"][1] == 12
    # the soprano eighth lands on the dotted quarter's end (6), not at 8
    onsets = {(t, o) for t, o in zip(c["tokens"], c["onsets"])}
    assert ("PITCH_69", c["bars"][1] + 6) in onsets       # a4 = MIDI 69


def test_meter_change_rejected(tmp_path):
    """A bar that does not match its declared meter means an unhandled meter
    change; the score is excluded rather than mistokenized."""
    p = tmp_path / "bad.krn"
    p.write_text(UNEQUAL.replace("*M3/4\t*M3/4", "*M4/4\t*M4/4"))
    assert parse_chorale(p) is None
