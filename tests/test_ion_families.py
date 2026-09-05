"""Taxonomy regression tests; family labels never enter model features."""
import torch
from cf_bild.ion_families import classify_cation, classify_anion


def test_distinguish_diazolium_regioisomers():
    assert classify_cation('CCn1ccc[n+]1C') == 'pyrazolium'
    assert classify_cation('CCn1cc[n+](C)c1') == 'imidazolium'


def test_morpholinium_is_not_piperidinium():
    assert classify_cation('CC[N+]1(C)CCOCC1') == 'morpholinium'
    assert classify_cation('CC[N+]1(C)CCCCC1') == 'piperidinium'


def test_nonaromatic_diazacycle_not_imidazolium():
    assert classify_cation('CC[NH+]1C=CN(C)C1') == 'other N-heterocycle'


def test_anion_functional_groups():
    assert classify_anion('CC1(C(=O)[O-])CCCN1') == 'amino-acid'
    assert classify_anion('O=C([O-])C1CCCN1') == 'amino-acid'
    assert classify_anion('O=C[O-]') == 'carboxylate'
    assert classify_anion('Cl[Fe-](Cl)(Cl)Cl') == 'other'
    assert classify_anion('O=C([O-])O') == 'other'
    assert classify_anion('CS(=O)(=O)[O-]') == 'sulfonate/sulfate'
