#!/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR AGPL-3.0-or-later OR CERN-OHL-S-2.0+
from pdkmaster.design import library as _lbry
from pdkmaster.io import klayout as _ioklay, coriolis as _iocorio

from c4m.pdk import sky130

memlib = _lbry.Library(
    name="ethmacmem",
    tech=sky130.tech, cktfab=sky130.cktfab, layoutfab=sky130.layoutfab,
)

fab = sky130.Sky130SP6TFactory(lib=memlib, name_prefix="")
block = fab.block(
    address_groups=(3, 3, 2), word_size=32, we_size=4, cell_name="flexmem_spram_256x32_4we",
)

# Be sure layout is generated before calling merge
block.layout

_ioklay.merge(memlib)

exp = _iocorio.export.FileExporter(tech=sky130.tech, gds_layers=sky130.gds_layers)
with open("pythonlib/ethmacmem.py", "w") as f:
    f.write(exp(memlib, routinggauge=fab.spec.stdcelllib.routinggauge[0]))
