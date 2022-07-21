# -*- Mode:Python -*-

import os
import sys
import socket
import helpers
from   pathlib import Path

assert 'PDKMASTER_TOP' in os.environ, 'PDKMASTER_TOP not set'
PdkMasterTop = os.environ['PDKMASTER_TOP']
NdaDirectory = PdkMasterTop + '/libs.tech/coriolis/techno'
helpers.setNdaTopDir( NdaDirectory )

# add local path
p = Path(__file__).parents[1].joinpath("non_generateds")
sys.path.insert(0, str(p))

import Cfg
from   CRL       import AllianceFramework, RoutingLayerGauge
from   helpers   import overlay, l, u, n
from   node130.sky130 import techno, StdCellLib
from   pythonlib import ethmacmem

techno.setup()
StdCellLib.setup()
ethmacmem.setup()

af = AllianceFramework.get()

with overlay.CfgCache(priority=Cfg.Parameter.Priority.UserFile) as cfg:
    cfg.misc.catchCore           = False
    cfg.misc.minTraceLevel       = 12300
    cfg.misc.maxTraceLevel       = 12400
    cfg.misc.info                = False
    cfg.misc.paranoid            = False
    cfg.misc.bug                 = False
    cfg.misc.logMode             = True
    cfg.misc.verboseLevel1       = True
    cfg.misc.verboseLevel2       = True
    cfg.etesian.graphics         = 2
    cfg.anabatic.topRoutingLayer = 'm4'
    cfg.katana.eventsLimit       = 4000000
    af  = AllianceFramework.get()
    lg5 = af.getRoutingGauge('StdCellLib').getLayerGauge( 5 )
    lg5.setType( RoutingLayerGauge.PowerSupply )
    env = af.getEnvironment()
    env.setCLOCK( '^sys_clk$|^ck|^jtag_tck$' )
