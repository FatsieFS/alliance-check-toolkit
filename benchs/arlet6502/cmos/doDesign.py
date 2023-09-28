#!/usr/bin/env python3

import sys
import traceback
from   coriolis.Hurricane  import DbU, Breakpoint, PythonAttributes
from   coriolis            import CRL, Cfg
from   coriolis.helpers    import loadUserSettings, setTraceLevel, trace, overlay, l, u, n
from   coriolis.helpers.io import ErrorMessage, WarningMessage, catch
loadUserSettings()
from   coriolis            import plugins
#from   coriolis.Seabreeze  import SeabreezeEngine
from   coriolis.plugins.block.block         import Block
from   coriolis.plugins.block.configuration import IoPin, GaugeConf
from   coriolis.plugins.block.spares        import Spares
from   coriolis.plugins.core2chip.niolib    import CoreToChip
from   coriolis.plugins.chip.configuration  import ChipConf
from   coriolis.plugins.chip.chip           import Chip


af = CRL.AllianceFramework.get()


def scriptMain ( **kw ):
    """The mandatory function to be called by Coriolis CGT/Unicorn."""
    with overlay.CfgCache(priority=Cfg.Parameter.Priority.UserFile) as cfg:
        cfg.misc.catchCore              = False
        cfg.misc.info                   = False
        cfg.misc.paranoid               = False
        cfg.misc.bug                    = False
        cfg.misc.logMode                = False
        cfg.misc.verboseLevel1          = True
        cfg.misc.verboseLevel2          = True
        cfg.misc.minTraceLevel          = 16000
        cfg.misc.maxTraceLevel          = 17000

    global af
    rvalue = True
    try:
       #setTraceLevel( 550 )
       #Breakpoint.setStopLevel( 100 )
        buildChip = False
        cell, editor = plugins.kwParseMain( **kw )
        cell = af.getCell( 'arlet6502', CRL.Catalog.State.Logical )
        if editor:
            editor.setCell( cell ) 
            editor.setDbuMode( DbU.StringModePhysical )
        ioPadsSpec = [ (IoPin.WEST , None, 'di_0'       , 'di(0)'  , 'di(0)'  )
                     , (IoPin.WEST , None, 'di_1'       , 'di(1)'  , 'di(1)'  )
                     , (IoPin.WEST , None, 'di_2'       , 'di(2)'  , 'di(2)'  )
                     , (IoPin.WEST , None, 'di_3'       , 'di(3)'  , 'di(3)'  )
                     , (IoPin.WEST , None, 'iopower_0'  , 'iovdd'  )
                     , (IoPin.WEST , None, 'power_0'    , 'vdd'    )
                     , (IoPin.WEST , None, 'ground_0'   , 'vss'    )
                     , (IoPin.WEST , None, 'ioground_0' , 'vss'    )
                     , (IoPin.WEST , None, 'di_4'       , 'di(4)'  , 'di(4)'  )
                     , (IoPin.WEST , None, 'di_5'       , 'di(5)'  , 'di(5)'  )
                     , (IoPin.WEST , None, 'di_6'       , 'di(6)'  , 'di(6)'  )
                     , (IoPin.WEST , None, 'di_7'       , 'di(7)'  , 'di(7)'  )

                     , (IoPin.SOUTH, None, 'do_0'       , 'do(0)'  , 'do(0)'  )
                     , (IoPin.SOUTH, None, 'do_1'       , 'do(1)'  , 'do(1)'  )
                     , (IoPin.SOUTH, None, 'do_2'       , 'do(2)'  , 'do(2)'  )
                     , (IoPin.SOUTH, None, 'do_3'       , 'do(3)'  , 'do(3)'  )
                     , (IoPin.SOUTH, None, 'do_4'       , 'do(4)'  , 'do(4)'  )
                     , (IoPin.SOUTH, None, 'ioground_1' , 'vss'    )
                     , (IoPin.SOUTH, None, 'power_1'    , 'vdd'    )
                     , (IoPin.SOUTH, None, 'ground_1'   , 'vss'    )
                     , (IoPin.SOUTH, None, 'iopower_1'  , 'iovdd'  )
                     , (IoPin.SOUTH, None, 'do_5'       , 'do(5)'  , 'do(5)'  )
                     , (IoPin.SOUTH, None, 'do_6'       , 'do(6)'  , 'do(6)'  )
                     , (IoPin.SOUTH, None, 'do_7'       , 'do(7)'  , 'do(7)'  )
                     , (IoPin.SOUTH, None, 'a_0'        , 'a(0)'   , 'a(0)'   )
                     , (IoPin.SOUTH, None, 'a_1'        , 'a(1)'   , 'a(1)'   )

                     , (IoPin.EAST , None, 'a_2'        , 'a(2)'   , 'a(2)'   )
                     , (IoPin.EAST , None, 'a_3'        , 'a(3)'   , 'a(3)'   )
                     , (IoPin.EAST , None, 'a_4'        , 'a(4)'   , 'a(4)'   )
                     , (IoPin.EAST , None, 'a_5'        , 'a(5)'   , 'a(5)'   )
                     , (IoPin.EAST , None, 'a_6'        , 'a(6)'   , 'a(6)'   )
                     , (IoPin.EAST , None, 'a_7'        , 'a(7)'   , 'a(7)'   )
                     , (IoPin.EAST , None, 'iopower_2'  , 'iovdd'  )
                     , (IoPin.EAST , None, 'power_2'    , 'vdd'    )
                     , (IoPin.EAST , None, 'ground_2'   , 'vss'    )
                     , (IoPin.EAST , None, 'ioground_2' , 'vss'    )
                     , (IoPin.EAST , None, 'a_8'        , 'a(8)'   , 'a(8)'   )
                     , (IoPin.EAST , None, 'a_9'        , 'a(9)'   , 'a(9)'   )
                     , (IoPin.EAST , None, 'a_10'       , 'a(10)'  , 'a(10)'  )
                     , (IoPin.EAST , None, 'a_11'       , 'a(11)'  , 'a(11)'  )
                     , (IoPin.EAST , None, 'a_12'       , 'a(12)'  , 'a(12)'  )
                     , (IoPin.EAST , None, 'a_13'       , 'a(13)'  , 'a(13)'  )

                     , (IoPin.NORTH, None, 'irq'        , 'irq'    , 'irq'    )
                     , (IoPin.NORTH, None, 'nmi'        , 'nmi'    , 'nmi'    )
                     , (IoPin.NORTH, None, 'rdy'        , 'rdy'    , 'rdy'    )
                     , (IoPin.NORTH, None, 'clk'        , 'clk'    , 'clk'    )
                     , (IoPin.NORTH, None, 'iopower_3'  , 'iovdd'  )
                     , (IoPin.NORTH, None, 'power_3'    , 'vdd'    )
                     , (IoPin.NORTH, None, 'ground_3'   , 'vss'    )
                     , (IoPin.NORTH, None, 'ioground_3' , 'vss'    )
                     , (IoPin.NORTH, None, 'reset'      , 'reset'  , 'reset'  )
                     , (IoPin.NORTH, None, 'we'         , 'we'     , 'we'     )
                     , (IoPin.NORTH, None, 'a_14'       , 'a(14)'  , 'a(14)'  )
                     , (IoPin.NORTH, None, 'a_15'       , 'a(15)'  , 'a(15)'  )
                     ]
        ioPinsSpec = [ (IoPin.WEST |IoPin.A_BEGIN, 'di({})'  ,    l(50.0), l(50.0),  8)
                     , (IoPin.WEST |IoPin.A_BEGIN, 'do({})'  , 14*l(50.0), l(50.0),  8)
                     , (IoPin.EAST |IoPin.A_BEGIN, 'a({})'   ,    l(50.0), l(50.0), 16)
                     
                    #, (IoPin.NORTH|IoPin.A_BEGIN, 'clk'     , 10*l(50.0),      0 ,  1)
                     , (IoPin.NORTH|IoPin.A_BEGIN, 'irq'     , 11*l(50.0),      0 ,  1)
                     , (IoPin.NORTH|IoPin.A_BEGIN, 'nmi'     , 12*l(50.0),      0 ,  1)
                     , (IoPin.NORTH|IoPin.A_BEGIN, 'rdy'     , 13*l(50.0),      0 ,  1)
                     , (IoPin.NORTH|IoPin.A_BEGIN, 'we'      , 14*l(50.0),      0 ,  1)
                    #, (IoPin.NORTH|IoPin.A_BEGIN, 'reset'   , 15*l(50.0),      0 ,  1)
                     ]
        arlet6502Conf = ChipConf( cell, ioPins=ioPinsSpec, ioPads=ioPadsSpec ) 
        arlet6502Conf.cfg.etesian.bloat               = 'disabled'
       #arlet6502Conf.cfg.etesian.bloat               = 'nsxlib'
        arlet6502Conf.cfg.etesian.uniformDensity      = True
        arlet6502Conf.cfg.etesian.aspectRatio         = 1.0
       # etesian.spaceMargin is ignored if the coreSize is directly set.
       #arlet6502Conf.cfg.etesian.spaceMargin         = 0.10
       #arlet6502Conf.cfg.anabatic.searchHalo         = 2
        arlet6502Conf.cfg.anabatic.globalIterations   = 10
        arlet6502Conf.cfg.anabatic.topRoutingLayer    = 'METAL5'
        arlet6502Conf.cfg.katana.hTracksReservedLocal = 10
        arlet6502Conf.cfg.katana.vTracksReservedLocal = 10
        arlet6502Conf.cfg.katana.hTracksReservedMin   = 7
        arlet6502Conf.cfg.katana.vTracksReservedMin   = 5
        arlet6502Conf.cfg.katana.trackFill            = 0
        arlet6502Conf.cfg.katana.runRealignStage      = True
        arlet6502Conf.cfg.block.spareSide             = l(7*50.0)
        arlet6502Conf.cfg.chip.supplyRailWidth        = l(250.0)
        arlet6502Conf.cfg.chip.supplyRailPitch        = l(150.0)
        arlet6502Conf.cfg.chip.use45corners           = True
        arlet6502Conf.editor              = editor
        arlet6502Conf.useSpares           = True
        arlet6502Conf.useHFNS             = False
        arlet6502Conf.bColumns            = 2
        arlet6502Conf.bRows               = 2
        arlet6502Conf.chipName            = 'chip'
        arlet6502Conf.coreSize            = ( l( 35*50.0), l( 39*50.0) )
        arlet6502Conf.chipSize            = ( l(  9400.0), l(10400.0) )
        arlet6502Conf.coreToChipClass     = CoreToChip
        if buildChip:
            arlet6502Conf.useHTree( 'clk_from_pad', Spares.HEAVY_LEAF_LOAD )
            arlet6502Conf.useHTree( 'reset_from_pad' )
            chipBuilder = Chip( arlet6502Conf )
            chipBuilder.doChipNetlist()
            chipBuilder.doChipFloorplan()
            rvalue = chipBuilder.doPnR()
            chipBuilder.save()
        else:
            arlet6502Conf.useHTree( 'clk', Spares.HEAVY_LEAF_LOAD )
            arlet6502Conf.useHTree( 'reset' )
            blockBuilder = Block( arlet6502Conf )
            rvalue = blockBuilder.doPnR()
            blockBuilder.save()
    except Exception as e:
        catch( e )
        rvalue = False
    sys.stdout.flush()
    sys.stderr.flush()
    return rvalue


if __name__ == '__main__':
    rvalue = scriptMain()
    shellRValue = 0 if rvalue else 1
    sys.exit( shellRValue )
