
from coriolis.helpers   import l, u, n
from coriolis.Hurricane import DataBase, Technology, Layer, BasicLayer, DiffusionLayer, \
                               TransistorLayer, RegularLayer, ContactLayer, ViaLayer

__all__ = [ 'setup' ]


def setup ():
    tech = DataBase.getDB().getTechnology()
    tech.addLayerAlias( 'nwm'     ,  'nWell'     )
    tech.addLayerAlias( 'activ' ,  'active'    )
   #tech.addLayerAlias( 'poly'    ,  'poly'      )
    tech.addLayerAlias( 'psdm'    ,  'pImplant'  )
    tech.addLayerAlias( 'nsdm'    ,  'nImplant'  )
    tech.addLayerAlias( 'cont'   ,  'cut0'      )
    tech.addLayerAlias( 'M1'      ,  'metal1'    )
    tech.addLayerAlias( 'via12'    ,  'cut1'      )
    tech.addLayerAlias( 'M2'      ,  'metal2'    )
    tech.addLayerAlias( 'via23'     ,  'cut2'      )
    tech.addLayerAlias( 'M3'      ,  'metal3'    )
    tech.addLayerAlias( 'via34'    ,  'cut3'      )
    tech.addLayerAlias( 'M4'      ,  'metal4'    )
    tech.addLayerAlias( 'via45'    ,  'cut4'      )
    tech.addLayerAlias( 'M5'      ,  'metla5'    )
    tech.addLayerAlias( 'via56'    ,  'cut5'      )
    tech.addLayerAlias( 'M6'      ,  'metal6'    )
    tech.addLayerAlias( 'M1.block', 'blockage1'  )
    tech.addLayerAlias( 'M2.block', 'blockage2'  )
    tech.addLayerAlias( 'M3.block', 'blockage3'  )
    tech.addLayerAlias( 'M4.block', 'blockage4'  )
    tech.addLayerAlias( 'M5.block', 'blockage5'  )
    tech.addLayerAlias( 'M6.block', 'blockage6'  )
    tech.addLayerAlias( 'capm'    ,  'metcap'    )
    tech.addLayerAlias( 'capm'    ,  'metcapdum' )
    tech.addLayerAlias( 'M5'      ,  'metbot'    )

    nWell     = tech.getBasicLayer( 'nwm'     )
    active    = tech.getBasicLayer( 'activ' )
    poly      = tech.getBasicLayer( 'poly'    )
    pImplant  = tech.getBasicLayer( 'psdm'    )
    nImplant  = tech.getBasicLayer( 'nsdm'    )
    cut0      = tech.getBasicLayer( 'cont'   )
    metal1    = tech.getBasicLayer( 'M1'      )
    cut1      = tech.getBasicLayer( 'via12'    )
    metal2    = tech.getBasicLayer( 'M2'      )
    cut2      = tech.getBasicLayer( 'via23'     )
    metal3    = tech.getBasicLayer( 'M3'      )
    cut3      = tech.getBasicLayer( 'via34'    )
    metal4    = tech.getBasicLayer( 'M4'      )
    cut4      = tech.getBasicLayer( 'via45'    )
    metal5    = tech.getBasicLayer( 'M5'      )
    cut5      = tech.getBasicLayer( 'via56'    )
    metal6    = tech.getBasicLayer( 'M6'      )
    blockage1 = tech.getBasicLayer( 'blockage1' )
    blockage2 = tech.getBasicLayer( 'blockage2' )
    blockage3 = tech.getBasicLayer( 'blockage3' )
    blockage4 = tech.getBasicLayer( 'blockage4' )
    blockage5 = tech.getBasicLayer( 'blockage5' )
    blockage6 = tech.getBasicLayer( 'blockage6' )

    # Composite/Symbolic layers.
    NWELL       = RegularLayer   .create( tech, 'NWELL'      , nWell    )
   #PWELL       = RegularLayer   .create( tech, 'PWELL'      , pWell    )
    NTIE        = DiffusionLayer .create( tech, 'NTIE'       , nImplant , active, nWell)
    PTIE        = DiffusionLayer .create( tech, 'PTIE'       , pImplant , active, None)
    NDIF        = DiffusionLayer .create( tech, 'NDIF'       , nImplant , active, None )
    PDIF        = DiffusionLayer .create( tech, 'PDIF'       , pImplant , active, None )
    GATE        = DiffusionLayer .create( tech, 'GATE'       , poly     , active, None )
    NTRANS      = TransistorLayer.create( tech, 'NTRANS'     , nImplant , active, poly, None )
    PTRANS      = TransistorLayer.create( tech, 'PTRANS'     , pImplant , active, poly, nWell )
    POLY        = RegularLayer   .create( tech, 'POLY'       , poly     )
    METAL1      = RegularLayer   .create( tech, 'METAL1'     , metal1   )
    METAL2      = RegularLayer   .create( tech, 'METAL2'     , metal2   )
    METAL3      = RegularLayer   .create( tech, 'METAL3'     , metal3   )
    METAL4      = RegularLayer   .create( tech, 'METAL4'     , metal4   )
    METAL5      = RegularLayer   .create( tech, 'METAL5'     , metal5   )
    METAL6      = RegularLayer   .create( tech, 'METAL6'     , metal6   )
    CONT_BODY_N = ContactLayer   .create( tech, 'CONT_BODY_N', nImplant , active, cut0, metal1, None )
    CONT_BODY_P = ContactLayer   .create( tech, 'CONT_BODY_P', pImplant , active, cut0, metal1, None )
    CONT_DIF_N  = ContactLayer   .create( tech, 'CONT_DIF_N' , nImplant , active, cut0, metal1, None )
    CONT_DIF_P  = ContactLayer   .create( tech, 'CONT_DIF_P' , pImplant , active, cut0, metal1, None )
    CONT_POLY   = ViaLayer       .create( tech, 'CONT_POLY'  ,              poly, cut0, metal1 )
    
    # VIAs for symbolic technologies.
    VIA12      = ViaLayer    .create( tech, 'VIA12'     , metal1, cut1, metal2  )
    VIA23      = ViaLayer    .create( tech, 'VIA23'     , metal2, cut2, metal3  )
   #VIA23cap   = ViaLayer    .create( tech, 'VIA23cap'  , metcap, cut2, metal3  )
    VIA34      = ViaLayer    .create( tech, 'VIA34'     , metal3, cut3, metal4  )
    VIA45      = ViaLayer    .create( tech, 'VIA45'     , metal4, cut4, metal5  )
    VIA56      = ViaLayer    .create( tech, 'VIA56'     , metal5, cut5, metal6  )
    #BLOCKAGE1  = RegularLayer.create( tech, 'BLOCKAGE1' , blockage1  )
    #BLOCKAGE2  = RegularLayer.create( tech, 'BLOCKAGE2' , blockage2  )
    #BLOCKAGE3  = RegularLayer.create( tech, 'BLOCKAGE3' , blockage3  )
    #BLOCKAGE4  = RegularLayer.create( tech, 'BLOCKAGE4' , blockage4  )
    #BLOCKAGE5  = RegularLayer.create( tech, 'BLOCKAGE5' , blockage5  )
    #BLOCKAGE6  = RegularLayer.create( tech, 'BLOCKAGE6' , blockage6  )
    
    tech.setSymbolicLayer( CONT_BODY_N.getName() )
    tech.setSymbolicLayer( CONT_BODY_P.getName() )
    tech.setSymbolicLayer( CONT_DIF_N .getName() )
    tech.setSymbolicLayer( CONT_DIF_P .getName() )
    tech.setSymbolicLayer( CONT_POLY  .getName() )
    tech.setSymbolicLayer( POLY       .getName() )
    tech.setSymbolicLayer( METAL1     .getName() )
    tech.setSymbolicLayer( METAL2     .getName() )
    tech.setSymbolicLayer( METAL3     .getName() )
    tech.setSymbolicLayer( METAL4     .getName() )
    tech.setSymbolicLayer( METAL5     .getName() )
    tech.setSymbolicLayer( METAL6     .getName() )
    #tech.setSymbolicLayer( BLOCKAGE1  .getName() )
    #tech.setSymbolicLayer( BLOCKAGE2  .getName() )
    #tech.setSymbolicLayer( BLOCKAGE3  .getName() )
    #tech.setSymbolicLayer( BLOCKAGE4  .getName() )
    #tech.setSymbolicLayer( BLOCKAGE5  .getName() )
    #tech.setSymbolicLayer( BLOCKAGE6  .getName() )
    tech.setSymbolicLayer( VIA12      .getName() )
    tech.setSymbolicLayer( VIA23      .getName() )
    tech.setSymbolicLayer( VIA34      .getName() )
    tech.setSymbolicLayer( VIA45      .getName() )
    tech.setSymbolicLayer( VIA56      .getName() )
    
    NWELL.setExtentionCap( nWell, l(0.0) )
   #PWELL.setExtentionCap( pWell, l(0.0) )
    
    NTIE.setMinimalSize   (           l(3.0) )
    NTIE.setExtentionCap  ( nWell   , l(1.5) )
    NTIE.setExtentionWidth( nWell   , l(0.5) )
    NTIE.setExtentionCap  ( nImplant, l(1.0) )
    NTIE.setExtentionWidth( nImplant, l(0.5) )
    NTIE.setExtentionCap  ( active  , l(0.5) )
    NTIE.setExtentionWidth( active  , l(0.0) )
    
    PTIE.setMinimalSize   (           l(3.0) )
    PTIE.setExtentionCap  ( nWell   , l(1.5) )
    PTIE.setExtentionWidth( nWell   , l(0.5) )
    PTIE.setExtentionCap  ( nImplant, l(1.0) )
    PTIE.setExtentionWidth( nImplant, l(0.5) )
    PTIE.setExtentionCap  ( active  , l(0.5) )
    PTIE.setExtentionWidth( active  , l(0.0) )
    
    NDIF.setMinimalSize   (           l(3.0) )
    NDIF.setExtentionCap  ( nImplant, l(1.0) )
    NDIF.setExtentionWidth( nImplant, l(0.5) )
    NDIF.setExtentionCap  ( active  , l(0.5) )
    NDIF.setExtentionWidth( active  , l(0.0) )
    
    PDIF.setMinimalSize   (           l(3.0) )
    PDIF.setExtentionCap  ( pImplant, l(1.0) )
    PDIF.setExtentionWidth( pImplant, l(0.5) )
    PDIF.setExtentionCap  ( active  , l(0.5) )
    PDIF.setExtentionWidth( active  , l(0.0) )
    
    GATE.setMinimalSize   (           l(1.0) )
    GATE.setExtentionCap  ( poly    , l(1.5) )
    
    NTRANS.setMinimalSize   (           l( 1.0) )
    NTRANS.setExtentionCap  ( nImplant, l(-1.0) )
    NTRANS.setExtentionWidth( nImplant, l( 2.5) )
    NTRANS.setExtentionCap  ( active  , l(-1.5) )
    NTRANS.setExtentionWidth( active  , l( 2.0) )
    
    PTRANS.setMinimalSize   (           l( 1.0) )
    PTRANS.setExtentionCap  ( nWell   , l(-1.0) )
    PTRANS.setExtentionWidth( nWell   , l( 4.5) )
    PTRANS.setExtentionCap  ( pImplant, l(-1.0) )
    PTRANS.setExtentionWidth( pImplant, l( 4.0) )
    PTRANS.setExtentionCap  ( active  , l(-1.5) )
    PTRANS.setExtentionWidth( active  , l( 3.0) )
    
    POLY .setMinimalSize   (           l(1.0) )
    POLY .setExtentionCap  ( poly    , l(0.5) )
    #POLY2.setMinimalSize   (           l(1.0) )
    #POLY2.setExtentionCap  ( poly    , l(0.5) )
    
    METAL1 .setMinimalSize   (           l(1.0) )
    METAL1 .setExtentionCap  ( metal1  , l(0.5) )
    METAL2 .setMinimalSize   (           l(1.0) )
    METAL2 .setExtentionCap  ( metal2  , l(1.0) )
    METAL3 .setMinimalSize   (           l(1.0) )
    METAL3 .setExtentionCap  ( metal3  , l(1.0) )
    METAL4 .setMinimalSize   (           l(1.0) )
    METAL4 .setExtentionCap  ( metal4  , l(1.0) )
    METAL4 .setMinimalSpacing(           l(3.0) )
    METAL5 .setMinimalSize   (           l(2.0) )
    METAL5 .setExtentionCap  ( metal5  , l(1.0) )
    #METAL6 .setMinimalSize   (           l(2.0) )
    #METAL6 .setExtentionCap  ( metal6  , l(1.0) )
    #METAL7 .setMinimalSize   (           l(2.0) )
    #METAL7 .setExtentionCap  ( metal7  , l(1.0) )
    #METAL8 .setMinimalSize   (           l(2.0) )
    #METAL8 .setExtentionCap  ( metal8  , l(1.0) )
    #METAL9 .setMinimalSize   (           l(2.0) )
    #METAL9 .setExtentionCap  ( metal9  , l(1.0) )
    #METAL10.setMinimalSize   (           l(2.0) )
    #METAL10.setExtentionCap  ( metal10 , l(1.0) )
    
    # Contacts (i.e. Active <--> Metal) (symbolic).
    CONT_BODY_N.setMinimalSize(           l( 1.0) )
    CONT_BODY_N.setEnclosure  ( nWell   , l( 1.5), Layer.EnclosureH|Layer.EnclosureV )
    CONT_BODY_N.setEnclosure  ( nImplant, l( 1.5), Layer.EnclosureH|Layer.EnclosureV )
    CONT_BODY_N.setEnclosure  ( active  , l( 1.0), Layer.EnclosureH|Layer.EnclosureV )
    CONT_BODY_N.setEnclosure  ( metal1  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    
    CONT_BODY_P.setMinimalSize(           l( 1.0) )
    #CONT_BODY_P.setEnclosure  ( pWell   , l( 1.5), Layer.EnclosureH|Layer.EnclosureV )
    CONT_BODY_P.setEnclosure  ( pImplant, l( 1.5), Layer.EnclosureH|Layer.EnclosureV )
    CONT_BODY_P.setEnclosure  ( active  , l( 1.0), Layer.EnclosureH|Layer.EnclosureV )
    CONT_BODY_P.setEnclosure  ( metal1  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    
    CONT_DIF_N.setMinimalSize(           l( 1.0) )
    CONT_DIF_N.setEnclosure  ( nImplant, l( 1.0), Layer.EnclosureH|Layer.EnclosureV )
    CONT_DIF_N.setEnclosure  ( active  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    CONT_DIF_N.setEnclosure  ( metal1  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    
    CONT_DIF_P.setMinimalSize(           l( 1.0) )
    CONT_DIF_P.setEnclosure  ( pImplant, l( 1.0), Layer.EnclosureH|Layer.EnclosureV )
    CONT_DIF_P.setEnclosure  ( active  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    CONT_DIF_P.setEnclosure  ( metal1  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    
    CONT_POLY.setMinimalSize(           l( 1.0) )
    CONT_POLY.setEnclosure  ( poly    , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    CONT_POLY.setEnclosure  ( metal1  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    
    # VIAs (i.e. Metal <--> Metal) (symbolic).
    VIA12 .setMinimalSize   (           l( 1.0) )
    VIA12 .setEnclosure     ( metal1  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    VIA12 .setEnclosure     ( metal2  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    VIA12 .setMinimalSpacing(           l( 4.0) )
    VIA23 .setMinimalSize   (           l( 1.0) )
    VIA23 .setEnclosure     ( metal2  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    VIA23 .setEnclosure     ( metal3  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    VIA23 .setMinimalSpacing(           l( 4.0) )
    VIA34 .setMinimalSize   (           l( 1.0) )
    VIA34 .setEnclosure     ( metal3  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    VIA34 .setEnclosure     ( metal4  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    VIA34 .setMinimalSpacing(           l( 4.0) )
    VIA45 .setMinimalSize   (           l( 1.0) )
    VIA45 .setEnclosure     ( metal4  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    VIA45 .setEnclosure     ( metal5  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    VIA45 .setMinimalSpacing(           l( 4.0) )
    #VIA56 .setMinimalSize   (           l( 1.0) )
    #VIA56 .setEnclosure     ( metal5  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    #VIA56 .setEnclosure     ( metal6  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    #VIA56 .setMinimalSpacing(           l( 4.0) )
    #VIA67 .setMinimalSize   (           l( 1.0) )
    #VIA67 .setEnclosure     ( metal6  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    #VIA67 .setEnclosure     ( metal7  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    #VIA67 .setMinimalSpacing(           l( 4.0) )
    #VIA78 .setMinimalSpacing(           l( 4.0) )
    #VIA78 .setMinimalSize   (           l( 1.0) )
    #VIA78 .setEnclosure     ( metal7  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    #VIA78 .setEnclosure     ( metal8  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    #VIA78 .setMinimalSpacing(           l( 4.0) )
    #VIA89 .setMinimalSize   (           l( 1.0) )
    #VIA89 .setEnclosure     ( metal8  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    #VIA89 .setEnclosure     ( metal9  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    #VIA89 .setMinimalSpacing(           l( 4.0) )
    #VIA910.setMinimalSize   (           l( 1.0) )
    #VIA910.setEnclosure     ( metal9  , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    #VIA910.setEnclosure     ( metal10 , l( 0.5), Layer.EnclosureH|Layer.EnclosureV )
    #VIA910.setMinimalSpacing(           l( 4.0) )
    
    # Blockages (symbolic).
    #BLOCKAGE1 .setMinimalSize (             l( 1.0) )
    #BLOCKAGE1 .setExtentionCap( blockage1 , l( 0.5) )
    #BLOCKAGE2 .setMinimalSize (             l( 2.0) )
    #BLOCKAGE2 .setExtentionCap( blockage2 , l( 0.5) )
    #BLOCKAGE3 .setMinimalSize (             l( 2.0) )
    #BLOCKAGE3 .setExtentionCap( blockage3 , l( 0.5) )
    #BLOCKAGE4 .setMinimalSize (             l( 2.0) )
    #BLOCKAGE4 .setExtentionCap( blockage4 , l( 0.5) )
    #BLOCKAGE5 .setMinimalSize (             l( 2.0) )
    #BLOCKAGE5 .setExtentionCap( blockage5 , l( 1.0) )
    #BLOCKAGE6 .setMinimalSize (             l( 2.0) )
    #BLOCKAGE6 .setExtentionCap( blockage6 , l( 1.0) )
    #BLOCKAGE7 .setMinimalSize (             l( 2.0) )
    #BLOCKAGE7 .setExtentionCap( blockage7 , l( 1.0) )
    #BLOCKAGE8 .setMinimalSize (             l( 2.0) )
    #BLOCKAGE8 .setExtentionCap( blockage8 , l( 1.0) )
    #BLOCKAGE9 .setMinimalSize (             l( 2.0) )
    #BLOCKAGE9 .setExtentionCap( blockage9 , l( 1.0) )
    #BLOCKAGE10.setMinimalSize (             l( 2.0) )
    #BLOCKAGE10.setExtentionCap( blockage10, l( 1.0) )
