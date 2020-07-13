set(CXX_BASE_FILES
    AnisotropicMaterialBlock.C
    CheckPoint.C
    CurvilinearInterface2.C
    ESSI3D.C
    ESSI3DHDF5.C
    EW.C
    EtreeFile.C
    Filter.C
    ForcingTwilight.C
    GeographicProjection.C
    GridGenerator.C
    GridGeneratorGaussianHill.C
    GridGeneratorGeneral.C
    GridPointSource.C
    Image.C
    Image3D.C
    MaterialBlock.C
    MaterialData.C
    MaterialIfile.C
    MaterialInvtest.C
    MaterialPfile.C
    MaterialProperty.C
    MaterialRfile.C
    MaterialSfile.C
    MaterialVolimagefile.C
    Mspace.C
    Parallel_IO.C
    Patch.C
    Polynomial.C
    Qspline.C
    RHS43DEV.C
    RandomizedMaterial.C
    Sarray.C
    SecondOrderSection.C
    Source.C
    SuperGrid.C
    TestEcons.C
    TestRayleighWave.C
    TestTwilight.C
    TimeSeries.C
    addmemvarforcing2.C
    consintp.C
    curvilinear4sgwind.C
    curvilinearGrid.C
    evenIevenJinterp.C
    evenIoddJinterp.C
    geodyn.C
    material.C
    oddIevenJinterp.C
    oddIoddJinterp.C
    parallelStuff.C
    parseInputFile.C
    readhdf5.C
    rhs4th3windc2.C
    sachdf5.C
    sacutils.C
    setupRun.C
    solve.C
    time_functions.C)

set(FORTRAN_BASE_FILES
    lamb_exact_numquad.f
    rayleighfort.f)

set(CXX_COMPUTE_FILES
    addsg4windc.C            
    addsgdc.C		 
    anisomtrltocurvilinearc.C
    bcfortanisgc.C		 
    bcfortc.C		 	   
    bcfreesurfcurvanic.C	 
    bndryOpNoGhostc.C	 
    boundaryOpc.C		 
    checkanisomtrlc.C	 
    computedtanisoc.C	 
    curvilinear4sgc.C	 
    energy4c.C		 
    gradientsc.C		 
    ilanisocurvc.C		 	   
    innerloop-ani-sgstr-vcc.C
    randomfield3dc.C	 
    rhs4curvilinearc.C	 
    rhs4curvilinearsgc.C	 
    rhs4th3fortc.C		 	   
    scalar_prodc.C		 	   
    solerr3c.C		 
    testsrcc.C		 
    tw_ani_stiffc.C		 	   
    tw_aniso_force_ttc.C	 
    tw_aniso_forcec.C	 
    twilightfortc.C		 	   
    twilightsgfortc.C	 
    updatememvarc.C		 
    velsumc.C)

set(FORTRAN_COMPUTE_FILES
    addsg4wind.f
    addsgd.f
    anisomtrltocurvilinear.f
    bcfort.f
    bcfortanisg.f
    bcfreesurfcurvani.f
    bndryOpNoGhost.f
    boundaryOp.f
    checkanisomtrl.f
    computedtaniso.f
    curvilinear4sg.f
    energy4.f
    gradients.f
    ilanisocurv.f
    innerloop-ani-sgstr-vc.f
    randomfield3d.f
    rhs4curvilinear.f
    rhs4curvilinearsg.f
    rhs4th3fort.f
    scalar_prod.f
    solerr3.f
    testsrc.f
    tw_ani_stiff.f
    tw_aniso_force.f
    tw_aniso_force_tt.f
    twilightfort.f
    twilightsgfort.f
    updatememvar.f
    velsum.f)

blt_add_library(
  NAME sw4_core
  SOURCES ${CXX_BASE_FILES}
  DEPENDS_ON sw4_meta_cxx
)

blt_add_library(
  NAME sw4_core_fortran
  SOURCES ${FORTRAN_COMPUTE_FILES} ${FORTRAN_BASE_FILES}
  DEPENDS_ON sw4_meta
)

blt_add_library(
  NAME sw4_kernels
  SOURCES ${CXX_COMPUTE_FILES}
  DEPENDS_ON sw4_meta_cxx cuda
)

blt_add_executable(
  NAME sw4
  DEPENDS_ON sw4_core sw4_core_fortran sw4_kernels
)
