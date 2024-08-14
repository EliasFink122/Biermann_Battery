c   Description of the Pollux input data
c
c   Pollux runs on a two dimensional mesh, usually in R - Z geometry
c
c   re - /--------------------------------------------\
c        |          ++                                |
c        |          ++                                | <--
c        |          ++                                |
c        |          ++                                | <-- Laser
c        |          ++                                |
c        |          ++                                | <--
c        |          ++                                |
c    0 - \--------------------------------------------/
c
c        |          |                                 |
c       zb       solid foil                           zf
c
c   The mesh is cartesian and equally spaced
c
c   The pollux input data is formatted so it is easiest to edit an existing
c   data set.  The variables on each line ar as follows:
c   
c   Line 1: (a80) mjltext
c   Line 2: (2L7)  Class, Pond
c   Line 3: (4e11.3) zb, zf, zb0, zf0
c   Line 4: (2e11.3) re, re0
c   Line 5: (6i6) it, jt, in0, in1, jn, idim
c   Line 6: (5e11.3) rho, eint, f, z, gamma
c   Line 7: (6e11.3) dt, totime, tout, anomr, anomt, rhocut
c   Line 8: (5e11.3) power, freql, trise, cut, focus
c   Line 9: (4e11.3) ref, step, acc, tccut
c   Line10: (2i6) n, nstop
c
c   The variables have the following meanings:
c
c   mjltext:  Text comment describing the run
c   Class:    Logical switch for classical(true) or anomalous (false)
c             plasma coefficients
c   Pond:     Logical switch to include a crude desription of the 
c             ponderomotive force
c   zb:       (cm)  Z coordinate of the left hand edge of the mesh
c   zf:       (cm)  Z coordinate of the right hand side of the mesh
c   zb0:      (cm)  Z coordinate at which rezoning the left hand edge
c             will stop ( set zbo > zb to stop rezoning)
c   zf0:      (cm)  Z coordinate of the right hand edge of the mesh
c             at which rezoning will stop (set zf0 < zf to inhibit)
c   re:       (cm)  R coordinate of the free edge of the mesh
c   re0:      (cm)  R coordinate at which rezoning in R will stop
c             (set re0 < re to inhibit R rezoning)
c   it:       Number of cells in Z direction.  Must be at least 2 less
c             than the array dimensions to allow for edge cells
c   jt:       Number of cells in the R direction.  Must be at least 1
c             less than the array dimension to allow for edge cells
c   in0:      Number of cells between the left edge of the mesh and the
c             'back' of the foil target
c   in1:      Number of cells in Z at solid density in the target
c   jn:       Number of cells in R at solid density in the target
c   idim:     0 for X-Y geometry or 1 for R-Z geometry
c   rho:      (gcm^-3) Initial solid density
c   eint1     (K) Initial temperature
c   F:        Average Mass number of the target material
c   Z:        Average atomic number of the target material
c   Gamma:    Specific heat ration for the ion equation of state
c   dt:       (s) Initial time step - will be determined internally after
c             15 time steps
c   totime:   (s) Simulation time at which the run will stop
c   tout:     (s) Time interval between printouts
c   anomr:    Anomalous ion collision coefficient, if Class = False
c   anomt:    Anomalous electron collision coefficient, if Class = False
c   rhocut:   Multiple of critical density below which the thermal
c             conduction constraint on time step will be ignored
c   power:    (erg cm^-2 s^-1) peak laser power
c   freql:    (micron) Laser wavelength
c   trise:    (s) Laser pulse risetime
c   cut:      Multiple of trise, after which laser power is set to zero
c   focus:    (cm) laser spot size
c   ref:      Artificial reflectivity at critical density (1 - ref)
c             of the power reaching critical will be absorbed
c   step:     Density ratio applied to the ramp at front and back
c             of the foil target
c   acc:      Accuracy condition used in the energy constraint on time
c             step.  Do not normally need to change
c   tccut:    Multiple of critical density below which the advection
c             equations are approximated
c   n:        Initial time step number, should only be non zero for a restart
c             (not currently implemented)
c   nstop:    Time step number at which the run will stop
c             (the run will stop at whichever of the nstop and tottime
c             conditions occurs first)
c
      PROGRAM MAIN
      include 'polx.i'
      LOGICAL ODD,CLASS,POND
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /BKODD/ ODD,CLASS,POND
      COMMON /BKSTOP/ NSTOP,NOUT
      OPEN(5, FILE='pchris.dat', STATUS='OLD')
C     OPEN(2, FILE='pchris.dat', STATUS='OLD')
      CALL INPUT
      CLOSE(5)
C     CLOSE(2)
      CALL SET
      IF (N.NE.0) CALL INPUT1
      CALL STATE
    1 N=N+1
      IF(TIME.GE.TOTOUT) CALL OUTPUT
      IF (N/5*5.EQ.N.AND.N.GT.10) CALL TIMING
      CALL KNETIC
      CALL ABSORB
      call radloss
      CALL ACCLN
      CALL ENERGY
      CALL EQUIL
      CALL ADVECT
      CALL STATE
      CALL ERRORS
      TIME=TIME+DT
      ODD=.NOT.ODD
      IF (N.EQ.NSTOP) then
        call output
        stop
      endif
      if (mod(n,10) .eq. 0) call rgeout
      IF (TIME.LT.TOTIME) GO TO 1
      END
      SUBROUTINE INPUT
      include 'polx.i'
      LOGICAL ODD,CLASS,POND
      CHARACTER*80 MJLTEXT
      COMMON /TEXT/ MJLTEXT
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /TFC /cmult,eemin,nstate
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /ENER/ EINT1(nnz,nnr),EINT2(nnz,nnr)
      COMMON /LASAR/ FOCUS,POWER,TRISE,CUT
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /BK1/ F,Z,FREQL,TCCUT,ANOMR,ANOMT,RHOCUT
      COMMON /BK2/ IDIM,IN0,IN1,JN,STEP
      COMMON /BKABS/ REF,ROCRIT
      COMMON /BKACC1/ ACC,ACC1
      COMMON /BKRG/ RG(2),GAMMA(2)
      COMMON /BKODD/ ODD,CLASS,POND
      COMMON /BKSTEP/ RHO0,RHOLT
      COMMON /BKSTOP/ NSTOP,NOUT
C     LEVEL2,RHO,RHO1,EINT1,EINT2
C****
      READ (5,4322) MJLTEXT
4322  FORMAT (A80)
      READ (5,1) CLASS,POND
    1 FORMAT (2L7)
      READ (5,2) ZB,ZF,ZB0,ZF0
    2 FORMAT (4E11.3)
      READ (5,3) RE,RE0
    3 FORMAT (2E11.3)
      READ (5,4) IT,JT,IN0,IN1,JN,IDIM
    4 FORMAT (6I6)
      READ (5,5) RHO(1,1),EINT1(1,1),F,Z,GAMMA(1)
    5 FORMAT (5E11.3)
      READ (5,6) DT,TOTIME,TOUT,ANOMR,ANOMT,RHOCUT
    6 FORMAT (6E11.3)
      READ (5,5) POWER,FREQL,TRISE,CUT,FOCUS
      READ (5,2) REF,STEP,ACC,TCCUT
      READ (5,7) N,NSTOP
    7 FORMAT (2I6)
      READ(5,1234)NOUT,NSTATE
 1234 FORMAT(2I3)
      IF (N.NE.0) RETURN
      WRITE (6,11) FREQL,POWER,FOCUS,TRISE,CUT,REF,TCCUT
   11 FORMAT (1H1,4X,'LASER WAVELENGTH',E11.3,4X,'LASER POWER ON AXIS',
     1E11.3,4X,'FOCAL RADIUS',E11.3/4X,'LASER RISE TIME',E11.3,4X,
     2'TOTAL DURATION OF LASER PULSE',E11.3,4X,'REFLECTION COEFFT.',
     3E11.3/4X,'CUT OFF FACTOR',E11.3)
      WRITE (6,12) RHO(1,1),EINT1(1,1),F,Z,GAMMA(1),RHOCUT
   12 FORMAT (1H0,4X,'SOLID DENSITY',E11.3,4X,'INITIAL TEMPERATURE',
     1E11.3/4X,'MASS NUMBER',F11.2,4X,'IONIC CHARGE',F11.2,4X,
     2'ION GAMMA',E11.3/4X,'DENSITY LIMIT FACTOR',E11.3)
      WRITE (6,13) DT,TOTIME,TOUT,ZB,ZF,RE,ZB0,ZF0,RE0,IT,JT
   13 FORMAT(1H0,4X,'INITIAL TIME STEP',E11.3,4X,'TOTAL RUN TIME',E11.3,
     14X,'OUTPUT INTERVAL',E11.3/4X,'MESH SIZE :',4X,'BACK',E11.3,4X,
     2'FRONT',E11.3,4X,'SIDE',E11.3/4X,'MESH LIMIT :',4X,'BACK',E11.3,
     24X,'FRONT',E11.3,4X,'SIDE',E11.3/4X,'NO OF CELLS :',4X,
     3'Z DIRN.',I4,4X,'R DIRN',I4)
      WRITE (6,14) ANOMR,ANOMT
   14 FORMAT (1H0,4X,'ANOMALOUS DIFFUSION FACTORS:',4X,'IONIC',E11.3,
     14X,'ELECTRONIC',E11.3)
      WRITE (6,15) IN1,JN,IN0,IDIM,STEP,ACC
   15 FORMAT (1H0,4X,'NO OF SOLID CELLS',4X,'Z DIRN :',I6,4X,'R DIRN :',
     1I6,4X,'NO OF STEP CELLS AT BACK :',I6/4X,'DIMENSION FACTOR :',I6,
     24X,'DENSITY STEP FACTOR',E11.3,4X,'ACCURACY FACTOR',E11.3)
      IF (CLASS) GO TO 30
      WRITE (6,32)
   32 FORMAT (4X,'ION-ACOUSTIC INSTABILITY FLUX LIMITED')
      GO TO 34
   30 WRITE (6,31) ANOMT
   31 FORMAT (4X,'CLASSICAL FLUX LIMIT WITH FACTOR',2X,E10.2)
   34 CONTINUE
      IF (POND) WRITE (6,35)
   35 FORMAT (4X,'PONDEROMOTIVE FORCE INCLUDED')
      WRITE (6,22)
   22 FORMAT(4X,'CLASSICAL DIFFUSION')
      IF(NSTATE.GE.1)WRITE(6,2345)
 2345 FORMAT(//,10X,'CORRECTED THOMAS-FERMI-KIRZHNITS E.O.S. INCLUDED')
      RETURN
      END
      SUBROUTINE INPUT1
      include 'polx.i'
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /ENER/ E1(nnz,nnr),E2(nnz,nnr)
      COMMON /ERR/ ERROR(4)
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /VEL/ U(nnz,nnr),U1(nnz,nnr),V(nnz,nnr),V1(nnz,nnr)
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /TOT/ TOTAL(3,4)
C     LEVEL2,RHO,RHO1,E1,E2,U,U1,V,V1
      READ (5,1) DT,TIME,ZB,ZF,RE
      CALL CORD
      READ (5,1) ((RHO(I,J),E1(I,J),E2(I,J),U(I,J),V(I,J),
     1J=1,JT),I=2,IT1)
      READ (5,1) TOTAL(1,1),TOTAL(2,1),TOTAL(3,1),TOTAL(3,2),ERROR(4)
    1 FORMAT (5E16.8)
      TOTOUT=TIME+TOUT-AMOD(TIME,TOUT)
      CALL RESET1 (RHO)
      CALL RESET1(E1)
      CALL RESET1(E2)
      CALL RESET(U)
      CALL RESET(V)
      RETURN
      END
      FUNCTION ZELOSS(ZRHO,ZTE,ZTI,ZDZ,ZE)
      include 'polx.i'
      DELTAE=5.0E15*ZRHO*ZDZ*(1.0-ZTE/5.0E7)/(ZE+1.0)
      IF (DELTAE.GT.ZE) DELTAE=ZE
      ZELOSS=ZE-DELTAE
      RETURN
      END
c
      subroutine radloss
      include 'polx.i'
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /ENER/ EINT1(nnz,nnr),EINT2(nnz,nnr)
      COMMON /LASAR/ FOCUS,POWER,TRISE,CUT
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /AREA/ SR(nnr),SR1(nnr),SZ(nnr),VOL(nnr)
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /BK1/ F,Z,FREQL,TCCUT,ANOMR,ANOMT,RHOCUT
      COMMON /TEMP/ T(nnz,nnr,2)
      COMMON /ABS1/ XI(nnz,nnr),W(nnz,nnr)
      common /rloss/ wrad(nnz,nnr), totrad
c   calculate an energy radiated per cell per timestep based on
c   Summers and McWhirter volume rates with a Black Body limit and
c   an internal energy limit
      do i=1, it
        do j=1, jt
c
      zsm = 1.0e+17/t(i,j,2)
      zpow = zsm * z * rho(i,j)*rho(i,j)*vol(j)/(f*f*3.0e-17)
      zbb = 5.7e-5 *t(i,j,2)*t(i,j,2)*t(i,j,2)*t(i,j,2)*(sr(j)+sz(j))
      if (zpow .gt. zbb) zpow = zbb
      zlimit = 0.1*eint2(i,j)/dt
      if (zpow .gt. zlimit) zpow = zlimit
      wrad(i,j) = zpow * dt
      if (t(i,j,2) .lt. 2.0e5) wrad(i,j) = 0.0
c      
        enddo
      enddo
      end
C
      SUBROUTINE OUTPUT
      include 'polx.i'
      CHARACTER*80 MJLTEXT
      COMMON /LASAR/FOCUS,POWER,TRISE,CUT
      COMMON /TEXT/ MJLTEXT
      COMMON /BKSTOP/ NSTOP,NOUT
      COMMON /PRES/ P(nnz,nnr),PF(nnz,nnr)
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /ENER/ EINTI(nnz,nnr),EINT(nnz,nnr)
      COMMON/BKABS/REF,ROCRIT
      COMMON /ERR/ ERROR(4)
      COMMON /INCRE/ DTR,DTZ,DTTR,DTTZ,DTW,DTCFL,DTERR
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /SCALE/ RHOSC,TSCALE,USCALE,VSCALE,BSCALE
      COMMON /TEMP/ TI(nnz,nnr),TE(nnz,nnr)
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /VEL/ U(nnz,nnr),U1(nnz,nnr),V(nnz,nnr),V1(nnz,nnr)
      COMMON /ABS1/ XI(nnz,nnr),W(nnz,nnr)
      common /rloss/ wrad(nnz,nnr), totrad
      DIMENSION EALPHA(nnr)
      DIMENSION UNUSED(nnz,nnr),HEIGTS(16)
C     LEVEL2,RHO,RHO1,EINTI,EINT,TI,TE,U,U1,V,V1
C
      WRITE (6,10) N,TIME,(ERROR(I),I=1,3)
   10 FORMAT(1H1,6X,'ITERATION NO. ',I6,2X,'TIME  ',E10.2/
     16X,'ERRORS :',4X,'U VEL :',E10.2,6X,'V VEL :',E10.2,6X,
     2'ENERGY :',E10.2)
      WRITE (6,16) DT,DTR,DTZ,DTTR,DTTZ,DTW,DTCFL,DTERR
   16 FORMAT (1H0,3X,'NEW TIME STEP',2X,E11.3/4X,'CELL EMPTY R',2X,
     1E11.3,4X,'CELL EMPTY Z',2X,E11.3,4X,'THERM COND R',2X,
     2E11.3,4X,'THERM COND Z',2X,E11.3/4X,
     3'ABS ENERGY',4X,E11.3,4X,/'CFL CONDITION',1X,E11.3,4X,
     4'ACCURACY COND.',E11.3)
      WRITE (6,18) ZB,ZF,RE
   18 FORMAT (1H0,'MESH BOUNDARIES :  BACK',1PE11.3,4X,'FRONT',1PE11.3,
     14X,'SIDE',1PE11.3)
C
      IF(NOUT.NE.4)GO TO 33
      DO 4 J1=1,JT,2
      J2=J1+1
      IF (J2.LE.JT) GO TO 3
      J2=JT
      IF (J1.NE.J2) GO TO 3
      WRITE (NOUT,17) J1
   17 FORMAT (1H1,10HR CELL NO.,I6/11H Z CELL NO.,2X,5HZ VEL,4X,5HR VEL,
     13X,7HDENSITY,3X,8HION TEMP,2X,7HEL TEMP)
      GO TO 2
    3 WRITE (NOUT,15) J1,J2
   15 FORMAT (1H1,10HR CELL NO.,I6,T66,10HR CELL NO.,I6/11H Z CELL NO.,
     12(2X,5HZ VEL,4X,5HR VEL,3X,7HDENSITY,3X,8HION TEMP,
     22X,7HEL TEMP,1X))
    2 DO 4 I=2,IT1
      WRITE (NOUT,14) I,(U(I,J),V(I,J),RHO(I,J),TI(I,J),TE(I,J),
     1J=J1,J2)
   14 FORMAT (3X,I4,3X,1P10E9.2)
    4 CONTINUE
   33 CONTINUE
      TOTOUT=TOTOUT+TOUT
C
      PTOT1=0.0
      PTOT2=0.0
      TEMAX=0.0
      PMAX =0.0
      WRITE(6,6000)
 6000 FORMAT(//,10X,'DATA SUMMARY',//
     +  10X,'CELL NO',7X,'PK TEMP',7X,'PK PRESSURE',
     +  3X,'+VE MOMENTUM',2X,'-VE MOMENTUM',//)
 
      DO 100 J=2,JT
      PTOT1=0.0
      PTOT2=0.0
      PMAX=0.0
      TEMAX=0.0
      DM1=0.0
      DM2=0.0
      DO 90  I=2,IT
      IF(U(I,J).GE.0.0) GO TO 80
      PTOT2=PTOT2 - RHO(I,J)*U(I,J)
      DM2=DM2+RHO(I,J)
      GO TO 85
   80 PTOT1=PTOT1 + RHO(I,J)*U(I,J)
      DM1=DM1+RHO(I,J)
   85 CONTINUE
      IF(TE(I,J).GT.TEMAX) TEMAX=TE(I,J)
      IF( P(I,J).GT. PMAX)  PMAX= P(I,J)
   90 CONTINUE
      IF(DM1.NE.0.0)PTOT1=PTOT1/DM1
      IF(DM2.NE.0.0)PTOT2=PTOT2/DM2
      WRITE(6,6001)J,TEMAX,PMAX,PTOT1,PTOT2,DM1,DM2
 6001 FORMAT(9X,I4,7X,1P6E14.4)
C
  100 CONTINUE
C
      JW=JT/2
      WRITE(6,6100)JW
 6100 FORMAT(4X,'VARIABLE LISTING ALONG ROW J =',I6)
      DO 110 I=2,IT1
      WRITE(6,6101)I,RHO(I,JW),TE(I,JW),EINT(I,JW),w(i,jw),wrad(i,jw)
  110 CONTINUE
 6101 FORMAT(I4,1P5E10.1)
C
      DO 210 J=1,5
      HEIGTS(3*J-2) = 1.0*10.0**(J+3)
      HEIGTS(3*J-1) = 2.0*10.0**(J+3)
      HEIGTS(3*J  ) = 5.0*10.0**(J+3)
  210 CONTINUE
      HEIGTS(16)    = 1.0E9
c     CALL HDR(HEIGTS,16,'TE              ')
c     CALL PCONT(TE   ,nnz,IT,JT,HEIGTS,16)
c     CALL HDR(HEIGTS,16,'U VELOCITY      ')
c     CALL PCONT(U,nnz,IT,JT,HEIGTS,16)
c     CALL HDR(HEIGTS,16,'V VELOCITY      ')
c     CALL PCONT(V,nnz,IT,JT,HEIGTS,16)
      DO 220 J=1,16
      HEIGTS(J)=HEIGTS(J)*1.0E-7
  220 CONTINUE
      HEIGTS(16)=ROCRIT
c     CALL HDR(HEIGTS,16,'RHO             ')
c     CALL PCONT(RHO  ,nnz,IT,JT,HEIGTS,16)
c
c  put the time step somewhere visible
c
      write(6,6123) n, time
 6123 format (1x, 'time step = ', i5, 'time = ', 1pe10.3)
      write(6,6124) totrad
 6124 format('Total energy radiated = ', 1pe12.2)
c
      RETURN
      END
C
      SUBROUTINE RGEOUT
      include 'polx.i'
      CHARACTER*1 CODE(nnz), CC(52)
      DATA CC/'A','B','C','D','E','F','G','H',
     :        'I','J','K','L','M','N','O','P',
     :        'Q','R','S','T','U','V','W','X',
     :        'Y','Z','a','b','c','d','e','f',
     :        'g','h','i','j','k','l','m','n',
     :        'o','p','q','r','s','t','u','v',
     :        'w','x','y','z' /
      COMMON /DENS/ RHO(nnz,nnr), RHO1(nnz,nnr)
      COMMON /CELL/ JTM1, JT, JT1, IT, IT1, IT2, KN, KN1
      COMMON /TEMP/ TI(nnz,nnr), TE(nnz,nnr)
      common /vel/  u(nnz,nnr), u1(nnz,nnr), v(nnz,nnr), v1(nnz,nnr)
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /BKABS/ REF,ROCRIT
      write(11, '(1pe10.3, i5, i5)') time, it2-1, jt1
      write(12, '(1pe10.3, i5, i5)') time, it2-1, jt1
      write(13, '(1pe10.3, i5, i5)') time, it2-1, jt1
      DO 10 I=1,80
   10 CODE(I) = 'Z'
      DO 100 I=2,IT2
c   write Rho to fort.11
      DO 90  J=1,JT1
      K = 50 + (10.0*ALOG10(RHO(I,J) + 1.0e-12))
      IF (K.GT.52)K=52
      IF (K.LT.1) K=1
      code(j)=cc(k)
c  mark rocrit on the laser side only
      if (rho(i,j).ge.rocrit .and. rho(i+1,j).lt.rocrit) code(j)='z'
   90 continue
      WRITE(11,6000)(code(j), j=1, jt1)
c   write Te to fort.12
      DO 95 J=1,JT1
      K = (15.0*ALOG10(TE(I,J)+0.1)) - 75
      IF (K.GT.52) K=52
      IF (K.LT.1 ) K=1
      CODE(J) = CC(K)
   95 CONTINUE
      WRITE(12,6000)(code(j), j=1, jt1)
c   write u and v to fort.13
      do 97 j=1, jt1
      k = 5.0 * alog10(amax1(abs(u(i,j)), 1.0e4)) - 20.0
      if (u(i,j) > 0.0) then
        k = 26 + k
      else
        k = 26 - k
      endif
      if (k .gt. 52) k = 52
      if (k .lt.  1) k = 1
      code(j) = cc(k)
   97 continue
      write(13, 6000) (code(j), j=1, jt1)
      do 98 j=1, jt1
      k = 5.0 * alog10(amax1(abs(v(i,j)), 1.0e4)) - 20.0
      if (v(i,j) > 0.0) then
        k = 26 + k
      else
        k = 26 - k
      endif
      if (k .gt. 52) k = 52
      if (k .lt.  1) k = 1
      code(j) = cc(k)
   98 continue
      write(13, 6000) (code(j), j=1, jt1)
 6000 FORMAT(1X,300A1)
  100 CONTINUE
      RETURN
      END
C
      SUBROUTINE PCONT(X,IDIM,IT,JT,HEIGHT,NC)
      include 'polx.i'
      DIMENSION X(IDIM,JT),HEIGHT(1)
      CHARACTER*1 CHAR(nnz),ICHAR(18)
      DATA ICHAR/'1','2','3','4','5','6','7','8','9','A','B','C',
     +           'D','E','F','*',' ','-'/
      CALL RESETC(CHAR,ICHAR(17),nnz)
      CALL RESETC(CHAR,ICHAR(18),IT)
      CHAR(IT+1)='|'
      WRITE(6,6000)(CHAR(i), i=1, it)
 6000 FORMAT(1X,'|',120A1)
      DO 100 JJ=1,JT
      J=JT+1-JJ
      CALL RESETC(CHAR,ICHAR(17),IT)
      DO 90 I=2,IT
      DO 80 N=1,NC
      TEST=(ABS(X(I-1,J))-HEIGHT(N))*(ABS(X(I,J))-HEIGHT(N))
C
   78 IF(TEST.GT.0.0) GO TO 80
      CHAR(I)=ICHAR(N)
   80 CONTINUE
   90 CONTINUE
      char(it+1) = '|'
      WRITE(6,6000)(CHAR(i), i=1, it+1)
  100 CONTINUE
      CALL RESETC(CHAR,ICHAR(18),IT+1)
      WRITE(6,6000)(CHAR(i), i=1, it+1)
      RETURN
      END
      SUBROUTINE RESETC(CHAR,ICHAR,N)
      include 'polx.i'
      CHARACTER*1 CHAR(1),ICHAR
      DO 10 I=1,N
      CHAR(I)=ICHAR
   10 CONTINUE
      RETURN
      END
      SUBROUTINE HDR(HEIGHT,NC,LABEL)
      include 'polx.i'
      character*16 label
      DIMENSION ILAB(16),HEIGHT(1)
      WRITE(6,6000)LABEL
 6000 FORMAT(1H1,1X,'CONTOUR PLOT OF ',A16)
      DO 10 I=1,NC
      ILAB(I)=I
   10 CONTINUE
      WRITE(6,6001)(ILAB(I),I=1,NC)
 6001 FORMAT(1X,'CONTOUR NO.'/,(8I12))
      WRITE(6,6002)(HEIGHT(I),I=1,NC)
 6002 FORMAT(1X,'CONTOUR HEIGHT',/,(1P8E12.3))
      RETURN
      END
      SUBROUTINE CORD
      include 'polx.i'
      COMMON /AREA/ SR(nnr),SR1(nnr),SZ(nnr),VOL(nnr)
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /BK2/ IDIM,IN0,IN1,JN,STEP
      COMMON /BKSTEP/ RHO0,RHOLT
      DR=RE/FLOAT(JT)
      DZ=(ZF-ZB)/FLOAT(IT)
      IF (IDIM.EQ.0) GO TO 1
      DO 10 J=1,JT1
      SZ(J)=6.2832*(FLOAT(J)-0.5)*DR**2
      SR(J)=6.2832*(FLOAT(J)-1.0)*DZ*DR
      SR1(J)=6.2832*(FLOAT(J)-0.5)*DZ*DR
   10 VOL(J)=SZ(J)*DZ
      RETURN
    1 DO 11 J=1,JT1
      SZ(J)=DR
      SR1(J)=DZ
      SR(J)=DZ
   11 VOL(J)=DR*DZ
      RETURN
      END
      SUBROUTINE SET
      include 'polx.i'
      COMMON /AREA/ SR(nnr),SR1(nnr),SZ(nnr),VOL(nnr)
      COMMON /TFC / cmult,eemin,nstate
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /DRIVE/ W1,T1,T2,SPIT1,SPIT2,DELTA0,GAMA01,
     1RECNE,REST,CCFL,CC25(2),T3,RTE,ACOUST,XI1,THERM
      COMMON /ENER/ EINT1(nnz,nnr),EINT2(nnz,nnr)
      COMMON /ERR/ ERROR(4)
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /SCALE/ RHOSC,TSCALE,USCALE,VSCALE,BSCALE
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /TOT/ TOTAL(3,4)
      COMMON /BK1/ F,Z,FREQL,TCCUT,ANOMR,ANOMT,RHOCUT
      COMMON /BK2/ IDIM,IN0,IN1,JN,STEP
      COMMON /BKABS/ REF,ROCRIT
      COMMON /BKACC1/ACC,ACC1
      COMMON /BKRG/ RG(2),GAMMA1(2)
      COMMON /BKSTEP/ RHO0,RHOLT
      common /beam/ bprof(nnr)
      common /rloss/ wrad(nnz,nnr), totrad
C     LEVEL2,RHO,RHO1,EINT1,EINT2
      DIMENSION RO1(nnr)
      JTM1=JT-1
      JT1=JT+1
      IT1=IT+1
      IT2=IT+2
      KN=JT*IT
      KN1=JT1*IT2
      CALL CORD
      ACC1=1.0E-10*ACC
      GAMMA1(1)=GAMMA1(1)-1.0
      GAMMA1(2)=0.666667
      RG(1)=8.32E7/(F*GAMMA1(1))
      RG(2)=1.248E8*Z/F
      RTE=RG(1)+RG(2)
      RTE=RG(2)/RTE
      REF=1.0-REF
      FREQL=3.0E14/FREQL
      XI1=F*FREQL*FREQL/Z
      ROCRIT=1.965E-32*XI1
      XI1=4.845E21/XI1
      RHOLT=1.0E-6*ROCRIT
      RHO0=RHO(1,1)
      I0=IN0+2
      IN1=I0+IN1-1
      DO 1 I=I0,IN1
    1 RHO(I,1)=RHO0
      IF (IN0.EQ.0) GO TO 20
      IN0=IN0+1
      DO 2 I=1,IN0
      I1=I0-1
      RHO(I1,1)=STEP*RHO(I0,1)
    2 I0=I1
   20 IF (IN1.EQ.IT) GO TO 30
      DO 3 I=IN1,IT
      I1=I+1
    3 RHO(I1,1)=STEP*RHO(I,1)
   30 JN1=JN+1
      DO 4 I=2,IT1
      DO 5 J=2,JN
    5 RHO(I,J)=RHO(I,1)
      IF (JN.EQ.JT) GO TO 4
      J0=JN
      DO 6 J=JN1,JT
      RHO(I,J)=STEP*RHO(I,J0)
    6 J0=J
    4 CONTINUE
c
      DO 7 I=2,IT1
      DO 7 J=1,JT
      IF (RHO(I,J).LT.RHOLT) RHO(I,J)=RHOLT
    7 CONTINUE
      RHOLT=TCCUT*ROCRIT
C
C   DEFINE THE VARIABLES FOR TFC E.O.S.
C
C     ZT=EINT1(1,1)
C     ZV=1.6725E-30*F/RHO0
C     TFP1=10.0*TFP(ZT,ZV,Z,0.0)
C     TFP2=10.0*TFP(ZT,ZV,Z,1.0)
C     PION=RHO0*ZT*RG(1)*GAMMA1(1)
C     TFP1=TFP1+PION
C     TFP2=TFP2+PION
C     IF(ABS(TFP1-TFP2).GT.1.0)
C    ;  CMULT=(TFP1-1.0E9)/(TFP1-TFP2)
C     PINIT=PION+10.0*TFP(ZT,ZV,Z,CMULT)
C     WRITE(6,6000)ZT,ZV,TFP1,TFP2,PION,CMULT,PINIT
C6000 FORMAT(10X,1P7E12.3)
C
      EINT2(1,1)=RG(2)*EINT1(1,1)
      EEMIN=3.0*RG(2)*EINT1(1,1)
      EINT1(1,1)=RG(1)*EINT1(1,1)
      TOTAL(3,1)=0.0
      DO 8 J=1,JT
      DO 8 I=2,IT1
      EINT1(I,J)=EINT1(1,1)
      EINT2(I,J)=EINT2(1,1)
      ZV=1.6725E-30*F/RHO(I,J)
C     IF(NSTATE.GE.1)EINT2(I,J)=10.*TFU(ZT,ZV,Z,CMULT)/RHO(I,J)
C
      TOTAL(3,1)=TOTAL(3,1)+(EINT1(I,J)+EINT2(I,J))*RHO(I,J)*VOL(J)
    8 CONTINUE
      CALL RESET1(RHO)
      CALL RESET1(EINT1)
      CALL RESET1(EINT2)
      I=IFIX(Z+0.5)
      IF (I.GT.4) GO TO 15
      GO TO (11,12,13,14),I
   11 GAMA01=11.92
      DELTA0=3.7703
      GO TO 10
   12 GAMA01=5.118
      DELTA0=1.0465
      GO TO 10
   13 GAMA01=3.525
      DELTA0=0.5814
      GO TO 10
   14 GAMA01=2.841
      DELTA0=0.4106
      GO TO 10
   15 GAMA01=1.20
      DELTA0=0.0961
   10 CONTINUE
      THERM=6.213E5*GAMA01/DELTA0
      W1=Z/F
      SPIT1=1.15*ALOG10(W1)
      SPIT2=-13.3-SPIT1
      SPIT1=-15.0-SPIT1
      CC25(2)=1.260E19*W1/RG(2)
      REST=5.953E-12/W1
      RECNE=8.635E-6/W1
      T2=SQRT(F)
      ACOUST=0.005/T2
      CC25(1)=4.544E-16/T2
      CCFL=CC25(1)*F*T2/(5.484E11*ANOMR*(GAMMA1(1)+1.0))
      T1=8.244E-26/(Z*W1)
      T2=2.818E-30*W1*T2/Z**3
      T3=1836.12*F/(3.0*Z*GAMMA1(1)+2.0)
      ANOMT=0.5/ANOMT
      W1=0.96*W1
      RHOSC=1.1*RHO0
      TSCALE=1.0E2
      USCALE=1.0E6
      VSCALE=1.0E6
      BSCALE=1.0E3
      RHO0=RHOCUT*ROCRIT
c   define the laser beam spatial profile here
      do j=1, jt
c        bprof(j) = 0.5 + rand()
c        bprof(j) = 1.0 * exp(-((j-jt/2)/(jt/4))**4)
         bprof(j) = 1.0 * exp(-REAL((j-jt/2)/(jt/4))**4)
      enddo
c  zero the radiated energy
      totrad = 0.0
      RETURN
      END
      SUBROUTINE OUTPT1
      include 'polx.i'
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /ENER/ E1(nnz,nnr),E2(nnz,nnr)
      COMMON /ERR/ ERROR(4)
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /VEL/ U(nnz,nnr),U1(nnz,nnr),V(nnz,nnr),V1(nnz,nnr)
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /TOT/ TOTAL(3,4)
C     LEVEL2,RHO,RHO1,E1,E2,U,U1,V,V1
      WRITE (7,1) DT,TIME,ZB,ZF,RE
      WRITE (7,1) ((RHO(I,J),E1(I,J),E2(I,J),U(I,J),V(I,J),
     1J=1,JT),I=2,IT1)
      WRITE (7,1) TOTAL(1,1),TOTAL(2,1),TOTAL(3,1),TOTAL(3,2),ERROR(4)
    1 FORMAT (5E16.8)
      RETURN
      END
      SUBROUTINE TIMING
      include 'polx.i'
C
C   THIS SUBROUTINE ADJUSTS THE TIME STEP
C
      LOGICAL ODD,CLASS,POND
      REAL LAMDA1
      REAL KAPPI0(nnz,nnr),KAPPA0(nnz,nnr)
      COMMON/TEMP/TI(nnz,nnr),TE(nnz,nnr)
      COMMON /ABS1/ WQ(nnz,nnr),W(nnz,nnr)
      COMMON /TFC / cmult,eemin,nstate
      COMMON /AREA/ SR(nnr),SR1(nnr),SZ(nnr),VOL(nnr)
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /COEFFT/ KAPPI0,KAPPA0,TAU(nnz,nnr)
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /DRIVE/ W1,T1,T2,SPIT1,SPIT2,DELTA0,
     1GAMA01,RECNE,REST,CCFL,CC25(2),T3,RTE,ACOUST,XI1,THERM
      COMMON /ENER/ EINT1(nnz,nnr),EINT2(nnz,nnr)
      COMMON /ERR/ ERROR(4)
      COMMON /INCRE/ DTR,DTZ,DTTR,DTTZ,DTW,DTCFL,DTERR
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /PRES/ P(nnz,nnr),PF(nnz,nnr)
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /VEL/ U(nnz,nnr),U1(nnz,nnr),V(nnz,nnr),V1(nnz,nnr)
      COMMON /BKACC1/ ACC,ACC1
      COMMON /BKODD/ ODD,CLASS,POND
      COMMON /BKRG/ RGI,RGE,GAMMAI,GAMMAE
      COMMON /BKSTEP/ RHO0,RHOLT
C     LEVEL2,WQ,W,KAPPI0,KAPPA0,
C    1TAU,RHO,RHO1,EINT1,EINT2,P,PF,U,U1,V,V1
      IF (DR.LT.DZ) GO TO 20
      LAMDA1=DZ*DZ/(1.0+GAMMAI)
      GO TO 30
   20 LAMDA1=DR*DR/(1.0+GAMMAI)
   30 DR1=0.2*DR
      DZ1=0.2*DZ
      DTTR=DT+DT
      DTTZ=DTTR
      DTW=DTTZ
      DTZ=DTW
      DTR=DTZ
      DTCFL=DTR*DTR
      IM1=2
      DO 1 I=2,IT1
      I1=I+1
      JM1=1
      DO 2 J=1,JT
      J1=J+1
C
C   CELL EMPTYING CONDITION IN R DIRECTION
C
      DT1=DR1/(ABS(V(I,J))+0.000001)
      IF (DT1.LT.DTR) DTR=DT1
C
C   CELL EMPTYING CONDITION IN Z DIRECTION
C
      DT1=DZ1/(ABS(U(I,J))+0.000001)
      IF (DT1.LT.DTZ) DTZ=DT1
C
C   ABSORBED ENERGY CONDITION
C
      IF (RHO(I,J).LT.1.0E-20) GO TO 2
      IF(ABS(W(I,J)).GT.1.0E-10)
     : DT1=0.5*DT*(TI(I,J)*RGI+TE(I,J)*RGE)
     1*RHO(I,J)*VOL(J)/ABS(W(I,J))
      IF (DT1.LT.DTW) DTW=DT1
      IF (RHO(I,J).LT.RHOLT) GO TO 13
      IF (I.EQ.IT1) GO TO 13
C
C   THERMAL CONDUCTION CONDITION IN Z DIRECTION
C
      DT1=ABS((KAPPA0(I1,J)+KAPPA0(I,J))*(TE(I1,J)*RGE-TE(I,J)*RGE)/DZ)
      IF (DT1.LT.1.0) GO TO 12
      DT1=0.25*(TE(I1,J)*RGE+TE(I,J)*RGE)/DT1
      IF (DT1.LT.DTTZ) DTTZ=DT1
   12 CONTINUE
C
C   THERMAL CONDUCTION CONDITION IN R DIRECTION
C
      IF (J.EQ.JT) GO TO 13
      DT1=ABS((KAPPA0(I,J1)+KAPPA0(I,J))*(TE(I,J1)*RGE-TE(I,J)*RGE)/DR)
      IF (DT1.LT.1.0) GO TO 13
      DT1=0.25*(TE(I,J1)*RGE+TE(I,J)*RGE)/DT1
      IF (DT1.LT.DTTR) DTTR=DT1
   13 CONTINUE
C
C   COURANT-LEVY-FRIEDRICHS CONDITION
C
      IF(P(I,J).LE.0.0) GO TO 2
      DT1=LAMDA1/(P(I,J)*RHO1(I,J))
    3 IF (DT1.LT.DTCFL) DTCFL=DT1
    2 JM1=J
    1 IM1=I
      DTCFL=SQRT(DTCFL)
      DTERR=DT*ACC/ERROR(4)
      DT=AMIN1(DTR,DTZ,DTW,DTTR,DTTZ,DTCFL)
      IF(DT.GT.5.0E-14)RETURN
C     WRITE(6,6000)DTR,DTZ,DTW,DTTR,DTTZ,DTCFL
 6000 FORMAT(10X,'WARNING TIME STEP LESS THAN 5.0E-14'
     :,/,10X,1P6E13.4,/,10X,'MINIMUM TIME STEP USED')
      DT=5.0E-14
      RETURN
      END
      SUBROUTINE ABSORB
      include 'polx.i'
C
C   THIS SUBROUTINE CALCULATES THE ABSORBED LASER ENERGY
C
      LOGICAL ODD,CLASS,POND
      DIMENSION TAU(nnr)
      COMMON /ABCD/ A(nnrz),B(nnrz),C(nnrz),
     +D(nnrz),E(nnrz),F(nnrz),
     1G(nnrz)
      COMMON /ABS1/ XI(nnz,nnr),W(nnz,nnr)
      common /rloss/ wrad(nnz,nnr), totrad
      COMMON /AREA/ SR(nnr),SR1(nnr),SZ(nnr),VOL(nnr)
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /LASAR/ FOCUS,POWER,TRISE,CUT
      COMMON /LASIN/ ESUP,EREJEC,EABS
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /PRES/ P(nnz,nnr),PF(nnz,nnr)
      COMMON /TIM/totime,tout,totout,dt,time,n
      COMMON /TOT/ TOTAL(3,4)
      COMMON /VEL/ U(nnz,nnr),U1(nnz,nnr),V(nnz,nnr),V1(nnz,nnr)
      COMMON /BKABS/ REF,ROCRIT
      COMMON /BKODD/ ODD,CLASS,POND
      common /beam/ bprof(nnr)
      EQUIVALENCE (TAU(1),A(1))
C     LEVEL2,TAU,A,B,C,D,E,F,G,XI,W,RHO,RHO1,P,PF,U,U1,V,V1
      ESUP=0.0
      EREJEC=0.0
      DO 10 J=1,JT
      DO 10 I=2,IT2
      PF(I,J)=0.0
   10 W(I,J)=0.0
c
c   the temporal profile comes in here
c
      TIMER=TIME+0.5*DT
      TIMER=TIMER/TRISE
      IF (TIMER.GT.CUT) RETURN
      alasr1 = alaser(timer)*power*dt
c   the spatial profile is assumed set up in bprof(j)
      DO 7 J=1,JT
      CI=ALASR1*SZ(J)*bprof(j)
      ESUP=ESUP+CI
      I0=IT1
      I1=IT1
      WD1=0.0
      DO 1 I=2,IT1
      IF (CI.LE.0.0) GO TO 6
      I0=I0-1
      IF (RHO(I1,J).GT.ROCRIT) GO TO 2
      IF (XI(I1,J).LT.1.0E2) GO TO 11
      IF (POND) CI=CI*(1.0+WD1)
      W(I1,J)=W(I1,J)+CI
      CI=0.0
      GO TO 7
   11 TAU(I1)=EXP(-XI(I1,J))
      IF (POND) GO TO 12
      CI0=CI*TAU(I1)
      W(I1,J)=W(I1,J)+CI-CI0
      GO TO 13
   12 X=2.0*ROCRIT/(RHO(I0,J)+RHO(I1,J))
      IF (X.GT.0.8) X=0.8
      X=3.3333333E-11*(1.0-0.5*X)/SQRT(1.0-X)
      WD0=0.5*X*(U(I1,J)-U(I0,J))
      IF (WD0.GT.0.5) WD0=0.5
      CI0=CI*(1.0+WD1)/(1.0-WD0)*TAU(I1)
      W(I1,J)=W(I1,J)+(1.0+WD1)*(1.0-TAU(I1))*CI
      PF(I0,J)=X*CI0/DT
      WD1=WD0
   13 I1=I0
    1 CI=CI0
      GO TO 6
    2 CI0=REF*CI
      CI=CI-CI0
      IF (I1.EQ.IT1) GO TO 3
      I=I1+1
      DI=CI0*(RHO(I1,J)-ROCRIT)/(RHO(I1,J)-RHO(I,J))
      W(I,J)=W(I,J)+DI
      W(I1,J)=W(I1,J)+CI0-DI
      IF (CI) 7,7,4
    3 W(I1,J)=W(I1,J)+CI0
      GO TO 6
    4 WD1=0.0
      DO 5 I1=I,IT1
      IF (CI.LT.0.0) GO TO 6
      IF (POND) GO TO 50
      CI0=CI*TAU(I1)
      W(I1,J)=W(I1,J)+CI-CI0
      GO TO 5
   50 I0=I1+1
      X=2.0*ROCRIT/(RHO(I1,J)+RHO(I0,J))
      IF (X.GT.0.8) X=0.8
      X=3.333333E-11*(1.0-0.5*X)/SQRT(1.0-X)
      WD0=0.5*X*(U(I0,J)-U(I1,J))
      IF (WD0.GT.0.5) WD0=0.5
      CI0=CI*(1.0+WD1)/(1.0-WD0)
      W(I1,J)=W(I1,J)+(1.0+WD1)*(1.0-TAU(I1))*CI
      PF(I0,J)=PF(I0,J)+X*CI0/DT
      WD1=WD0
    5 CI=CI0
    6 EREJEC=EREJEC+CI
    7 CONTINUE
      EABS=ESUP-EREJEC
      TOTAL(3,4)=TOTAL(3,4)+EABS
c  put a radiative loss function in here
      zerad = 0.0
      do i=2, it2
        do j=1, jt1
          w(i,j) = w(i,j) - wrad(i,j)
          zerad = zerad + wrad(i,j)
        enddo
      enddo
      totrad = totrad + zerad
      total(3,4) = total(3,4) - zerad
c
      RETURN
      END
      FUNCTION ALASER(T)
      include 'polx.i'
C
C   THIS FUNCTION CALCULATES THE LASER PULSE SHAPE
C
      ALASER = 0.0
      if (t .gt. 0.0) go to 1
      return
    1 IF (T.GT.0.5) GO TO 2
      ALASER=1.21306*T
      RETURN
    2 IF (T.LE.1.0) GO TO 3
      SQ=(1.0-T)**2
      GO TO 4
    3 SQ=2.0*(1.0-T)**2
    4 ALASER=EXP(-SQ)
      RETURN
      END
      SUBROUTINE KNETIC
      include 'polx.i'
C
C   CALCULATION OF THE TRANSPORT COEFFICENTS
C
      REAL LAMBDA,LAMDA1
      REAL KAPPA0
      LOGICAL ODD,CLASS,POND
      COMMON /ABS1/ XI(nnz,nnr),WW(nnz,nnr)
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /COEFFT/ KAPPA0(nnz,nnr,2),TAU(nnz,nnr)
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /DRIVE/ W1,T1,T2,SPIT1,SPIT2,DELTA0,
     1GAMA01,RECNE,REST,CCFL,CC25(2),T3,RTE,ACOUST,XI1,THERM
      COMMON /ENER/ EINTI(nnz,nnr),EINTE(nnz,nnr)
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /TEMP/ TI(nnz,nnr),TE(nnz,nnr)
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /BK1/ F,Z,FREQL,TCCUT,ANOMR,ANOMT,RHOCUT
      COMMON /BKABS/ REF,ROCRIT
      COMMON /BKODD/ ODD,CLASS,POND
      COMMON /BKRG/ RGI,RGE,GAMMAI,GAMMAE
      COMMON /BKSTEP/ RHO0,RHOLT
C     LEVEL2,XI,WW,KAPPA0,TAU,RHO,RHO1,EINTI,EINTE,TI,TE
      DUM=XI1*DZ
      IM1=2
      DO 1 I=2,IT1
      I1=I+1
      JM1=1
      DO 2 J=1,JT
      J1=J+1
      WW(I,J)=0.0
      ALNRHO=ALOG(RHO(I,J))
      ALNTE=ALOG(TE(I,J))
      IF (TE(I,J).GT.5.65E5) GO TO 11
      SPITZ=SPIT1-0.4799*(ALNRHO-3.0*ALNTE)
      GO TO 12
   11 SPITZ=SPIT2-0.4799*ALNRHO+0.9989*ALNTE
   12 IF (SPITZ.LT.1.0) SPITZ=1.0
      Y=SQRT(TE(I,J))
      T=T1*TE(I,J)*Y/(SPITZ*RHO(I,J))
      LAMBDA=THERM*Y*T
      GRADT=SQRT(((TE(I1,J)-TE(IM1,J))/DZ)**2+
     1((TE(I,J1)-TE(I,JM1))/DR)**2)
      IF (TE(I,J).LE.TI(I,J).OR.CLASS) GO TO 16
      FL=GRADT*LAMBDA/(ACOUST*TE(I,J))
      IF (FL.LE.1.0) GO TO 16
      LAMBDA=LAMBDA/FL
      T=LAMBDA/(THERM*Y)
   16 FL=1.0/(1.0+ANOMT*GRADT*LAMBDA/TE(I,J))
C
C  CALCULATION OF THE LASER ABSORPTION COEFFICIENT IN EACH CELL
C
   19 IF (RHO(I,J).GE.ROCRIT) GO TO 18
      X=RHO(I,J)/ROCRIT
      SPITZ=SPITZ+0.5*ALOG(X)
      IF (SPITZ.LT.1.0) SPITZ=1.0
      XI(I,J)=DUM*RHO(I,J)/(T*(1.0-X**2)*SPITZ)
      GO TO 15
   18 XI(I,J)=0.0
   15 CONTINUE
      TAU(I,J)=T*T3
      DELTA=1.0/DELTA0
C
C  CALCULATION OF THE THERMAL CONDUCTIVITIES OF THE PLASMA
C
      FL=FL*DELTA
      Y=FL*T*CC25(2)*RHO(I,J)*TE(I,J)
      KAPPA0(I,J,2)=Y*GAMA01
      X2=TI(I,J)*SQRT(TI(I,J))
      T=T2*X2/(SPITZ*RHO(I,J))
      FL=0.5*SQRT(((TI(I1,J)-TI(IM1,J))/DZ)**2+
     1((TI(I,J1)-TI(I,JM1))/DR)**2)
      X2=X2/(1.0+CCFL*TI(I,J)*FL/RHO(I,J))
      KAPPA0(I,J,1)=CC25(1)*X2*TI(I,J)
    2 JM1=J
    1 IM1=I
      DO 4 K=1,2
    4 CALL RESET(KAPPA0(1,1,K))
      RETURN
      END
      SUBROUTINE ACCLN
      include 'polx.i'
C
C   THIS SUBROUTINE ADVANCES THE VELOCITY TO THE INTERMEDIATE STEP
C
      COMMON /AREA/ SR(nnr),SR1(nnr),SZ(nnr),VOL(nnr)
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /PRES/ P(nnz,nnr),PF(nnz,nnr)
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /TOT/ TOTAL(3,4)
      COMMON /VEL/ U(nnz,nnr),U1(nnz,nnr),V(nnz,nnr),V1(nnz,nnr)
C     LEVEL2,RHO,RHO1,P,PF,U,U1,V,V1
      DUM=DT/DZ
      DO 11 J=1,JT
      DUM2=DUM*VOL(J)
      DUM1=0.5*DUM2
      RV=0.5*(P(2,J)+P(1,J))+PF(2,J)
      TOTAL(1,4)=TOTAL(1,4)+DUM2*RV
      DO 12 I=2,IT1
      I1=I+1
      RV1=0.5*(P(I1,J)+P(I,J))+PF(I1,J)
      U1(I,J)=U(I,J)-DUM*RHO1(I,J)*(RV1-RV)
      U(I,J)=0.5*(U(I,J)+U1(I,J))
   12 RV=RV1
      TOTAL(1,4)=TOTAL(1,4)-DUM2*RV
      TOTAL(3,4)=TOTAL(3,4)-0.5*DUM2*
     1((P(IT1,J)+P(IT2,J)+PF(IT2,J))*U(IT1,J)-
     2(P(1,J)+P(2,J)+PF(2,J))*U(2,J))
      U1(1,J)=U1(2,J)
      U(1,J)=U(2,J)
      U1(IT2,J)=U1(IT1,J)
      U(IT2,J)=U(IT1,J)
   11 CONTINUE
      DUM=DT/DR
      DUM1=0.5*DT
      DUM2=SR(JT1)*DT
      DUM3=0.01989437*DUM*SR1(1)
      DO 21 I=2,IT1
      RV=0.0
      PV=0.0
      TOTAL (2,4)=TOTAL(2,4)+DUM*RV*VOL(1)
      DO 22 J=1,JT
      J1=J+1
      RV1=0.5*SR(J1)*(P(I,J1)-P(I,J))
      DV=(RV+RV1+PV)/SR1(J)
      V1(I,J)=V(I,J)-DUM*RHO1(I,J)*DV
      V(I,J)=0.5*(V(I,J)+V1(I,J))
   22 RV=RV1
      TOTAL(3,4)=TOTAL(3,4)-0.5*DUM2*(P(I,JT)+P(I,JT1))*V(I,JT)
      V1(I,JT1)=V1(I,JT)
      V(I,JT1)=V(I,JT)
   21 CONTINUE
      CALL RESET(V)
      CALL RESET(V1)
      CALL RESET(U)
      CALL RESET(U1)
      RETURN
      END
      SUBROUTINE ENERGY
      include 'polx.i'
C
C   THIS SUBROUTINE ADVANCES THE ENERGY TO THE INTERMEDIATE STEP.
C   THE ENERGY IS ADVANCED BY THE ICCG METHOD
C
      REAL KAPPA0(nnz,nnr,2)
      DIMENSION A(nnrz),B(nnrz),C(nnrz),D(nnrz),
     +          E(nnrz),F(nnrz),G(nnrz)
      DIMENSION E1(nnrz)
      COMMON /ABCD/ A,B,C,D,E,F,G
      COMMON /ABS1/ EX(nnz,nnr,2)
      COMMON /AREA/ SR(nnr),SR1(nnr),SZ(nnr),VOL(nnr)
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /COEFFT/ KAPPA0,TAU(nnz,nnr)
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /ENER/ EINT(nnz,nnr,2)
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /TOT/ TOTAL(3,4)
      COMMON /VEL/ U(nnz,nnr),U1(nnz,nnr),V(nnz,nnr),V1(nnz,nnr)
      COMMON /BKRG/ RG(2),GAMMA(2)
      EQUIVALENCE (E1(1),G(1))
C     LEVEL2,A,B,C,D,E,F,G,E1,EX,KAPPA0,
C    1TAU,RHO,RHO1,EINT,U,U1,V,V1
      DUMZ=0.5*DT/DZ
      DUMR=0.5*DT/DR
      DO 1 IP=1,2
      DUM=0.25*GAMMA(IP)*DT
C
C  IMPLICIT CALCULATION IN Z DIRECTION
C
      K=0
      KM=JT
      K1=1
      I=2
      IM1=2
      DO 10 I1=3,IT2
      J=1
      JM1=1
      RVR1=0.0
      DO 11 J1=2,JT1
      K=K+1
      KM=KM+1
      K1=K1+1
      E1(K)=EINT(I,J,IP)
      RV=DUMR*SR(J1)*(KAPPA0(I,J1,IP)+KAPPA0(I,J,IP))
      IF (J.NE.1) IF (J-JT) 110,111,110
      B(K)=0.0
  110 B(K1)=RV
      C(K)=B(K)+B(K1)
      GO TO 12
  111 B(K1)=0.0
      C(K)=B(K)
   12 RV=DUMZ*SZ(J)*(KAPPA0(I1,J,IP)+KAPPA0(I,J,IP))
      IF (I.NE.2) IF (I-IT1)  120,121,120
      A(K)=0.0
  120 A(KM)=RV
      C(K)=C(K)+A(K)+A(KM)
      GO TO 13
  121 A(KM)=0.0
      C(K)=C(K)+A(K)
   13 RVZ=DUM*SZ(J)*(U(I1,J)-U(IM1,J))
      RVR=DUM*SR(J1)*(V(I,J1)+V(I,J))
      RV=(RVZ+RVR-RVR1)/VOL(J)
      IF (ABS(RV).GT.0.8) RV=SIGN(0.8,RV)
      DM=RHO(I,J)*VOL(J)
      C(K)=C(K)+DM*(1.0+RV)
      F(K)=DM*(1.0-RV)*EINT(I,J,IP)
      IF (IP.EQ.2) F(K)=F(K)+EX(I,J,IP)
      RVR1=RVR
      JM1=J
   11 J=J1
      IM1=I
   10 I=I1
      CALL ICCG(E1,F,A,B,C,JT,KN)
      K=0
      DO 1 I=2,IT1
      DO 1 J=1,JT
      K=K+1
    1 EINT(I,J,IP)=E1(K)
      CALL RESET1(EINT(1,1,1))
      CALL RESET1(EINT(1,1,2))
      RETURN
      END
      SUBROUTINE ADVECT
      include 'polx.i'
C
C   THIS SUBROUTINE ADVANCES THE FLOW VARIABLES BY THE FLUX TERMS
C
      LOGICAL EDGE
      DIMENSION EI1(nnz,nnr),EE1(nnz,nnr),U0(nnz,nnr),V0(nnz,nnr)
      COMMON /ABCD/ A(nnz,nnr),B(nnrz),C(nnrz),
     +D(nnrz),E(nnrz),F(nnrz),
     1G(nnrz)
      COMMON /AREA/ SR(nnr),SR1(nnr),SZ(nnr),VOL(nnr)
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /ENER/ EI(nnz,nnr),EE(nnz,nnr)
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /BKSTEP/ RHO0,RHOLT
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /TOT/ TOTAL(3,4)
      COMMON /VEL/ U(nnz,nnr),U1(nnz,nnr),V(nnz,nnr),V1(nnz,nnr)
      EQUIVALENCE (EE1(1,1),B(1)),(EI1(1,1),C(1)),
     1(U0(1,1),D(1)),(V0(1,1),E(1))
C     LEVEL2,A,B,C,D,E,F,G,RHO,RHO1,EI,EE,EI1,EE1,U,U0,U1,
C    1V,V0,V1
      DO 3 J=1,JT
      DO 3 I=2,IT1
      U0(I,J)=U(I,J)
      V0(I,J)=V(I,J)
      DUM=EI(I,J)+0.5*(U1(I,J)**2+V1(I,J)**2)
      A(I,J)=EI(I,J)/DUM
      EI(I,J)=DUM
      U12=RHO(I,J)*VOL(J)
      RHO1(I,J)=U12
      U1(I,J)=U12*U1(I,J)
      V1(I,J)=U12*V1(I,J)
      EE1(I,J)=U12*EE(I,J)
    3 EI1(I,J)=U12*EI(I,J)
      CALL RESET1(RHO1)
      CALL RESET(U0)
      CALL RESET(U1)
      CALL RESET(V0)
      CALL RESET(V1)
      CALL RESET1(EE1)
      CALL RESET1(EI1)
      CALL RESET1(EI)
      VE=0.0
      UF=0.0
      UB=0.0
c   these pieces of code to do simple rezoning:
c
c   average velocity rezoning for Rayleigh-Taylor
      SUM=0.0
      SUM1=0.0
      DO 30 I=2,IT1
      DO 30 J=1,JT
      IF(U0(I,J).GE.0.0)GO TO 30
      SUM=SUM+U0(I,J)*VOL(J)*RHO(I,J)
      SUM1=SUM1+VOL(J)*RHO(I,J)
   30 CONTINUE
      VBAR=0.0
      IF(SUM1.GT.0.0)VBAR=SUM/SUM1
c     UB=VBAR
c     UF=VBAR
c
c  for the thin foils expand the mesh after the laser pulse
c
      if (time .gt. 5.0e-11) then
        ub = -2.0e8
        uf = +1.0e8
      endif
c
      ZB=ZB+UB*DT
      ZF=ZF+UF*DT
      DO 300 I=1,IT2
      uu = ub+(float(i-1)*(uf-ub)/float(it2-1))
      DO 300 J=1,JT
      U0(I,J)=U0(I,J)-uu
  300 CONTINUE
      CALL CORD
  314 CONTINUE
C
C   FLUX TRANSPORT IN THE Z DIRECTION
C
      DUM=0.5*DT/DZ
      DU=0.25*DZ/DT
      DO 1 J=1,JT
      DO 10 I=2,IT2
      IM1=I-1
      U12=0.5*(U0(IM1,J)+U0(I,J))
      IF (ABS(U12).GT.DU) U12=SIGN(DU,U12)
      IF (U12.GT.0.0) GO TO 11
      DM=SZ(J)*RHO(I,J)*U12*DT
      U12=U12*DR*DT
      CALL TRANS(U1(I,J),U1(IM1,J),U(I,J),DM)
      CALL TRANS(V1(I,J),V1(IM1,J),V(I,J),DM)
      CALL TRANS(EE1(I,J),EE1(IM1,J),EE(I,J),DM)
      CALL TRANS(EI1(I,J),EI1(IM1,J),EI(I,J),DM)
      GO TO 12
   11 DM=SZ(J)*RHO(IM1,J)*U12*DT
      U12=U12*DR*DT
      CALL TRANS(U1(I,J),U1(IM1,J),U(IM1,J),DM)
      CALL TRANS(V1(I,J),V1(IM1,J),V(IM1,J),DM)
      CALL TRANS(EE1(I,J),EE1(IM1,J),EE(IM1,J),DM)
      CALL TRANS(EI1(I,J),EI1(IM1,J),EI(IM1,J),DM)
   12 IF (I.NE.IT2) GO TO 120
      TOTAL(1,4)=TOTAL(1,4)-DM*U(IT1,J)
      TOTAL(2,4)=TOTAL(2,4)-DM*V(IT1,J)
      TOTAL(3,4)=TOTAL(3,4)-DM*(EE(IT1,J)+EI(IT1,J))
      GO TO 121
  120 IF (I.NE.2) GO TO 121
      TOTAL(1,4)=TOTAL(1,4)+DM*U(2,J)
      TOTAL(2,4)=TOTAL(2,4)+DM*V(2,J)
      TOTAL(3,4)=TOTAL(3,4)+DM*(EE(2,J)+EI(2,J))
  121 RHO1(I,J)=RHO1(I,J)+DM
   10 RHO1(IM1,J)=RHO1(IM1,J)-DM
      I0=2
      I1=3
      DRHO0=0.25*RHO1(2,J)
      DU0=0.25*U1(2,J)
      DV0=0.25*V1(2,J)
      DEE0=0.25*EE1(2,J)
      DEI0=0.25*EI1(2,J)
      DRHO1=RHO1(3,J)-RHO1(2,J)
      DU1=U1(3,J)-U1(2,J)
      DV1=V1(3,J)-V1(2,J)
      DEE1=EE1(3,J)-EE1(2,J)
      DEI1=EI1(3,J)-EI1(2,J)
      IF (DRHO1.LT.0.0) DRHO0=-DRHO0
      IF (DU1.LT.0.0) DU0=-DU0
      IF (DV1.LT.0.0) DV0=-DV0
      IF (DEE1.LT.0.0) DEE0=-DEE0
      IF (DEI1.LT.0.0) DEI0=-DEI0
      DO 1 I2=4,IT2
      EDGE=I2.EQ.IT2
      ETA=DUM*ABS(U0(I1,J)+U0(I0,J))
      IF (ETA.GT.0.25) ETA=0.25
      ETA=0.5*ETA*(1.0-ETA)
      CALL ANTDIF(RHO1(I2,J),RHO1(I1,J),RHO1(I0,J),SZ(J),SZ(J),DRHO2,
     1DRHO1,DRHO0,ETA,EDGE,.FALSE.)
      CALL ANTDIF(U1(I2,J),U1(I1,J),U1(I0,J),SZ(J),SZ(J),DU2,DU1,DU0,
     1ETA,EDGE,.FALSE.)
      CALL ANTDIF(V1(I2,J),V1(I1,J),V1(I0,J),SZ(J),SZ(J),DV2,DV1,DV0,
     1ETA,EDGE,.FALSE.)
      CALL ANTDIF(EE1(I2,J),EE1(I1,J),EE1(I0,J),SZ(J),SZ(J),DEE2,DEE1,
     1DEE0,ETA,EDGE,.FALSE.)
      CALL ANTDIF(EI1(I2,J),EI1(I1,J),EI1(I0,J),SZ(J),SZ(J),DEI2,DEI1,
     1DEI0,ETA,EDGE,.FALSE.)
      I0=I1
    1 I1=I2
      NPOINT=4
      IF (JTM1) 5,5,8
    8 IF (RE.GE.RE0) GO TO 84
      DO 80 I=2,IT1
   80 IF (RHO(I,JT).GT.RHO0) VE=AMAX1(VE,V0(I,JT))
      DU=(RE0-RE)/DT
      IF (VE.LT.DU) GO TO 81
      VE=DU
      RE0=0.9999999*RE0
   81 RE=RE+VE*DT
      DU=VE/FLOAT(JT)
      UM=-0.5*DU
      DO 83 J=1,JT1
      UM=UM+DU
      DO 83 I=2,IT1
   83 V0(I,J)=V0(I,J)-UM
      NPOINT=5
      CALL CORD
   84 DM=1.0/(DR*DZ)
      DO 4 J=1,JT
      DO 4 I=2,IT1
      RHO(I,J)=RHO1(I,J)/VOL(J)
      U12=1.0/RHO1(I,J)
      U(I,J)=U12*U1(I,J)
      V(I,J)=U12*V1(I,J)
      EE(I,J)=U12*EE1(I,J)
    4 EI(I,J)=U12*EI1(I,J)
      NPOINT=6
      CALL RESET1(RHO)
      CALL RESET(U)
      CALL RESET(V)
      CALL RESET1(EE)
      CALL RESET1(EI)
C
C   FLUX TRANSPORT IN THE R DIRECTION
C
      DUM=0.5*DT/DR
      DUM1=0.25/SR(2)
      DU=0.25*DR/DT
      DO 2 I=2,IT1
      DO 23 J=2,JT1
      JM1=J-1
      U12=0.5*(V0(I,JM1)+V0(I,J))
      IF (ABS(U12).GT.DU) U12=SIGN(DU,U12)
      IF (U12) 20,23,21
   20 DM=SR(J)*RHO(I,J)*U12*DT
      U12=U12*DZ*DT
      CALL TRANS(U1(I,J),U1(I,JM1),U(I,J),DM)
      CALL TRANS(V1(I,J),V1(I,JM1),V(I,J),DM)
      CALL TRANS(EE1(I,J),EE1(I,JM1),EE(I,J),DM)
      CALL TRANS(EI1(I,J),EI1(I,JM1),EI(I,J),DM)
      GO TO 22
   21 DM=SR(J)*RHO(I,JM1)*U12*DT
      U12=U12*DZ*DT
      CALL TRANS(U1(I,J),U1(I,JM1),U(I,JM1),DM)
      CALL TRANS(V1(I,J),V1(I,JM1),V(I,JM1),DM)
      CALL TRANS(EE1(I,J),EE1(I,JM1),EE(I,JM1),DM)
      CALL TRANS(EI1(I,J),EI1(I,JM1),EI(I,JM1),DM)
   22 IF (J.NE.JT1) GO TO 220
      TOTAL(1,4)=TOTAL(1,4)-DM*U(I,JT)
      TOTAL(2,4)=TOTAL(2,4)-DM*V(I,JT)
      TOTAL(3,4)=TOTAL(3,4)-DM*(EE(I,JT)+EI(I,JT))
  220 RHO1(I,J)=RHO1(I,J)+DM
      RHO1(I,JM1)=RHO1(I,JM1)-DM
   23 CONTINUE
      J0=1
      J1=2
      DRHO0=DUM1*RHO1(I,1)
      DU0=DUM1*U1(I,1)
      DV0=DUM1*V1(I,1)
      DEE0=DUM1*EE1(I,1)
      DEI0=DUM1*EI1(I,1)
      DRHO1=RHO1(I,2)/SR1(2)-RHO1(I,1)/SR1(1)
      DU1=U1(I,2)/SR1(2)-U1(I,1)/SR1(1)
      DV1=V1(I,2)/SR1(2)-V1(I,1)/SR1(1)
      DEE1=EE1(I,2)/SR1(2)-EE1(I,1)/SR1(1)
      DEI1=EI1(I,2)/SR1(2)-EI1(I,1)/SR1(1)
      IF (DRHO1.LT.0.0) DRHO0=-DRHO0
      IF (DU1.LT.0.0) DU0=-DU0
      IF (DV1.LT.0.0) DV0=-DV0
      IF (DEE1.LT.0.0) DEE0=-DEE0
      IF (DEI1.LT.0.0) DEI0=-DEI0
      DO 2 J2=3,JT1
      EDGE=J2.EQ.JT1
      ETA=DUM*ABS(V0(I,J1)+V0(I,J0))
      IF (ETA.GT.0.25) ETA=0.25
      ETA=0.5*ETA*(1.0-ETA)
      CALL ANTDIF(RHO1(I,J2),RHO1(I,J1),RHO1(I,J0),SR1(J0),SR(J1),
     1DRHO2,DRHO1,DRHO0,ETA,EDGE,.TRUE.)
      CALL ANTDIF(U1(I,J2),U1(I,J1),U1(I,J0),SR1(J0),SR(J1),DU2,DU1,
     1DU0,ETA,EDGE,.TRUE.)
      CALL ANTDIF(V1(I,J2),V1(I,J1),V1(I,J0),SR1(J0),SR(J1),DV2,DV1,
     1DV0,ETA,EDGE,.TRUE.)
      CALL ANTDIF(EE1(I,J2),EE1(I,J1),EE1(I,J0),SR1(J0),SR(J1),DEE2,
     1DEE1,DEE0,ETA,EDGE,.TRUE.)
      CALL ANTDIF(EI1(I,J2),EI1(I,J1),EI1(I,J0),SR1(J0),SR(J1),DEI2,
     1DEI1,DEI0,ETA,EDGE,.TRUE.)
      J0=J1
    2 J1=J2
      NPOINT=7
    5 CONTINUE
      DO 6 J=1,JT
      DO 6 I=2,IT1
      TOTAL(1,3)=TOTAL(1,3)+U1(I,J)
      TOTAL(2,3)=TOTAL(2,3)+V1(I,J)
      TOTAL(3,3)=TOTAL(3,3)+EE1(I,J)+EI1(I,J)
      RHO(I,J)=RHO1(I,J)/VOL(J)
      U12=1.0/RHO1(I,J)
      U(I,J)=U12*U1(I,J)
      V(I,J)=U12*V1(I,J)
      EE(I,J)=U12*EE1(I,J)
      EI(I,J)=U12*EI1(I,J)
      DUM=EI(I,J)-0.5*(U(I,J)**2+V(I,J)**2)
      IF (DUM.LE.0.0) GO TO 60
      EI(I,J)=DUM
      GO TO 6
   60 ET=EI(I,J)
      EI(I,J)=A(I,J)*EI(I,J)
      U12=1.0
      DIV=ET-DUM
      IF(DIV.NE.0.0.AND.ET.GT.EI(I,J))
     :  U12=SQRT((ET-EI(I,J))/DIV)
      U(I,J)=U12*U(I,J)
      V(I,J)=U12*V(I,J)
    6 CONTINUE
      NPOINT=8
      CALL RESET1 (RHO)
      CALL RESET(U)
      CALL RESET(V)
      CALL RESET1(EE)
      CALL RESET1(EI)
      RETURN
      END
      SUBROUTINE ANTDIF(A3,A2,A1,S1,S,DELTA3,DELTA2,DELTA1,ETA,EDGE,
     1SWEEP)
      include 'polx.i'
      DIMENSION S1(3)
      LOGICAL EDGE,SWEEP
C     LEVEL2,A3,A2,A1
      IF (EDGE) IF (DELTA2) 2,8,2
      IF (SWEEP) GO TO 1
      DELTA3=A3-A2
      IF (DELTA2) 40,7,50
    1 DELTA3=A3/S1(3)-A2/S1(2)
      IF (DELTA2) 41,7,51
    2 IF (SWEEP) GO TO 3
      DELTA3=SIGN((0.25*A2),DELTA2)
      IF (DELTA2) 40,7,50
    3 DELTA3=SIGN((0.25*A2/S),DELTA2)
      IF (DELTA2) 41,7,51
   40 FLUX=S*AMIN1(0.0,AMAX1(DELTA1,ETA*DELTA2,DELTA3))
      GO TO 6
   50 FLUX=S*AMAX1(0.0,AMIN1(DELTA1,ETA*DELTA2,DELTA3))
      GO TO 6
   41 FLUX=AMIN1(0.0,AMAX1(S1(1)*DELTA1,ETA*S*DELTA2,S1(2)*DELTA3))
      GO TO 6
   51 FLUX=AMAX1(0.0,AMIN1(S1(1)*DELTA1,ETA*S*DELTA2,S1(2)*DELTA3))
    6 A2=A2+FLUX
      A1=A1-FLUX
    7 DELTA1=DELTA2
      DELTA2=DELTA3
    8 RETURN
      END
      SUBROUTINE ERRORS
      include 'polx.i'
      COMMON /ERR/ ERROR(4)
      COMMON /LASIN/ ESUP,EREJEC,EABS
      COMMON /TOT/ TOTAL(3,4)
      DO 1 I=1,3
      TOTAL(I,1)=TOTAL(I,1)+TOTAL(I,4)
      IF (ABS(TOTAL(I,1)).LT.1.0E-20) GO TO 2
      ERROR(I)=(TOTAL(I,3)-TOTAL(I,1))/TOTAL(I,3)
      GO TO 3
    2 ERROR(I)=TOTAL(I,3)-TOTAL(I,1)
    3 IF (I.NE.3) GO TO 4
      ERROR(4)=ABS(TOTAL(3,3)-TOTAL(3,2)-TOTAL(3,4))/TOTAL(3,3)
      TOTAL(3,2)=TOTAL(3,3)
    4 TOTAL(I,3)=0.0
    1 TOTAL(I,4)=0.0
      RETURN
      END
      SUBROUTINE RESET(X)
      include 'polx.i'
      DIMENSION X(nnz,nnr)
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
C     LEVEL2,X
      DO 1 I=2,IT1
    1 X(I,JT1)=X(I,JT)
      DO 2 J=1,JT1
      X(IT2,J)=X(IT1,J)
    2 X(1,J)=X(2,J)
      RETURN
      END
      SUBROUTINE RESET1(X)
      include 'polx.i'
      DIMENSION X(nnz,nnr)
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
C     LEVEL2,X
      IF (RE.LT.RE0) GO TO 11
      DO 10 I=2,IT1
   10 X(I,JT1)=X(I,JT)
      GO TO 1
   11 DO 12 I=2,IT1
   12 X(I,JT1)=0.0
    1 IF (ZB.GT.ZB0) GO TO 21
      DO 20 J=1,JT1
   20 X(1,J)=X(2,J)
      GO TO 2
   21 DO 22 J=1,JT1
   22 X(1,J)=0.0
    2 IF (ZF.LT.ZF0) GO TO 31
      DO 30 J=1,JT1
   30 X(IT2,J)=X(IT1,J)
      RETURN
   31 DO 32 J=1,JT1
   32 X(IT2,J)=0.0
      RETURN
      END
      SUBROUTINE TRANS(X,X1,X0,DM)
      include 'polx.i'
C     LEVEL2,X,X1,X0
      X=X+X0*DM
      X1=X1-X0*DM
      RETURN
      END
      SUBROUTINE STATE
      include 'polx.i'
C
C   THIS SUBROUTINE CALCULATES THE STATE VARIABLES OF THE GAS FROM
C   THE THERMODYNAMIC EQUATION OF STATE
C
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /ENER/ EINT(nnz,nnr,2)
      COMMON /TFC / cmult,eemin,nstate
      COMMON /LENGTH/ DR,DZ,RE,RE0,ZB,ZB0,ZF,ZF0
      COMMON /PRES/ P(nnz,nnr),PF(nnz,nnr)
      COMMON /TEMP/ T(nnz,nnr,2)
      COMMON /BKRG/ RG(2),GAMMA(2)
C     LEVEL2,RHO,RHO1,EINT,P,PF,T
      DO 1 J=1,JT
      DO 1 I=2,IT1
      RHO1(I,J)=1.0/RHO(I,J)
      P(I,J)=RHO(I,J)*(GAMMA(1)*EINT(I,J,1)+GAMMA(2)*EINT(I,J,2))
      DO 1 K=1,2
    1 T(I,J,K)=EINT(I,J,K)/RG(K)
C     IF(NSTATE.GE.1)CALL TFEOS
      DO 11 K=1,2
      DO 11 I=2,IT1
      DO 11 J=1,JT
      IF(T(I,J,K).LT.1.0E4)T(I,J,K)=1.0E4
   11 CONTINUE
      DO 2 K=1,2
    2 CALL RESET1(T(1,1,K))
      CALL RESET(RHO1)
      CALL RESET1(P)
      RETURN
      END
      SUBROUTINE EQUIL
      include 'polx.i'
C
C   THIS SUBROUTINE CALCULATES THE EQUILIBRATION BETWEEN THE ELECTRON
C   AND ION TEMPERATURES
C
      REAL KAPPA0
      COMMON /CELL/ JTM1,JT,JT1,IT,IT1,IT2,KN,KN1
      COMMON /TFC / cmult,eemin,nstate
      COMMON /COEFFT/ KAPPA0(nnz,nnr,2),TAU(nnz,nnr)
      COMMON /DENS/ RHO(nnz,nnr),RHO1(nnz,nnr)
      COMMON /DRIVE/ W1,T1,T2,SPIT1,SPIT2,DELTA0,
     1GAMA01,RECNE,REST,CCFL,CC25(2),T3,RTE,ACOUST,XI1,THERM
      COMMON /ENER/ EINTI(nnz,nnr),EINTE(nnz,nnr)
      COMMON /TIM/ TOTIME,TOUT,TOTOUT,TS,TIME,n
      COMMON /TEMP/ TI(nnz,nnr),TE(nnz,nnr)
      COMMON /BKRG/ RGI,RGE,GAMMA(2)
C     LEVEL2,KAPPA0,TAU,RHO,RHO1,
C    1EINTI,EINTE,TI,TE
      IF(NSTATE.GE.1)GO TO 100
      DO 1 J=1,JT
      DO 1 I=2,IT1
      TQ=TS/TAU(I,J)
      IF (TQ.GT.1.0E2) GO TO 2
      DT=(EINTE(I,J)/RGE-EINTI(I,J)/RGI)*EXP(-TQ)
      GO TO 3
    2 DT=0.0
    3 TQ=EINTI(I,J)+EINTE(I,J)
      DT=RTE*(TQ+RGI*DT)
      EINTI(I,J)=TQ-DT
    1 EINTE(I,J)=DT
      RETURN
  100 DO 110 J=1,JT
      DO 110 I=2,IT1
      TQ=TS/TAU(I,J)
      IF(TQ.GT.100.)GO TO 102
      IF(EINTE(I,J).LT.0.0)GO TO 102
      DT=(TE(I,J)-TI(I,J))*EXP(-TQ)
      GO TO 103
  102 DT=0.0
  103 DE=0.5*DT*RGI
      EINTE(I,J)=EINTE(I,J)-DE
      EINTI(I,J)=EINTI(I,J)+DE
  110 CONTINUE
      END
      SUBROUTINE ICCG(X,Y,R,S,T,M,N)
      include 'polx.i'
C
C  SOLVES THE MATRIX EQUATION :
C  -R(K+M)*X(K+M)-S(K+1)*X(K+1)+T(K)*X(K)-S(K)*X(K-1)-R(K)*X(K-M)=Y(K)
C  USING ICCG
C
      DIMENSION X(N),Y(N),R(N),S(N),T(N)
      LOGICAL CHECK
      DIMENSION R1(nnrz),S1(nnrz),T1(nnrz),
     +          P(nnrz),Q(nnrz),Q1(nnrz)
      COMMON /BKABCD/ P,Q,Q1,R1,S1,T1,A1(nnrz),B1(nnrz),C1(nnrz)
      COMMON /BKACC1/ ACC,ACC1
C     LEVEL2,X,Y,R,S,T,P,Q,Q1,R1,S1,T1,A1,B1,C1
C
C  CALCULATE THE CHOLESKY INVERSE
C
      K0=1-M
      K1=1
      T1(1)=T(1)
      DO 1  K=2,N
      K0=K0+1
      S(K)=-S(K)
      S1(K)=S(K)
      IF (K0) 10,10,11
   10 T1(K)=T(K)-S1(K)**2/T1(K1)
      GO TO 12
   11 R(K)=-R(K)
      R1(K)=R(K)
      IF (M.EQ.2) S1(K)=S1(K)-R1(K)*S1(K1)/T1(K0)
      T1(K)=T(K)-S1(K)**2/T1(K1)-R1(K)**2/T1(K0)
   12 A=1.0E-4*ABS(T(K))
      IF (A.LT.1.0E-40) A=1.0E-40
      IF (ABS(T1(K)).LT.A) T1(K)=A
    1 K1=K
      K0=1-M
      K1=1
      DO 14 K=2,N
      K0=K0+1
      S1(K)=S1(K)/T1(K1)
      IF (K0) 14,14,13
   13 R1(K)=R1(K)/T1(K0)
   14 K1=K
C
C  PERFORM CONJUGATE GRADIENT ITERATION
C
      CALL PROD1(X,Q,R,S,T,M,N)
      DO 20 K=1,N
   20 Q(K)=Y(K)-Q(K)
      CALL INVRT1(Q,P,R1,S1,T1,M,N)
      B=0.0
      DO 21 K=1,N
   21 B=B+P(K)*Q(K)
      IF (ABS(B).LT.1.0E-10) RETURN
      DO 2 L=1,N
      CALL PROD1(P,Q1,R,S,T,M,N)
      A=0.0
      DO 22 K=1,N
   22 A=A+P(K)*Q1(K)
      A=B/A
      CHECK=.FALSE.
      DO 23 K=1,N
      X(K)=X(K)+A*P(K)
      Q(K)=Q(K)-A*Q1(K)
      IF (CHECK) GO TO 23
      IF (ABS(Y(K)).GT.1.0E-10) GO TO 230
      CHECK=ABS(Q(K)).GT.ACC1
      GO TO 23
  230 CHECK=ABS(Q(K)/Y(K)).GT.ACC
   23 CONTINUE
      IF (CHECK) GO TO 25
      CALL PROD1(X,Q1,R,S,T,M,N)
      DO 24 K=1,N
      IF (ABS(Y(K)).GT.1.0E-10) GO TO 240
      IF (ABS(Y(K)-Q1(K)).GT.ACC1) GO TO 25
      GO TO 24
  240 IF (ABS(1.0-Q1(K)/Y(K)).GT.ACC) GO TO 25
   24 CONTINUE
      RETURN
   25 CALL INVRT1(Q,Q1,R1,S1,T1,M,N)
      A=B
      B=0.0
      DO 26 K=1,N
   26 B=B+Q(K)*Q1(K)
      IF (ABS(B).LT.1.0E-10) RETURN
      A=B/A
      DO 2 K=1,N
    2 P(K)=Q1(K)+A*P(K)
      RETURN
      END
      SUBROUTINE INVRT1(X,Y,R,S,T,M,N)
      include 'polx.i'
      DIMENSION X(N),Y(N),R(N),S(N),T(N)
C     LEVEL2,X,Y,R,S,T
      K1=1
      K0=1-M
      Y(1)=X(1)
      DO 1 K=2,N
      K0=K0+1
      IF (K0) 10,10,11
   10 Y(K)=(X(K)-S(K)*Y(K1))
      GO TO 1
   11 Y(K)=(X(K)-S(K)*Y(K1)-R(K)*Y(K0))
    1 K1=K
      DO 2 K=1,N
    2 Y(K)=Y(K)/T(K)
      K=N
      K0=N+M
      DO 3 KK=2,N
      K1=K
      K=K-1
      K0=K0-1
      IF (K0.LE.N) GO TO 30
      Y(K)=(Y(K)-S(K1)*Y(K1))
      GO TO 3
   30 Y(K)=(Y(K)-S(K1)*Y(K1)-R(K0)*Y(K0))
    3 CONTINUE
      RETURN
      END
      SUBROUTINE PROD1(X,Y,R,S,T,M,N)
      include 'polx.i'
      DIMENSION X(N),Y(N),R(N),S(N),T(N)
C     LEVEL2,X,Y,R,S,T
      K0=2-M
      K3=1+M
      K1=1
      K=2
      Y(1)=T(1)*X(1)+S(2)*X(2)+R(K3)*X(K3)
      DO 1 K2=3,M+1
      K=K2-1
      K1=K2-2
      K0=K-M
      K3=K+M
      Y(K)=T(K)*X(K)+S(K)*X(K1)+S(K2)*X(K2)+R(K3)*X(K3)
    1 CONTINUE
      DO 2 K2=M+2,N-M+1
      K=K2-1
      K1=K2-2
      K0=K-M
      K3=K+M
      Y(K)=T(K)*X(K)+S(K)*X(K1)+S(K2)*X(K2)+R(K3)*X(K3)
     +    +R(K)*X(K0)
    2 CONTINUE
      DO 3 K2=N-M+2,N
      K=K2-1
      K1=K2-2
      K0=K-M
      K3=K+M
      Y(K)=T(K)*X(K)+S(K)*X(K1)+S(K2)*X(K2)
     +    +R(K)*X(K0)
    3 CONTINUE
      K0=K0+1
      K1=K
      Y(N)=T(N)*X(N)+S(N)*X(K1)+R(N)*X(K0)
      RETURN
      END
      BLOCK DATA
      include 'polx.i'
      REAL KAPPA0
      LOGICAL ODD,CLASS,POND
      COMMON /ABCD/ A(nnrz),B(nnrz),C(nnrz),
     +D(nnrz),E(nnrz),F(nnrz),
     1G(nnrz)
      COMMON /BKABCD/ AA(nnrz),BB(nnrz),CC(nnrz),
     +DD(nnrz),EE(nnrz),
     1PP(nnrz),QQ(nnrz),Q1(nnrz),R(nnrz)
      COMMON /ABS1/ WQ(nnz,nnr),W(nnz,nnr)
      COMMON /COEFFT/ KAPPA0(nnz,nnr,2),TAU(nnz,nnr)
      COMMON /ERR/ ERROR(4)
      COMMON /PRES/ P(nnz,nnr),PF(nnz,nnr)
      COMMON /TIM/ totime,tout,totout,dt,time,n
      COMMON /TOT/ TOTAL(12)
      COMMON /VEL/ U(nnz,nnr),U1(nnz,nnr),V(nnz,nnr),V1(nnz,nnr)
      COMMON /BKODD/ ODD,CLASS,POND
      COMMON /INCRE/ DTR,DTZ,DTTR,DTTZ,DTW,DTCFL,DTERR
C     LEVEL2,A,B,C,D,E,F,G,AA,BB,CC,DD,EE,PP,QQ,Q1,R,WQ,W,
C    1KAPPA0,TAU,P,PF,U,U1,V,V1
      DATA A,B,C,D,E,F,G,AA,BB,CC,DD,EE,PP,QQ,Q1,R/nn16*0.0/
      DATA N/0/,TOTOUT,TIME/2*0.0/
      DATA KAPPA0,TAU /nn3*0.0/
      DATA P,PF/nn2*0.0/
      DATA U,U1,V,V1/nn4*0.0/
      DATA W,WQ/nn2*0.0/
      DATA ERROR,TOTAL/16*0.0/
      DATA ODD/.TRUE./
      DATA DTR,DTZ,DTTR,DTTZ,DTW,DTCFL,DTERR /7*0.0/
      END

