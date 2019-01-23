subroutine stop_on_err(msg)
  ! Print error message and stop
  use iso_fortran_env, only : error_unit
  character(len=*), intent(in) :: msg
  if(len_trim(msg) > 0) then
    write (error_unit,*) trim(msg)
    write (error_unit,*) "test_lw_solver_noscat stopping"
    stop
  end if
end subroutine

! --------------------------------------------------------------------
program lw_solver_1ang
  use mo_rte_kind, only: wp, wl
  use mo_optical_props, only: ty_optical_props_arry, &
    ty_optical_props_1scl, ty_optical_props_2str, &
    ty_optical_props_nstr
  use mo_source_functions, only: ty_source_func_lw
  !use mo_rte_lw, only: rte_lw
  use mo_rte_solver_kernels, only: lw_solver_noscat

  use mo_fluxes_bygpoint, only: ty_fluxes_bygpoint
  use mo_test_files_io, only: read_optical_prop_values, &
    read_lw_Planck_sources, read_direction, read_lw_bc, read_lw_rt, &
    write_gpt_fluxes
  implicit none
! ------------------------------------------------------------------

  integer :: ncol, nlay, ngpt
  integer :: nang
  integer :: b, nBlocks, colS, colE
  integer, parameter :: blockSize = 8
  real(wp) :: propAng ! propagation angle

  character(len=128) :: fileName = 'rrtmgp-inputs-outputs.nc'
  character(len=2) :: propAngS = '60'

  class(ty_optical_props_arry), allocatable :: atmos_full, atmos_block
  type (ty_source_func_lw) :: sources_full, sources_block

  real(wp), dimension(:,  :), allocatable :: sfc_emis, sfc_emis_gpt, &
    secant
  real(wp), dimension(:    ), allocatable :: t_sfc
  real(wp), dimension(:,:,:), allocatable, target :: flux_up, flux_dn

  logical :: top_at_1
  logical(wl) :: top_at_1L
  type(ty_fluxes_bygpoint) :: fluxes

  ! ----------------------------------------------------------------
  !
  ! In early implementations this called the LW solver at an 
  ! intermediate level in the call tree - after some error checking 
  ! had been done but before deciding e.g. whether to use 
  ! no-scattering or two-stream solvers. The current implementation 
  ! is functionally the same as compute_fluxes_from_optics but
  ! writes out g-point fluxes
  !
  ! ----------------------------------------------------------------

  call read_optical_prop_values(fileName, atmos_full)
  call read_lw_Planck_sources(fileName, sources_full)
  call read_direction(fileName, top_at_1)
  call read_lw_bc(fileName, t_sfc, sfc_emis)
  call read_lw_rt(fileName, nang)
  top_at_1L = top_at_1

  ncol = sources_full%get_ncol()
  nlay = sources_full%get_nlay()
  ngpt = sources_full%get_ngpt()

  ! grab the propagation angle
  call get_command_argument(1, propAngS)
  read(propAngS, '(f2.0)') propAng

  allocate(flux_up(ncol,nlay+1,ngpt), flux_dn(ncol,nlay+1,ngpt))
  allocate(secant(ncol, ngpt))

  ! initialize down fluxes only?
  flux_dn(:,MERGE(1, nlay+1, top_at_1),:) = 0._wp

  ! propagation angle for this experiment (angle optimization)
  ! will be the same for all columns
  secant(:,:) = 1._wp / cosd(propAng)

  ! for lw_solver_noscat(), we need a surface emissivity array that 
  ! is ncol x ngpt -- sfc_emis is nband x ncol. and from what i see, 
  ! the Garand profiles have sfc_emis of unity for all columns and
  ! bands
  allocate(sfc_emis_gpt(ncol, ngpt))
  sfc_emis_gpt(:,:) = 1._wp

  select type (atmos_full)
    class is (ty_optical_props_1scl)
      allocate(ty_optical_props_1scl::atmos_block)
    class is (ty_optical_props_2str)
      allocate(ty_optical_props_2str::atmos_block)
    class is (ty_optical_props_nstr)
      allocate(ty_optical_props_nstr::atmos_block)
  end select

  ! Loop over subsets of the problem
  nBlocks = ncol/blockSize ! Integer division

  call lw_solver_noscat(ncol, nlay, ngpt, top_at_1L, &
    secant, 0.5_wp,  atmos_full%tau, sources_full%lay_source, &
    sources_full%lev_source_inc, sources_full%lev_source_dec, &
    sfc_emis_gpt, sources_full%sfc_source, flux_up, flux_dn)

  call write_gpt_fluxes(fileName, flux_up, flux_dn)
end program lw_solver_1ang
