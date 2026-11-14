# Concrete Float32 Metal / Float64 FEM types from the coupled trace.
# NamedTuple values below are types, not caches, factors or device buffers.
_metal_namedtuple_type(fields::NamedTuple) = NamedTuple{keys(fields),Tuple{values(fields)...}}

function metal_coupled_types()
    core = BeatEngineCore
    coupled = BeatEngineCoupled
    cc = BeatEngineCoupledCondensed
    cache_timings = _metal_namedtuple_type((;
        fem_matrix_cache_s=Float64,
        interface_operator_cache_s=Float64,
        bem_space_cache_s=Float64,
        bem_singular_cache_s=Float64,
        bem_cpu_assembly_cache_s=Float64,
        bem_device_regular_cache_s=Float64,
        bem_device_singular_cache_s=Float64,
        bem_device_image_cache_s=Float64,
        bem_identity_cache_s=Float64,
        device_block_cache_s=Float64,
        field_cache_s=Float64,
    ))
    transducer_operators = _metal_namedtuple_type((;
        fem_surface=SparseArrays.SparseMatrixCSC{Float32, Int},
        fem_force=SparseArrays.SparseMatrixCSC{Float32, Int},
        bem_surface=SparseArrays.SparseMatrixCSC{Float32, Int},
        bem_force=SparseArrays.SparseMatrixCSC{Float32, Int},
        bem_normal_velocity=SparseArrays.SparseMatrixCSC{Float32, Int},
    ))
    base = _metal_namedtuple_type((;
        bem_backend=Symbol,
        symmetry_mode=Symbol,
        retained_fem_vertices=Vector{Int},
        stiffness=SparseArrays.SparseMatrixCSC{Float32, Int},
        mass=SparseArrays.SparseMatrixCSC{Float32, Int},
        bulk_loss_mass=SparseArrays.SparseMatrixCSC{Float32, Int},
        bulk_loss_factor_by_vertex=Vector{Float32},
        wall_impedance_operators=Vector{Any},
        thermoviscous_operator=Nothing,
        interface_operators=coupled.InterfaceOperators{Float32},
        p1=core.P1Space,
        dp0=core.DP0Space,
        rule=core.TriangleRule{Float32},
        cpu_assembly_cache=Nothing,
        device_cache=core.MetalRegularAssemblyCache{Float32, Nothing},
        device_singular_cache=core.MetalSingularCorrectionCache{Float32},
        device_image_singular_cache=Nothing,
        singular_cache=core.SingularCorrectionCache{Float32},
        identity_p1_p1=Matrix{Float32},
        identity_p1_dp0=Matrix{Float32},
        device_identity_cache=Nothing,
        device_bem_flux=Nothing,
        device_bem_flux_sparse=Nothing,
        device_sparse_blocks=Nothing,
        field_cache=core.MetalFieldEvaluationCache{Float32},
        cuda_fem_analysis=Base.RefValue{Any},
        timings=cache_timings,
    ))
    cache = _metal_namedtuple_type((;
        base=base,
        quadrature_bundles=Dict{Int, Any},
        metal_combined_identity_store=Dict{Int, Any},
        metal_combined_identity_lock=ReentrantLock,
        base_quadrature_order=Int,
        singular_order=Int,
        timings=cache_timings,
        interface_mass_store=Dict{Symbol, Any},
        mumps_store=Dict{Symbol, Any},
        fem_float64_store=Dict{Symbol, Any},
    ))
    quadrature_bundle = _metal_namedtuple_type((;
        order=Int,
        rule=core.TriangleRule{Float32},
        cpu_assembly_cache=Nothing,
        device_cache=core.MetalRegularAssemblyCache{Float32, Nothing},
        identity_p1_p1=Matrix{Float32},
        identity_p1_dp0=Matrix{Float32},
        field_cache=core.MetalFieldEvaluationCache{Float32},
    ))
    # Merging the selected quadrature bundle replaces existing field types
    # without moving their keys, then appends its only new field, `order`.
    prepared = NamedTuple{(fieldnames(base)..., :order),Tuple{fieldtypes(base)...,Int}}
    condensation_timings = _metal_namedtuple_type((;
        analysis_s=Float64,
        factorization_s=Float64,
        schur_extraction_s=Float64,
        transducer_solves_s=Float64,
    ))
    condensation = _metal_namedtuple_type((;
        backend=Symbol,
        fem_solver_requested=Symbol,
        fem_solver_fallback_reason=Nothing,
        mumps_solver=cc.BeatEngineMumps.MumpsSchurSolver,
        mumps_owned=Bool,
        mumps_threads=Int,
        mumps_blas=String,
        fem_system=SparseArrays.SparseMatrixCSC{ComplexF64, Int},
        factorization=Nothing,
        interior_system=Nothing,
        interior_retained=Nothing,
        retained_interior=Nothing,
        schur=Matrix{ComplexF64},
        interior_vertices=Vector{Int},
        retained_vertices=Vector{Int},
        interior_count=Int,
        retained_count=Int,
        schur_block_columns=Int,
        schur_block_size=Int,
        schur_thread_count=Int,
        analysis_reused=Bool,
        factorization_cached=Bool,
        transducer_condensed=Bool,
        motion_interior=SparseArrays.SparseMatrixCSC{ComplexF64, Int},
        motion_solution=Matrix{ComplexF64},
        force_solution=Matrix{ComplexF64},
        motion_gamma=Matrix{ComplexF64},
        force_gamma=Matrix{ComplexF64},
        motion_force_correction=Matrix{ComplexF64},
        timings=condensation_timings,
    ))
    mass_block = _metal_namedtuple_type((;
        rows=Vector{Int},
        dofs=Vector{Int},
        contiguous=Bool,
        kind=Symbol,
        factor=SparseArrays.CHOLMOD.Factor{Float64, Int},
        fallback_reason=Nothing,
    ))
    mass_operator = _metal_namedtuple_type((;
        count=Int,
        blocks=Vector{mass_block},
        solver=Symbol,
    ))
    elimination = _metal_namedtuple_type((;
        bem_of_gamma=Vector{Int},
        gamma_dof=Vector{Int},
        mass_operator=mass_operator,
        schur_blocks=Vector{Matrix{ComplexF64}},
        motion_solution=Matrix{ComplexF64},
        interface_block=Matrix{ComplexF64},
    ))
    system_timings = _metal_namedtuple_type((;
        fem_system_s=Float64,
        bem_operator_s=Float64,
        bem_matrix_s=Float64,
        bem_combine_s=Float64,
        fem_condensation_s=Float64,
        fem_task_s=Float64,
        stage_overlap=Bool,
        block_assembly_s=Float64,
        interface_elimination_s=Float64,
        interface_mass_factorization_s=Float64,
        interface_mass_cached=Bool,
        interface_elimination_split=Dict{Symbol, Float64},
        coupled_factorization_s=Float64,
        replay_factorization_s=Float64,
    ))
    system = _metal_namedtuple_type((;
        fem_mesh=coupled.VolumeMesh{Float32},
        bem_mesh=core.BoundaryMesh{Float32},
        interface_map=coupled.ConformingInterfaceMap,
        interface_operators=coupled.InterfaceOperators{Float32},
        transducers=Vector{coupled.ElectrodynamicTransducer{Float32}},
        transducer_operators=transducer_operators,
        density=Float32,
        bulk_loss_factor=Float32,
        bulk_loss_factor_by_vertex=Vector{Float32},
        wall_admittances=Vector{ComplexF32},
        omega=Float32,
        wavenumber=Float32,
        field_cache=core.MetalFieldEvaluationCache{Float32},
        coupled=Nothing,
        factorization=cc.RefinedDenseLU,
        formulation=Symbol,
        regular_quadrature_order=Int,
        condensation=condensation,
        fem_range=UnitRange{Int},
        gamma_range=UnitRange{Int},
        retained_fem_vertices=Vector{Int},
        gamma_fem_vertices=Vector{Int},
        transducer_condensation=Bool,
        interface_elimination=Symbol,
        interface_elimination_data=elimination,
        gamma_row_range=UnitRange{Int},
        dense_scalar_type=DataType,
        fem_scalar_type=DataType,
        optimization_fallback_reasons=Vector{String},
        bem_range=UnitRange{Int},
        flux_range=UnitRange{Int},
        mechanical_range=UnitRange{Int},
        electrical_range=UnitRange{Int},
        bem_lhs=Nothing,
        bem_factorization=Nothing,
        bem_rhs_operator=Nothing,
        interface_radiation_replay=Nothing,
        prescribed_bem_rhs=Matrix{ComplexF32},
        prescribed_bem_neumann=SparseArrays.SparseMatrixCSC{ComplexF32, Int},
        bem_backend=Symbol,
        coupled_bem_assembly=Symbol,
        coupled_bem_assembly_fallback_reason=Nothing,
        linear_backend=Symbol,
        symmetry_mode=Symbol,
        cache=cache,
        owns_cache=Bool,
        validation_diagnostics=Bool,
        scalar_type=DataType,
        full_system_order=Int,
        solved_system_order=Int,
        timings=system_timings,
    ))
    solution = _metal_namedtuple_type((;
        fem_pressure=Vector{ComplexF32},
        bem_pressure=Vector{ComplexF32},
        interface_flux=Vector{ComplexF32},
        bem_neumann=Vector{ComplexF32},
        diaphragm_velocity=Vector{ComplexF32},
        voice_coil_current=Vector{ComplexF32},
        relative_residual=Nothing,
        fem_interior_residual=Float32,
        fem_rhs_condensation_s=Float64,
        fem_reconstruction_s=Float64,
        solve_split=Dict{Symbol, Float64},
        pressure_continuity_error=Float32,
        flux_conservation_error=Float32,
        all_bem_replay_error=Nothing,
        interface_map=coupled.ConformingInterfaceMap,
        interface_operators=coupled.InterfaceOperators{Float32},
    ))
    operators = _metal_namedtuple_type((;
        single_layer=Metal.MtlArray{ComplexF32, 2, Metal.SharedStorage},
        double_layer=Metal.MtlArray{ComplexF32, 2, Metal.SharedStorage},
        adjoint_double_layer=Metal.MtlArray{ComplexF32, 2, Metal.SharedStorage},
        hypersingular=Metal.MtlArray{ComplexF32, 2, Metal.SharedStorage},
    ))
    device_operators = _metal_namedtuple_type((;
        single_layer=Metal.MtlArray{ComplexF32, 2, Metal.SharedStorage},
        double_layer=Metal.MtlArray{ComplexF32, 2, Metal.SharedStorage},
        adjoint_double_layer=Metal.MtlArray{ComplexF32, 2, Metal.SharedStorage},
        hypersingular=Metal.MtlArray{ComplexF32, 2, Metal.SharedStorage},
        regular_pairs=Int,
        singular_pairs=Int,
        skipped_pairs=Int,
        image_singular_pairs=Int,
        on_gpu=Bool,
        gpu_backend=Symbol,
        host_staged_assembly=Bool,
        regular_kernel_threads=Int,
        regular_kernel_qpair_count=Int,
        regular_kernel_total_pairs=Int,
        regular_kernel_color_count=Int,
        regular_kernel_launches=Int,
        regular_kernel_mode=String,
        regular_assembly_mode=Symbol,
        singular_mode=Symbol,
    ))
    host_operators = _metal_namedtuple_type((;
        regular_pairs=Int,
        singular_pairs=Int,
        skipped_pairs=Int,
        image_singular_pairs=Int,
        gpu_backend=Symbol,
        host_staged_assembly=Bool,
        regular_kernel_threads=Int,
        regular_kernel_qpair_count=Int,
        regular_kernel_total_pairs=Int,
        regular_kernel_color_count=Int,
        regular_kernel_launches=Int,
        regular_kernel_mode=String,
        regular_assembly_mode=Symbol,
        singular_mode=Symbol,
        single_layer=Matrix{ComplexF32},
        double_layer=Matrix{ComplexF32},
        adjoint_double_layer=Matrix{ComplexF32},
        hypersingular=Matrix{ComplexF32},
        on_gpu=Bool,
        host_copy_of=Symbol,
        metal_backing=operators,
    ))
    # Combined A/C assembly (BLAB_METAL_COUPLED_BEM_ASSEMBLY, the Metal default).
    scatter = core.MetalSparseScatterCache{Metal.MtlArray{Int32, 1, Metal.PrivateStorage},
        Metal.MtlArray{Int32, 1, Metal.PrivateStorage}, Metal.MtlArray{ComplexF32, 1, Metal.PrivateStorage}}
    combined_identity = _metal_namedtuple_type((; p1_p1=scatter, p1_dp0=scatter))
    combined_device = _metal_namedtuple_type((;
        a=Metal.MtlArray{ComplexF32, 2, Metal.SharedStorage},
        c=Metal.MtlArray{ComplexF32, 2, Metal.SharedStorage},
        on_gpu=Bool,
        gpu_backend=Symbol,
        assembly_mode=Symbol,
    ))
    combined_operators = _metal_namedtuple_type((; combined=combined_device))
    return (; cache_timings, transducer_operators, base, cache, quadrature_bundle, prepared,
        condensation_timings, condensation, mass_block, mass_operator, elimination,
        system_timings, system, solution, operators, device_operators, host_operators,
        combined_identity, combined_device, combined_operators)
end

function metal_coupled_host_signatures()
    core = BeatEngineCore
    coupled = BeatEngineCoupled
    cc = BeatEngineCoupledCondensed
    types = metal_coupled_types()
    regular = core.MetalRegularAssemblyCache{Float32,Nothing}
    singular = core.MetalSingularCorrectionCache{Float32}
    signatures = Tuple{Any,Tuple}[]
    function add(f, args; kwargs...)
        if isempty(kwargs)
            push!(signatures, (f, args))
        else
            kwtype = _metal_namedtuple_type((; kwargs...))
            push!(signatures, (Core.kwcall, (kwtype, typeof(f), args...)))
        end
    end
    # 797.0 / 443.6 ms: exact driver keyword order and concrete Metal caches.
    add(cc.build_condensed_coupled_system,
        (coupled.VolumeMesh{Float32}, core.BoundaryMesh{Float32},
         coupled.ConformingInterfaceMap, Float32, Float32, Float32);
        quadrature_order=Int, regular_quadrature_order=Int, singular_order=Int,
        cache=types.cache, validation_diagnostics=Bool, retain_interface_radiation=Bool,
        symmetry_mode=Symbol, bulk_loss_factor_by_vertex=Vector{Float32},
        wall_impedances=Vector{NamedTuple}, thermoviscous_walls=Vector{NamedTuple},
        transducers=Vector{coupled.ElectrodynamicTransducer{Float32}},
        transducer_operators=types.transducer_operators,
        prescribed_bem_normal_velocity=SparseMatrixCSC{Float32,Int},
        allow_transducer_condensation=Bool, bem_operators=Nothing)
    # `bem_operators=Nothing` is every frequency the coupled sweep pipeline has not
    # started on (at least the first two of a sweep). Pipelined frequencies pass the
    # driver's producer closure and compile that call on the first one; not covered.
    add(cc.solve_condensed_coupled_excitations, (types.system, Vector{NamedTuple});
        reconstruct_interior=Bool)
    add(cc._fem_system_float64,
        (Dict{Symbol,Any}, coupled.VolumeMesh{Float32}, types.prepared,
         Float32, Float32, Float32))
    add(cc.release_condensed_coupled_system!, (types.system,))
    add(cc.interface_mass_diagnostics, (types.system,))
    add(cc.dense_solver_diagnostics, (types.system,))
    add(_condensed_split_timings, (types.system, Vector{types.solution}))
    add(merge, (types.base, types.quadrature_bundle))
    for type in (types.prepared, types.system, types.cache)
        add(getproperty, (type, Symbol))
    end
    add(hasproperty, (types.system, Symbol))
    add(get, (types.system, Symbol, Nothing))

    # Combined A/C assembly, in the build and on the sweep pipeline's producer task.
    add(cc._condensed_bem_assembly_plan, (types.cache, types.prepared, Type{Float32}, Int))
    for identity in (types.combined_identity, Nothing)
        add(cc._assemble_condensed_bem_operators, (core.BoundaryMesh{Float32}, types.prepared, Float32, Int);
            combined_identity=identity)
    end
    add(cc._combine_condensed_bem_operators!, (types.combined_operators, types.prepared, Float32))
    add(cc.assemble_condensed_bem_operators, (core.BoundaryMesh{Float32}, types.cache, Float32, Float32);
        regular_quadrature_order=Int, singular_order=Int)
    add(core.assemble_coupled_burton_miller_metal, (core.BoundaryMesh{Float32}, types.prepared, Float32);
        identity_cache=types.combined_identity)
    add(core.metal_host_coupled_burton_miller, (types.combined_device,))
    # Production releases the device tuple (`_combine_condensed_bem_operators!`).
    add(core.release_metal_coupled_burton_miller!, (types.combined_device,))

    # 107.1 / 197.7 / 131.6 ms, plus gather drivers. Only types are constructed.
    add(core.assemble_regular_galerkin_operators,
        (core.BoundaryMesh{Float32}, core.P1Space, core.DP0Space,
         Float32, core.TriangleRule{Float32});
        skip_singular=Bool, singular_order=Int, backend=Symbol, device_cache=regular,
        singular_cache=core.SingularCorrectionCache{Float32}, device_singular_cache=singular,
        symmetry_mode=Symbol)
    add(core._apply_metal_operator_p1_row_weights!,
        (types.operators, core.BoundaryMesh{Float32}, Symbol))
    add(core._launch_metal_regular_gather_kernels!, (types.operators, regular, Float32))
    add(core._launch_metal_gather_pair_kernels!,
        (types.operators, regular, Float32,
         Metal.MtlArray{Int32,1,Metal.PrivateStorage},
         Metal.MtlArray{Int32,1,Metal.PrivateStorage}, Int32,
         ntuple(_ -> Float32, 6)...))
    add(core._launch_metal_symmetry_regular_gather_kernels!,
        (types.operators, regular, singular, core.SymmetryTransform, Float32);
        skip_image_singular=Bool)
    add(core._launch_metal_singular_block_scatter_kernels!,
        (types.operators, regular, singular, Float32))
    add(core._launch_metal_singular_block_scatter_kernels!,
        (types.operators, regular, singular, Float32, core.SymmetryTransform))
    add(core.metal_host_operators, (types.device_operators,))
    add(core.burton_miller_neumann_matrices,
        (types.host_operators, Matrix{Float32}, Matrix{Float32}, Float32))
    add(core.release_operator_storage!, (types.host_operators,))

    for (f, tt, args, size) in metal_coupled_launch_types()
        if f !== core._metal_regular_pair_blocks_kernel!
            add(core._metal_launch, (typeof(f), Int, args...))
            add(core._metal_launch, (typeof(f), Int, args...); groupsize=Int)
        end
        # The traced HostKernel kwcall deliberately erases trailing arguments.
        kwtype = _metal_namedtuple_type((; threads=size, groups=size))
        push!(signatures, (Core.kwcall,
            (kwtype, Metal.HostKernel{typeof(f),tt}, first(args), Vararg{Any})))
    end
    return signatures
end

function metal_coupled_launch_types()
    core = BeatEngineCore
    kernels = (core._metal_regular_pair_blocks_kernel!, core._metal_gather_slp_adjoint_kernel!,
        core._metal_gather_dlp_hyp_kernel!, core._metal_singular_fused_blocks_kernel!,
        core._metal_singular_pair_gather_kernel!)
    shared_pair = (core._metal_gather_slp_adjoint_kernel!, core._metal_gather_dlp_hyp_kernel!,
        core._metal_singular_pair_gather_kernel!)
    launches = Tuple{Any,Type,Tuple,Type}[]
    for (f, tt) in metal_kernel_signatures()
        f in kernels || continue
        # Gather destinations (the first two arguments) are shared. Geometry,
        # indices and intermediate blocks are private. Expand fixed Varargs.
        args = Tuple(type <: Metal.MtlDeviceArray ?
            Metal.MtlArray{eltype(type),ndims(type),
                index <= 2 && f in shared_pair ? Metal.SharedStorage : Metal.PrivateStorage} : type
            for (index, type) in enumerate(fieldtypes(tt)))
        size = f === core._metal_regular_pair_blocks_kernel! ? Tuple{Int,Int} : Int
        push!(launches, (f, tt, args, size))
    end
    return launches
end

# Match a compiler-generated closure by its captured fields, then bind each
# type parameter through the field it represents. Fixed Core.Box fields are
# checked too. Aliased module bindings are deduplicated. Anything but exactly
# one match returns `nothing` with a warning naming the capture set, never a
# guess based on a gensym number or on enumeration order: the inventory then
# skips that entry instead of failing the package build. The strict check is
# compiled_metal_worker_tests.jl, which requires every closure to resolve (it runs
# in the macOS CI job without a GPU); metal_coupled_precompile_coverage_tests.jl,
# in hardware qualification (scripts/qualify_accelerator.py), measures the
# first-request compilation that remains.
function metal_captured_closure_type(mod::Module, captures::NamedTuple)
    candidates = Set{Type}()
    for name in names(mod; all=true)
        isdefined(mod, name) || continue
        wrapper = getfield(mod, name)
        (wrapper isa UnionAll || wrapper isa DataType) || continue
        T = Base.unwrap_unionall(wrapper)
        T isa DataType && T <: Function || continue
        captured_names = fieldnames(T)
        length(captured_names) == length(captures) &&
            all(name -> haskey(captures, name), captured_names) || continue
        fields = fieldtypes(T)
        parameters = Any[]
        bindable = true
        for parameter in T.parameters
            index = findfirst(field -> field === parameter, fields)
            # A parameter not carried by a captured field cannot be bound structurally: not a match.
            index === nothing && (bindable = false; break)
            push!(parameters, getproperty(captures, captured_names[index]))
        end
        bindable || continue
        concrete = isempty(parameters) ? wrapper : Core.apply_type(wrapper, parameters...)
        all(name -> fieldtype(concrete, name) === getproperty(captures, name), captured_names) || continue
        push!(candidates, concrete)
    end
    if length(candidates) != 1
        @warn "BEAT coupled Metal precompile: closure capture set matched $(length(candidates)) types; entry skipped" mod captures=keys(captures)
        return nothing
    end
    return only(candidates)
end

function metal_coupled_closure_types()
    cc = BeatEngineCoupledCondensed
    types = metal_coupled_types()
    timed_flux = metal_captured_closure_type(cc, (;
        system=types.system, flux_rhs_solution=Core.Box,
        rhs=Matrix{ComplexF64}, elimination=types.elimination))
    timed_dense = metal_captured_closure_type(cc, (;
        system=types.system, rhs=Matrix{ComplexF64}))
    solution_parts = metal_captured_closure_type(cc, (;
        system=types.system, fem_reconstruction_s=Float64,
        fem_interior_residual=Vector{Float32}, fem_pressure=Matrix{ComplexF32},
        interface_flux=Matrix{ComplexF32}, solution=Matrix{ComplexF32},
        fem_rhs_condensation_s=Float64, solve_split=Dict{Symbol,Float64},
        prescribed_bem_neumann=Matrix{ComplexF32}))
    condensation_options = _metal_namedtuple_type((;
        fem_solver=Symbol, mumps_store=Dict{Symbol,Any}, schur_block_columns=Int,
        motion_surface=SparseMatrixCSC{Float32,Int}, motion_force=SparseMatrixCSC{Float32,Int},
        schur_float64=Bool))
    fem_stage = metal_captured_closure_type(cc, (;
        presolve=Core.Box, mass_in_fem_stage=Core.Box, mass_operator=Core.Box,
        condensation_options=condensation_options, dense_type=Type{Float64},
        fem_system=Core.Box, normal_derivative_scale=ComplexF32, transducer_count=Int,
        interface_operators=BeatEngineCoupled.InterfaceOperators{Float32},
        gamma_fem_vertices=Vector{Int}, transducer_condensation=Bool,
        resolved_transducer_operators=types.transducer_operators,
        fem_task_s=Base.RefValue{Float64}))
    fem_task = fem_stage === nothing ? nothing : metal_captured_closure_type(cc, (; fem_stage=fem_stage))
    return (; timed_flux, timed_dense, solution_parts, fem_stage, fem_task)
end

function metal_coupled_runtime_signatures()
    cc = BeatEngineCoupledCondensed
    types = metal_coupled_types()
    closures = metal_coupled_closure_types()
    signatures = Type[Tuple{typeof(getproperty), BeatEngineCore.MetalGatherTables, Symbol}]
    # Entries whose closure could not be resolved are skipped (see metal_captured_closure_type).
    closures.fem_task === nothing || push!(signatures, Tuple{closures.fem_task})
    closures.timed_flux === nothing ||
        push!(signatures, Tuple{typeof(cc._split_timed!), closures.timed_flux, Dict{Symbol,Float64}, Symbol})
    closures.timed_dense === nothing ||
        push!(signatures, Tuple{typeof(cc._split_timed!), closures.timed_dense, Dict{Symbol,Float64}, Symbol})
    if closures.solution_parts !== nothing
        generator = Base.Generator{Base.OneTo{Int},closures.solution_parts}
        push!(signatures,
            Tuple{Type{Base.Generator}, closures.solution_parts, Base.OneTo{Int}},
            Tuple{typeof(collect), generator},
            Tuple{typeof(Base.collect_to_with_first!), Vector{types.solution}, types.solution, generator, Int})
    end
    keyword_body = typeof(Base.bodyfunction(which(Tuple{Metal.HostKernel})))
    for (f, tt, args, size) in metal_coupled_launch_types()
        kernel = Metal.HostKernel{typeof(f),tt}
        argument_tuple = Tuple{args...}
        push!(signatures,
            Tuple{keyword_body, size, size, Nothing, Bool, kernel, first(args), Vararg{Any}},
            Tuple{typeof(map), typeof(Metal.mtlconvert), argument_tuple},
            Tuple{typeof(Metal.encode_arguments!), Metal.MTL.MTLComputeCommandEncoder,
                kernel, Metal.KernelState, typeof(f), args...},
            Tuple{typeof(append!), Vector{Any}, Tuple{typeof(f),argument_tuple}})
        closure = metal_captured_closure_type(Metal, (;
            groups=size, threads=size, queue=Nothing, submit=Bool,
            kernel=kernel, args=argument_tuple))
        closure === nothing ||
            push!(signatures, Tuple{Type{Metal.ObjectiveC.Foundation.NSAutoreleasePool}, closure})
    end
    return signatures
end
