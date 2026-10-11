# Standalone bitwise gate or included by runtests.jl; reuse its loaded bundle.
module FieldOutputBatchTests

using Test, StaticArrays, JSON, Base64

if isdefined(Main, :CompiledWorkloadBundle)
    const Driver = Main.CompiledWorkloadBundle
elseif basename(dirname(Base.active_project())) == "julia_metal"
    import BeatEngineCompiledMetalBundle
    const Driver = BeatEngineCompiledMetalBundle
else
    import BeatEngineCompiledCpuBundle
    const Driver = BeatEngineCompiledCpuBundle
end
const Engine = Driver.BeatEngineCore

field_output(id, points; quantity="exterior_pressure", options=Dict{String,Any}()) = Dict(
    "id" => id, "quantity" => quantity, "target_ids" => [],
    "options" => merge(Dict{String,Any}("points_m" => points), options),
)

# Repeated points at different offsets, odd lengths, signed zero, and the real
# polar/sphere sizes exercise slicing and thread/block boundaries.
function field_outputs()
    polar = [[sin(a), -0.0, cos(a)] for a in range(0, pi; length=37)]
    sphere = [[sin(a) * cos(b), sin(a) * sin(b), cos(a)]
              for a in range(0, pi; length=37)
              for b in range(0, 2pi; length=73)[1:72]]
    return JSON.parse(JSON.json(Any[
        field_output("one", [[1.0, -0.0, 0.0]]),
        field_output("polar", polar),
        field_output("interface", [[0.0, 0.0, 2.0]]; quantity="interface_radiated_pressure"),
        Dict("id" => "boundary", "quantity" => "bem_boundary_pressure", "target_ids" => [],
             "options" => Dict{String,Any}()),
        field_output("sphere", sphere),
        field_output("repeat", reverse(polar)),
    ]))
end

# Compare every bit of both complex components (UInt32/UInt64, not == or ≈).
bits(values::AbstractVector{Complex{T}}) where {T} =
    reinterpret(T === Float32 ? UInt32 : UInt64, collect(values))

function check_quantity_bits(actual, expected)
    @test Dict(k => v for (k, v) in actual if k != "values") ==
          Dict(k => v for (k, v) in expected if k != "values")
    actual_values, expected_values = actual["values"], expected["values"]
    @test Dict(k => v for (k, v) in actual_values if k != "content_base64") ==
          Dict(k => v for (k, v) in expected_values if k != "content_base64")
    T = actual_values["dtype"] == "complex64" ? Float32 : Float64
    decoded_actual = reinterpret(Complex{T}, base64decode(actual_values["content_base64"]))
    decoded_expected = reinterpret(Complex{T}, base64decode(expected_values["content_base64"]))
    @test bits(decoded_actual) == bits(decoded_expected)
end

struct FieldEvaluationCounter
    point_counts::Vector{Int}
end

function Driver.exterior_field(points, counter::FieldEvaluationCounter, pressure, neumann,
                               wavenumber, cache, backend; cpu_kernel=:scalar)
    push!(counter.point_counts, length(points))
    return ComplexF32[ComplexF32(point[1], point[2]) for point in points]
end

function check_fields(evaluate, outputs, parsed, batch, T)
    together = evaluate(batch.points)
    @test all(isfinite, together)
    for output in outputs
        Driver.per_excitation_field_output(output) || continue
        id = String(output["id"])
        separate = evaluate(parsed[id])
        sliced = together[batch.ranges[id]]
        @test bits(sliced) == bits(separate)
        # Check the existing complex encoding, shape, axes and metadata too.
        check_quantity_bits(
            Driver.quantity_wire(output, Driver.rows([sliced], T), "Pa", ["excitation", "observation"]),
            Driver.quantity_wire(output, Driver.rows([separate], T), "Pa", ["excitation", "observation"]))
    end
end

@testset "exterior field batch points and backend eligibility" begin
    outputs = field_outputs()
    for T in (Float32, Float64)
        parsed = Driver.parse_field_output_points(outputs, T)
        for backend in (:cpu, :cuda, :rocm)
            batch = Driver.exterior_field_output_batch(outputs, parsed, backend)
            @test batch.points isa Vector{SVector{3,T}}
            @test batch.ranges == Dict("one" => 1:1, "polar" => 2:38,
                                       "sphere" => 39:2702, "repeat" => 2703:2739)
            for id in keys(batch.ranges)
                @test isequal(batch.points[batch.ranges[id]], parsed[id])
            end
        end
        @test Driver.exterior_field_output_batch(outputs, parsed, :metal) === nothing
        for backend in (:cpu, :cuda, :rocm)
            @test Driver.exterior_metal_field_output_batch(outputs, parsed, backend) === nothing
        end
        if T === Float64 || Engine.METAL_MODULE === nothing
            @test Driver.exterior_metal_field_output_batch(outputs, parsed, :metal) === nothing
        else
            metal_batch = Driver.exterior_metal_field_output_batch(outputs, parsed, :metal)
            @test metal_batch.indices == Dict("one" => 1, "polar" => 2, "sphere" => 3, "repeat" => 4)
            @test metal_batch.point_sets == [parsed[id] for id in ("one", "polar", "sphere", "repeat")]
            @test isempty(Driver.exterior_metal_field_output_batch(Any[], parsed, :metal).point_sets)
        end
        nonfield = [outputs[3], outputs[4]]
        empty_batch = Driver.exterior_field_output_batch(nonfield, parsed, :cpu)
        @test isempty(empty_batch.points) && isempty(empty_batch.ranges)
        @test isempty(Driver.exterior_field_output_batch(Any[], parsed, :cpu).points)
    end
end

@testset "weighted outputs are excluded from per-excitation field evaluations" begin
    outputs = Any[
        field_output("one", [[1.0, -0.0, 0.0]]),
        field_output("weighted", fill([3.0, 0.0, 1.0], 5);
                     options=Dict("excitation_weights" => [Dict("real" => 1.0)])),
        field_output("two", [[2.0, 0.0, 1.0], [2.0, 0.0, -1.0]]),
        field_output("sweep", fill([4.0, 0.0, 1.0], 7);
                     options=Dict("excitation_weights_sweep" => [Any[], [Dict("real" => 1.0)]])),
        field_output("empty-weights", [[5.0, 0.0, 1.0]];
                     options=Dict("excitation_weights" => Any[])),
    ]
    for T in (Float32, Float64)
        parsed = Driver.parse_field_output_points(outputs, T)
        for backend in (:cpu, :cuda, :rocm)
            batch = Driver.exterior_field_output_batch(outputs, parsed, backend)
            @test batch.ranges == Dict("one" => 1:1, "two" => 2:3)
            counter = FieldEvaluationCounter(Int[])
            # Each frequency/excitation evaluates only the three unweighted
            # points once, even when the outputs are requested in reverse order.
            for frequency in 1:2, excitation in 1:2
                fields = Dict{String,Any}()
                for id in ("two", "one", "two")
                    values = Driver.exterior_output_fields!(fields, id, parsed, counter,
                        nothing, nothing, T(frequency), nothing, backend, batch, nothing)
                    @test length(values) == length(parsed[id])
                end
                @test Set(keys(fields)) == Set(["one", "two"])
            end
            @test counter.point_counts == fill(3, 4)
            # A sweep with empty weights at this frequency still needs the
            # existing per-output fallback; it must not enter the batch.
            fields = Dict{String,Any}()
            Driver.exterior_output_fields!(fields, "sweep", parsed, counter,
                nothing, nothing, one(T), nothing, backend, batch, nothing)
            @test counter.point_counts[end] == 7
            @test Set(keys(fields)) == Set(["sweep"])
            Driver.exterior_output_fields!(fields, "one", parsed, counter,
                nothing, nothing, one(T), nothing, backend, batch, nothing)
            @test counter.point_counts[end-1:end] == [7, 3]
            # Unsupported batch paths continue to evaluate one output at a time.
            fallback_counter = FieldEvaluationCounter(Int[])
            fallback_fields = Dict{String,Any}()
            for id in ("two", "one")
                Driver.exterior_output_fields!(fallback_fields, id, parsed, fallback_counter,
                    nothing, nothing, one(T), nothing, backend, nothing, nothing)
            end
            @test fallback_counter.point_counts == [2, 1]
        end
        metal_batch = Driver.exterior_metal_field_output_batch(outputs, parsed, :metal)
        if metal_batch !== nothing
            @test metal_batch.indices == Dict("one" => 1, "two" => 2)
            @test length.(metal_batch.point_sets) == [1, 2]
        end
    end
end

@testset "wire comparison detects signed zero" begin
    positive = Driver.quantity_wire(field_output("zero", [[0.0, 0.0, 0.0]]),
                                    ComplexF32[ComplexF32(0.0f0, 0.0f0)], "Pa", ["observation"])
    negative = Driver.quantity_wire(field_output("zero", [[0.0, 0.0, 0.0]]),
                                    ComplexF32[ComplexF32(-0.0f0, 0.0f0)], "Pa", ["observation"])
    decode(q) = reinterpret(ComplexF32, base64decode(q["values"]["content_base64"]))
    @test decode(positive) == decode(negative)
    @test bits(decode(positive)) != bits(decode(negative))
end

@testset "CPU field separate/concatenated bitwise equality" begin
    @test Threads.nthreads() > 1
    mktempdir() do directory
        path = joinpath(directory, "plate.msh")
        write(path, Driver.workload_plate_mesh())
        for T in (Float32, Float64), symmetry in (:off, :xy), order in (2, 4)
            mesh = Engine.load_gmsh22_with_tags(path, one(T))
            cache = Engine.build_field_evaluation_cache(mesh, Engine.triangle_rule(T, order); symmetry_mode=symmetry)
            outputs = field_outputs()
            # A coincident source exercises the scalar skip and SIMD zero-radius mask.
            push!(outputs, field_output("coincident", [collect(cache.source_points[1])]))
            parsed = Driver.parse_field_output_points(outputs, T)
            batch = Driver.exterior_field_output_batch(outputs, parsed, :cpu)
            for drive in 1:2, kernel in (:scalar, :simd), convention in
                (Engine.NEGATIVE_TIME_PHASOR, Engine.POSITIVE_TIME_PHASOR)
                pressure = Complex{T}[Complex{T}(sin(0.3i + drive), cos(0.7i - drive)) for i in eachindex(mesh.vertices)]
                neumann = Complex{T}[Complex{T}(0.1sin(1.1i + drive), 0.2cos(0.9i - drive)) for i in eachindex(mesh.faces)]
                Engine.with_phasor_convention(convention) do
                    evaluate(points) = Engine.evaluate_galerkin_field_cpu(
                        points, mesh, pressure, neumann, T(2pi * 3000 / 343), cache; kernel)
                    check_fields(evaluate, outputs, parsed, batch, T)
                end
            end
        end
    end
end

@testset "compiled exterior/coupled batch preserves output wire and order" begin
    mktempdir() do directory
        path = joinpath(directory, "plate.msh")
        write(path, Driver.workload_plate_mesh())
        settings = Driver.coupled_workload_environment(; mumps=false)
        withenv(settings..., "OPENBLAS_NUM_THREADS" => "1") do
            old_threads = Driver.BLAS.get_num_threads()
            Driver.BLAS.set_num_threads(1)
            try
                for coupled in (false, true), T in (Float32, Float64)
                    request = coupled ? Driver.coupled_workload_request(; tiny=true) : Driver.workload_request(path, "xy")
                    request["solver_options"]["precision"] = T === Float32 ? "float32" : "float64"
                    request["frequencies_hz"] = [1000.0, 2000.0]
                    second_port = deepcopy(only(request["compiled_system"]["excitation_ports"]))
                    second_port["id"] = "port:second"
                    push!(request["compiled_system"]["excitation_ports"], second_port)
                    push!(request["excitation_port_ids"], second_port["id"])
                    outputs = field_outputs()
                    # Interface decomposition is intentionally unchanged.
                    coupled || filter!(o -> o["quantity"] != "interface_radiated_pressure", outputs)
                    if coupled
                        # Synthesis stays before field evaluation, including per-frequency weights.
                        push!(outputs, field_output("weighted", [[0.0, 0.0, 1.0]];
                            options=Dict("excitation_weights" => [Dict("real" => 0.3, "imag" => -0.7),
                                                                   Dict("real" => -0.2, "imag" => 0.4)])))
                        push!(outputs, field_output("sweep", [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
                            options=Dict("excitation_weights_sweep" => [Any[], [Dict("real" => -0.2, "imag" => 0.4),
                                                                              Dict("real" => 0.5, "imag" => -0.1)]])))
                    end
                    request["outputs"] = outputs
                    @test Driver.BeatEngineContract.validate_system_request(request) === nothing
                    run = Driver.solve_coupled_workload(request)
                    @test !run.outcome.cancelled && run.outcome.solved_count == 2
                    @test length(run.results) == 2
                    for result in run.results
                        @test [q["id"] for q in result["quantities"]] == [o["id"] for o in outputs]
                        @test result["diagnostics"]["timings"]["field_s"] >= 0
                    end
                    for (index, output) in enumerate(outputs)
                        single = deepcopy(request)
                        single["outputs"] = [output]
                        separate = Driver.solve_coupled_workload(single)
                        @test length(separate.results) == 2
                        for frequency in 1:2
                            check_quantity_bits(run.results[frequency]["quantities"][index],
                                                only(separate.results[frequency]["quantities"]))
                        end
                    end
                end
            finally
                Driver.BLAS.set_num_threads(old_threads)
                Driver.reset_compiled_workload_state!()
            end
        end
    end
end

@testset "Metal per-output partitions; shared sources and buffers bitwise gate" begin
    if get(ENV, "BLAB_RUN_COUPLED_METAL", "0") != "1"
        @test_skip "Set BLAB_RUN_COUPLED_METAL=1 under julia_metal to run Metal field checks."
    else
        # A requested hardware gate must not silently pass using a CPU bundle.
        @test isdefined(Engine, :evaluate_galerkin_field_metal)
        metal = Engine.metal_module()
        @test metal.functional()
        withenv("BLAB_METAL_FIELD_CHUNKS" => "auto") do
            # The real output sizes choose different source partitions. This is
            # why the driver must NOT concatenate Metal outputs in auto mode.
            @test Engine._metal_field_chunk_count(37, 10000) !=
                  Engine._metal_field_chunk_count(3 * 37 + 2664, 10000)
        end
        mktempdir() do directory
            path = joinpath(directory, "plate.msh")
            write(path, Driver.workload_plate_mesh())
            # 512 sources: auto selects different partitions for 37 and 2664
            # points. A larger override also forces chunks past the last source.
            T = Float32
            mesh = Engine.load_gmsh22_with_tags(path, one(T))
            cpu_cache = Engine.build_field_evaluation_cache(mesh, Engine.tensor_triangle_rule(T, 4); symmetry_mode=:xy)
            cache = Engine.build_metal_field_evaluation_cache(cpu_cache)
            point_cache = nothing
            try
                outputs = field_outputs()
                push!(outputs, field_output("weighted", fill([3.0, 0.0, 1.0], 5);
                    options=Dict("excitation_weights" => [Dict("real" => 1.0)])))
                push!(outputs, field_output("sweep", fill([4.0, 0.0, 1.0], 7);
                    options=Dict("excitation_weights_sweep" => [Any[]])))
                parsed = Driver.parse_field_output_points(outputs, T)
                batch = Driver.exterior_field_output_batch(outputs, parsed, :cpu)
                metal_batch = Driver.exterior_metal_field_output_batch(outputs, parsed, :metal)
                point_sets = metal_batch.point_sets
                @test length.(point_sets) == [1, 37, 2664, 37]
                point_cache = Engine.prepare_metal_field_output_points(point_sets, cache.source_count)
                uploaded_points = point_cache.device_points
                @test point_cache.point_counts == [1, 37, 2664, 37]
                @test length(uploaded_points) == 3 * (1 + 37 + 2664 + 37)
                @test !haskey(metal_batch.indices, "weighted") && !haskey(metal_batch.indices, "sweep")
                withenv("BLAB_METAL_FIELD_CHUNKS" => "auto") do
                    @test Engine._metal_field_chunk_count(37, cache.source_count) !=
                          Engine._metal_field_chunk_count(2664, cache.source_count)
                end
                k = T(2pi * 3000 / 343)
                @testset "one sources/field/reduce launch for all output blocks" begin
                    pressure = ones(ComplexF32, length(mesh.vertices))
                    neumann = ones(ComplexF32, length(mesh.faces))
                    for chunks in ("auto", "1")
                        observed = Any[]
                        observe(job) = push!(observed, job.source.specTypes.parameters[1].instance)
                        empty!(metal.kernel_instances)
                        withenv("BLAB_METAL_FIELD_CHUNKS" => chunks) do
                            Base.ScopedValues.with(metal.GPUCompiler.compile_hook => observe) do
                                Engine.evaluate_galerkin_field_metal_outputs(
                                    point_sets, pressure, neumann, k, cache; point_cache=point_cache)
                            end
                        end
                        expected = Any[Engine._metal_fast_field_sources_kernel!, Engine._metal_fast_field_outputs_kernel!]
                        chunks == "1" || push!(expected, Engine._metal_fast_field_outputs_reduce_kernel!)
                        @test observed == expected
                    end
                end
                for chunks in ("auto", "1", "4", string(cache.source_count + 1)), drive in 1:2,
                    convention in (Engine.NEGATIVE_TIME_PHASOR, Engine.POSITIVE_TIME_PHASOR)
                    pressure = ComplexF32[ComplexF32(sin(0.3i + drive), cos(0.7i - drive)) for i in eachindex(mesh.vertices)]
                    neumann = ComplexF32[ComplexF32(0.1sin(1.1i + drive), 0.2cos(0.9i - drive)) for i in eachindex(mesh.faces)]
                    withenv("BLAB_METAL_FIELD_CHUNKS" => chunks) do
                        Engine.with_phasor_convention(convention) do
                            reference(points) = Engine._evaluate_galerkin_field_metal_fast(
                                points, pressure, neumann, Engine.outgoing_wavenumber(k), cache)
                            together = Engine.evaluate_galerkin_field_metal_outputs(point_sets, pressure, neumann, k, cache)
                            @test length(together) == length(point_sets)
                            # The same upload serves changed frequencies and excitations.
                            for frequency_k in (k, 2k)
                                retained = Engine.evaluate_galerkin_field_metal_outputs(
                                    point_sets, pressure, neumann, frequency_k, cache; point_cache=point_cache)
                                @test point_cache.device_points === uploaded_points
                                partition = Engine._metal_field_output_partition!(point_cache, cache.source_count)
                                @test partition === Engine._metal_field_output_partition!(point_cache, cache.source_count)
                                for (points, values) in zip(point_sets, retained)
                                    separate = Engine._evaluate_galerkin_field_metal_fast(
                                        points, pressure, neumann, Engine.outgoing_wavenumber(frequency_k), cache)
                                    @test bits(values) == bits(separate)
                                end
                            end
                            for (points, values) in zip(point_sets, together)
                                @test all(isfinite, values)
                                @test bits(values) == bits(reference(points))
                                single = only(Engine.evaluate_galerkin_field_metal_outputs([points], pressure, neumann, k, cache))
                                @test bits(single) == bits(reference(points))
                            end
                            # Empty outputs do not shift neighbouring nonempty blocks.
                            empty_points = SVector{3,T}[]
                            mixed = Engine.evaluate_galerkin_field_metal_outputs(
                                [empty_points, point_sets[1], empty_points, point_sets[2], empty_points], pressure, neumann, k, cache)
                            @test isempty(mixed[1]) && isempty(mixed[3]) && isempty(mixed[5])
                            @test bits(mixed[4]) == bits(together[2])
                            @test bits(mixed[2]) == bits(together[1])
                            @test isempty(Engine.evaluate_galerkin_field_metal_outputs(
                                Vector{SVector{3,T}}[], pressure, neumann, k, cache))
                        end
                    end
                end
                @testset "allocation failure preserves request-owned points and borrowed pressure" begin
                    pressure = zeros(ComplexF32, length(mesh.vertices))
                    neumann = zeros(ComplexF32, length(mesh.faces))
                    d_pressure = metal.MtlArray(pressure)
                    try
                        @test_throws MethodError Engine.evaluate_galerkin_field_metal_outputs(
                            point_sets, pressure, [:invalid], k, cache; point_cache=point_cache)
                        @test_throws MethodError Engine.evaluate_galerkin_field_metal_outputs(
                            point_sets, d_pressure, [:invalid], k, cache; point_cache=point_cache)
                        retained = Engine.evaluate_galerkin_field_metal_outputs(
                            point_sets, d_pressure, neumann, k, cache; point_cache=point_cache)
                        fresh = Engine.evaluate_galerkin_field_metal_outputs(point_sets, pressure, neumann, k, cache)
                        @test point_cache.device_points === uploaded_points
                        @test bits(Array(d_pressure)) == bits(pressure)
                        @test all(bits(a) == bits(b) for (a, b) in zip(retained, fresh))
                    finally
                        metal.synchronize()
                        metal.unsafe_free!(d_pressure)
                    end
                end
                @testset "request points survive quadrature cache replacement" begin
                    pressure = ones(ComplexF32, length(mesh.vertices))
                    neumann = ones(ComplexF32, length(mesh.faces))
                    for order in (2, 4)
                        other_cache = Engine.build_metal_field_evaluation_cache(
                            Engine.build_field_evaluation_cache(mesh, Engine.triangle_rule(T, order);
                                                                symmetry_mode=:xy))
                        try
                            retained = Engine.evaluate_galerkin_field_metal_outputs(
                                point_sets, pressure, neumann, k, other_cache; point_cache=point_cache)
                            for (points, values) in zip(point_sets, retained)
                                separate = Engine._evaluate_galerkin_field_metal_fast(
                                    points, pressure, neumann, Engine.outgoing_wavenumber(k), other_cache)
                                @test bits(values) == bits(separate)
                            end
                        finally
                            Engine.release_metal_field_evaluation_cache!(other_cache)
                        end
                    end
                    @test point_cache.device_points === uploaded_points
                end
                # Retain the fixed-partition concatenation gate as well.
                for chunks in ("1", "4"), drive in 1:2
                    pressure = ComplexF32[ComplexF32(sin(0.3i + drive), cos(0.7i - drive)) for i in eachindex(mesh.vertices)]
                    neumann = ComplexF32[ComplexF32(0.1sin(1.1i + drive), 0.2cos(0.9i - drive)) for i in eachindex(mesh.faces)]
                    withenv("BLAB_METAL_FIELD_CHUNKS" => chunks) do
                        @test Driver.exterior_field_output_batch(outputs, parsed, :metal) === nothing
                        evaluate(points) = Engine.evaluate_galerkin_field_metal(
                            points, mesh, pressure, neumann, T(2pi * 3000 / 343), cache)
                        check_fields(evaluate, outputs, parsed, batch, T)
                    end
                end
                @testset "poisoned work buffers reused across excitations and partitions" begin
                    reuse_points = point_sets[1:2]
                    counts = length.(reuse_points)
                    max_chunks = cache.source_count + 1
                    d_points = metal.MtlArray(vcat([vec(Engine._metal_eval_point_arrays(points, T)) for points in reuse_points]...))
                    d_pressure = metal.MtlArray{ComplexF32}(undef, length(mesh.vertices))
                    d_neumann = metal.MtlArray{ComplexF32}(undef, length(mesh.faces))
                    d_weights = metal.MtlArray{Engine._MetalFloat4}(undef, cache.source_count)
                    d_partials = metal.MtlArray{ComplexF32}(undef, sum(counts) * max_chunks)
                    d_potentials = metal.MtlArray{ComplexF32}(undef, sum(counts))
                    try
                        for chunk_counts in ([max_chunks, max_chunks], [1, 1], [1, 4], [4, 1], [max_chunks, max_chunks]), drive in 1:3
                            pressure = ComplexF32[drive == 3 ? 0 : ComplexF32(sin(0.3i + drive), cos(0.7i - drive)) for i in eachindex(mesh.vertices)]
                            neumann = ComplexF32[drive == 3 ? 0 : ComplexF32(0.1sin(1.1i + drive), 0.2cos(0.9i - drive)) for i in eachindex(mesh.faces)]
                            copyto!(d_pressure, pressure)
                            copyto!(d_neumann, neumann)
                            fill!(d_weights, Engine._metal_float4(NaN32, NaN32, NaN32, NaN32))
                            fill!(d_partials, ComplexF32(NaN32, NaN32))
                            fill!(d_potentials, ComplexF32(NaN32, NaN32))
                            partition = Engine._prepare_metal_field_output_partition(counts, chunk_counts, cache.source_count)
                            try
                                Engine._metal_fast_field_outputs!(d_potentials, d_partials, d_points, d_weights,
                                    d_pressure, d_neumann, partition, Engine.outgoing_wavenumber(k), cache)
                            finally
                                metal.synchronize()
                                metal.unsafe_free!(partition.device_table)
                            end
                            host = Array(d_potentials)
                            active_partials = sum((n * c for (n, c) in zip(counts, chunk_counts) if c > 1); init=0)
                            if active_partials > 0
                                @test all(isfinite, Array(view(d_partials, 1:active_partials)))
                            end
                            expected = ComplexF32[]
                            for (points, chunks) in zip(reuse_points, chunk_counts)
                                withenv("BLAB_METAL_FIELD_CHUNKS" => string(chunks)) do
                                    append!(expected, Engine._evaluate_galerkin_field_metal_fast(
                                        points, pressure, neumann, Engine.outgoing_wavenumber(k), cache))
                                end
                            end
                            @test bits(host) == bits(expected)
                        end
                    finally
                        for buffer in (d_points, d_pressure, d_neumann, d_weights, d_partials, d_potentials)
                            metal.unsafe_free!(buffer)
                        end
                    end
                end
            finally
                point_cache === nothing || Engine.release_metal_field_output_points!(point_cache)
                Engine.release_metal_field_evaluation_cache!(cache)
            end
            @testset "compiled Metal exterior/coupled output order and wire" begin
                withenv(Driver.coupled_workload_environment(; mumps=false)...,
                        "BLAB_METAL_FIELD_CHUNKS" => "auto", "OPENBLAS_NUM_THREADS" => "1") do
                    old_threads = Driver.BLAS.get_num_threads()
                    Driver.BLAS.set_num_threads(1)
                    try
                        for coupled in (false, true)
                            request = coupled ? Driver.coupled_workload_request(; tiny=true, bem_backend="metal") :
                                Driver.workload_request(path, "xy")
                            request["solver_options"]["bem_backend"] = "metal"
                            request["solver_options"]["precision"] = "float32"
                            request["frequencies_hz"] = [1000.0, 2000.0]
                            second_port = deepcopy(only(request["compiled_system"]["excitation_ports"]))
                            second_port["id"] = "port:second"
                            push!(request["compiled_system"]["excitation_ports"], second_port)
                            push!(request["excitation_port_ids"], second_port["id"])
                            outputs = field_outputs()
                            coupled || filter!(o -> o["quantity"] != "interface_radiated_pressure", outputs)
                            if coupled
                                push!(outputs, field_output("weighted", [[0.0, 0.0, 1.0]];
                                    options=Dict("excitation_weights" => [Dict("real" => 0.3, "imag" => -0.7),
                                                                           Dict("real" => -0.2, "imag" => 0.4)])))
                            end
                            request["outputs"] = outputs
                            @test Driver.BeatEngineContract.validate_system_request(request) === nothing
                            run = Driver.solve_coupled_workload(request)
                            @test !run.outcome.cancelled && run.outcome.solved_count == 2
                            @test length(run.results) == 2
                            for result in run.results
                                @test [q["id"] for q in result["quantities"]] == [o["id"] for o in outputs]
                            end
                            for (index, output) in enumerate(outputs)
                                single = deepcopy(request)
                                single["outputs"] = [output]
                                separate = Driver.solve_coupled_workload(single)
                                @test length(separate.results) == 2
                                for frequency in 1:2
                                    check_quantity_bits(run.results[frequency]["quantities"][index],
                                                        only(separate.results[frequency]["quantities"]))
                                end
                            end
                        end
                    finally
                        Driver.BLAS.set_num_threads(old_threads)
                        Driver.reset_compiled_workload_state!()
                    end
                end
            end
        end
    end
end

end # module
