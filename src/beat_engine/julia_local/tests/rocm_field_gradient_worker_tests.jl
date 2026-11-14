module RocmFieldGradientWorkerTests
using Test, LinearAlgebra, StaticArrays
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
BeatEngineCore._require_rocm!()

function array_file(directory, name, values, shape)
    open(joinpath(directory, name), "w") do io
        write(io, values)
    end
    Dict("file"=>name, "offset"=>0, "nbytes"=>sizeof(values),
        "dtype"=>binary_dtype_name(eltype(values)), "shape"=>shape,
        "order"=>"C", "byte_order"=>"little")
end

@testset "ROCm retained gradient requests" begin
    for T in (Float32, Float64), symmetry in ("off", "ground")
        mktempdir() do directory
            # Non-dyadic coordinates detect accidental Float32 geometry conversion.
            vertices = T[0.1234567890123,1,0, 1.1234567890123,1,0, 0.1234567890123,2,0]
            points = T[0.456789012345,1.3,0.7, 1.456789012345,2.3,1.7]
            pressure = Complex{T}[1+2im,2-im,3+im]
            neumann = Complex{T}[2+3im]
            arrays = Dict(
                "boundary_points_m"=>array_file(directory,"vertices",vertices,[3,3]),
                "boundary_triangles"=>array_file(directory,"faces",Int64[0,1,2],[1,3]),
                "points_m"=>array_file(directory,"points",points,[2,3]),
                "boundary_pressure"=>array_file(directory,"pressure",pressure,[3]),
                "boundary_normal_derivative"=>array_file(directory,"neumann",neumann,[1]),
            )
            request = Dict{String,Any}("binary_array_schema_version"=>1,"binary_arrays"=>arrays,
                "domain_key"=>"gradient", "precision"=>T == Float32 ? "float32" : "float64",
                "bem_backend"=>"rocm", "symmetry"=>symmetry, "quadrature_order"=>4,
                "field_quantity"=>"pressure_gradient_v1", "frequency_hz"=>500.,
                "binary_result_file"=>"result.bin",
                "sound_speed_m_per_s"=>343.)
            path = joinpath(directory,"request.json")
            try
                for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
                    request["phasor_convention"] = convention
                    actual = evaluate_bem_field_request(request,path)
                    @test size(actual) == (2,4)
                    @test eltype(actual) == Complex{T}
                    encoded = write_field_binary_result(request,path,actual)
                    @test encoded["shape"] == [2,4]
                    @test encoded["order"] == "C"
                    @test encoded["dtype"] == binary_dtype_name(Complex{T})
                    decoded = reinterpret(Complex{T},read(joinpath(directory,"result.bin")))
                    @test collect(decoded) == [actual[i,j] for i in 1:2 for j in 1:4]
                    key = "gradient|$(T)|rocm|$(symmetry)|q4|pressure_gradient_v1"
                    retained = BEM_FIELD_EVALUATION_CACHES[key]
                    @test retained.mesh.vertices[1][1] == vertices[1]
                    @test evaluate_bem_field_request(request,path) == actual
                    @test BEM_FIELD_EVALUATION_CACHES[key].field_cache === retained.field_cache
                    reference = evaluate_bem_field_request(merge(request,Dict("bem_backend"=>"cpu")),path)
                    @test isapprox(actual,reference;rtol=T == Float32 ? 3e-4 : 1e-10)
                    @test phasor_convention() == NEGATIVE_TIME_PHASOR
                end
                @test_throws "fourth-order quadrature" evaluate_bem_field_request(
                    merge(request,Dict("quadrature_order"=>3)),path)
            finally
                release_all_bem_field_evaluation_caches!()
            end
            @test isempty(BEM_FIELD_EVALUATION_CACHES)
        end
    end
end
end
