function mesh_file(io,mesh)
    println(io,"\$MeshFormat\n2.2 0 8\n\$EndMeshFormat\n\$Nodes\n",length(mesh.vertices))
    for (i,v) in enumerate(mesh.vertices)
        println(io,i," ",join(v," "))
    end
    println(io,"\$EndNodes\n\$Elements\n",length(mesh.faces))
    for (i,f) in enumerate(mesh.faces)
        println(io,i," 2 2 ",mesh.physical_tags[i]," ",mesh.physical_tags[i]," ",join(f," "))
    end
    println(io,"\$EndElements")
    flush(io)
end

function driver_request(file; tags=[2], ideal=false, backend=get(ENV, "BLAB_TEST_EXTERIOR_TRANSDUCER_BACKEND", "cpu"))
    r = JSON.parsefile(joinpath(@__DIR__,"..","..","beat_contract","example-exterior-request.json"))
    s = r["compiled_system"]
    s["contract_version"] = 3
    s["meshes"][1]["file"] = file
    s["meshes"][1]["scale_to_m"] = 1.0
    s["components"] = Any[]; s["boundaries"] = Any[]; s["excitation_ports"] = Any[]
    for (index,tag) in enumerate(tags)
        id,bid,pid = "driver:$index","boundary:$index","port:$index"
        isideal = ideal && index == length(tags)
        push!(s["boundaries"],Dict("id"=>bid,"name"=>bid,"kind"=>"moving","region_id"=>"region:exterior",
            "group"=>Dict("mesh_id"=>"mesh:exterior","dimension"=>2,"tag"=>tag,"name"=>nothing),"parameters"=>Dict()))
        params = isideal ? Dict{String,Any}() : Dict{String,Any}("re_ohm"=>6.,"le_h"=>0.0005,
            "bl_n_per_a"=>7.,"mmd_kg"=>0.015,"cms_m_per_n"=>0.0005,"rms_n_s_per_m"=>1.,"motion_axis"=>[0.,0.,1.])
        push!(s["components"],Dict("id"=>id,"name"=>id,"kind"=>isideal ? "ideal_velocity_source" : "electrodynamic_transducer",
            "boundary_ids"=>[bid],"parameters"=>params))
        push!(s["excitation_ports"],Dict("id"=>pid,"name"=>pid,"component_id"=>id,"kind"=>isideal ? "normal_velocity" : "voltage"))
    end
    r["excitation_port_ids"] = ["port:1"]
    r["frequencies_hz"] = [100.]
    r["solver_options"] = Dict("precision"=>"float64","bem_backend"=>backend,"quadrature_order"=>4,
        "singular_order"=>4,"regular_quadrature_mode"=>"fixed","transducer_reference_voltage_v"=>2.83)
    r["outputs"] = [Dict("id"=>q,"quantity"=>q,"target_ids"=>[],"options"=> q == "exterior_pressure" ?
        Dict("points_m"=>[[0.,0.,0.15],[0.,0.,2.],[1.,0.,2.]]) : Dict())
        for q in ("diaphragm_velocity","voice_coil_current","radiation_impedance_matrix","exterior_pressure",
                  "bem_boundary_pressure","bem_boundary_neumann")]
    r
end

function captured(r)
    mktemp() do _,io
        redirect_stdout(io) do
            solve_request(deepcopy(r);event_mode=true)
        end
        seekstart(io)
        results = [e["result"] for e in JSON.parse.(readlines(io)) if haskey(e,"result")]
        if get(r["solver_options"], "bem_backend", "cpu") == "rocm"
            @test !isempty(results)
            for result in results
                @test result["diagnostics"]["bem_backend"] == "rocm"
                # This helper is also reused by coupled wavelength sweeps.
                @test result["diagnostics"]["linear_solver"] in (
                    "rocm_rocsolver_dense_lu", "rocm_hybrid_cpu_sparse_schur_plus_rocsolver_dense_lu",
                )
            end
        end
        results
    end
end

function decoded(item)
    T = item["values"]["dtype"] == "complex64" ? ComplexF32 : ComplexF64
    values = ComplexF64.(reinterpret(T,base64decode(item["values"]["content_base64"])))
    shape = Tuple(Int.(item["values"]["shape"]))
    length(shape) == 1 ? values : permutedims(reshape(values,reverse(shape)))
end
quantities(r) = Dict(item["id"]=>item for item in r["quantities"])
relative(a,b) = norm(a-b)/norm(b)
