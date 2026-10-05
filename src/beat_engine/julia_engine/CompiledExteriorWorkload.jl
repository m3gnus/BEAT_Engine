# Shared by the compiled CPU and Metal bundles. The plate touches both symmetry
# planes, has non-adjacent triangles, and has singular pairs with its images.
function workload_plate_mesh()
    io = IOBuffer()
    print(io, WORKLOAD_HEAD, "\$Nodes\n9\n")
    for y in 0:2, x in 0:2
        println(io, 1 + x + 3y, " ", 0.04x, " ", 0.04y, " 0.0")
    end
    print(io, "\$EndNodes\n\$Elements\n8\n")
    face = 0
    for y in 0:1, x in 0:1
        a = 1 + x + 3y
        for vertices in ((a, a + 1, a + 4), (a, a + 4, a + 3))
            face += 1
            println(io, face, " 2 2 2 2 ", join(vertices, " "))
        end
    end
    print(io, "\$EndElements\n")
    return String(take!(io))
end

function representative_workload_request(mesh)
    request = workload_request(mesh, "xy")
    request["frequencies_hz"] = [1000.0, 20000.0]
    request["solver_options"] = merge(Dict{String,Any}(request["solver_options"]), Dict(
        "quadrature_order" => 4, "singular_order" => 4,
        "regular_quadrature_mode" => "fixed",
    ))
    sphere = [[sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]
              for theta in range(0.0, pi; length=37)
              for phi in range(0.0, 2pi; length=73)[1:72]]
    diagonal = [[sin(a) / sqrt(2), sin(a) / sqrt(2), cos(a)]
                for a in range(0.0, pi; length=37)]
    append!(request["outputs"], [
        Dict("id" => "sphere", "quantity" => "exterior_pressure", "target_ids" => [],
             "options" => Dict("points_m" => sphere)),
        Dict("id" => "diagonal", "quantity" => "exterior_pressure", "target_ids" => [],
             "options" => Dict("points_m" => diagonal)),
        Dict("id" => "surface:p", "quantity" => "bem_boundary_pressure", "target_ids" => [],
             "options" => Dict()),
        Dict("id" => "surface:q", "quantity" => "bem_boundary_neumann", "target_ids" => [],
             "options" => Dict()),
    ])
    return request
end
