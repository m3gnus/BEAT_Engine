module BeatEngineMeshData

using Base64, JSON, SHA
export decode_mesh_data, validate_mesh_sources, mesh_provenance

const WIDTHS = Dict("triangle" => 3, "triangle6" => 6, "tetra" => 4, "tetra10" => 10)

function unpack(raw, dtype, tail)
    raw isa AbstractDict && Set(keys(raw)) == Set(["dtype", "shape", "data"]) || error("Invalid mesh array descriptor.")
    raw["dtype"] == dtype || error("Unexpected mesh array dtype.")
    shape = raw["shape"]
    shape isa AbstractVector && length(shape) == length(tail) + 1 && shape[2:end] == tail || error("Invalid mesh array shape.")
    all(n isa Integer && !(n isa Bool) && n > 0 for n in shape) || error("Invalid mesh array dimensions.")
    expected = foldl(Base.checked_mul, Int.(shape); init=8)
    data = raw["data"]
    data isa AbstractString && ncodeunits(data) == 4 * cld(expected, 3) || error("Mesh byte count differs from shape.")
    bytes = base64decode(data)
    length(bytes) == expected || error("Mesh byte count differs from shape.")
    T = dtype == "<f8" ? Float64 : Int64
    # Decode little-endian buffers independently of the host byte order.
    bits = ltoh.(reinterpret(UInt64, bytes))
    values = collect(reinterpret(T, bits))
    return isempty(tail) ? values : permutedims(reshape(values, tail[1], shape[1]))
end

function decode_mesh_data(raw)
    raw isa AbstractDict && Set(keys(raw)) == Set(["schema_version", "points", "cells", "physical_names"]) || error("Invalid mesh_data fields.")
    typeof(raw["schema_version"]) == Int && raw["schema_version"] == 1 || error("Unsupported mesh_data version.")
    points = unpack(raw["points"], "<f8", [3])
    all(isfinite, points) || error("Mesh coordinates must be finite.")
    names = raw["physical_names"]
    names isa AbstractDict || error("Physical names must be an object.")
    groups = Set{Tuple{Int,Int}}()
    for (name, value) in names
        name isa AbstractString && !isempty(name) && value isa AbstractVector && length(value) == 2 || error("Invalid physical group.")
        all(n isa Integer && !(n isa Bool) for n in value) && value[1] > 0 && value[2] in (2, 3) || error("Invalid physical group.")
        push!(groups, (Int(value[1]), Int(value[2])))
    end
    raw["cells"] isa AbstractVector && !isempty(raw["cells"]) || error("Mesh requires cells.")
    cells = []
    for block in raw["cells"]
        block isa AbstractDict && Set(keys(block)) == Set(["type", "connectivity", "physical_tags"]) || error("Invalid mesh cell block.")
        kind = block["type"]
        haskey(WIDTHS, kind) || error("Unsupported mesh cell type.")
        indices = unpack(block["connectivity"], "<i8", [WIDTHS[kind]])
        tags = unpack(block["physical_tags"], "<i8", Int[])
        size(indices, 1) == length(tags) || error("Mesh cell and tag counts differ.")
        all(0 <= index < size(points, 1) for index in indices) || error("Mesh vertex index out of bounds.")
        all(length(Set(row)) == size(indices, 2) for row in eachrow(indices)) || error("Repeated mesh vertex index.")
        dimension = startswith(kind, "tetra") ? 3 : 2
        all((tag, dimension) in groups for tag in tags) || error("Mesh cell tag has no physical name.")
        push!(cells, (kind=kind, indices=indices .+ 1, tags=Int.(tags)))
    end
    return (points=points, cells=cells, names=names)
end

function validate_mesh_sources(system)
    for mesh in system["meshes"]
        if haskey(mesh, "mesh_data")
            isempty(mesh["file"]) || error("Mesh cannot supply both file and mesh_data.")
            # Content validation happens in the numerical loader, once per mesh.
            mesh["mesh_data"] isa AbstractDict || error("mesh_data must be an object.")
        else
            !isempty(mesh["file"]) || error("File-backed mesh requires a filename.")
        end
    end
end

"""
    payload_sha256(text) -> hex digest of the string's bytes

SHA.jl hashes a `String` through `codeunits`, whose 64-byte block copies each check aliasing with
`objectid`, which hashes the whole string: quadratic in its length. A 5.6 MB inline mesh request
spent about 85 s here (M1 Max). Hashing a byte copy gives the same digest in linear time.
"""
payload_sha256(text::AbstractString) = bytes2hex(sha256(Vector{UInt8}(text)))

function mesh_provenance(mesh, file_hash)
    if haskey(mesh, "mesh_data")
        # Request JSON is retained by the host; hash the actual received payload.
        return Dict("id" => mesh["id"], "source" => "memory", "file" => nothing,
                    "sha256" => payload_sha256(JSON.json(mesh["mesh_data"])))
    end
    return Dict("id" => mesh["id"], "file" => mesh["file"], "sha256" => file_hash(mesh["file"]))
end

end
