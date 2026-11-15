module BeatEngineProvenance

using SHA, LinearAlgebra

export engine_identity, runtime_identity

const ENGINE_ROOT = normpath(joinpath(@__DIR__, ".."))
const IDENTITY = Ref{Any}(nothing)
const RUNTIME = Ref{Any}(nothing)

file_hash(path) = path !== nothing && isfile(path) ? open(sha256, path) |> bytes2hex : nothing

file_stamp(path) = (stat = Base.stat(path); (stat.size, stat.mtime))

# Hashing the system image takes most of a worker's start-up (about 0.7 s for a
# 200 MB image). A package image only loads on the system image it was built
# against, so record that image's hash while generating the package image and
# reuse it when the running image is the same file with the same size and
# modification time. Anything else (no package image, another image) hashes.
const BUILD_IMAGE = let pointer = Base.JLOptions().image_file
    image = pointer == C_NULL ? nothing : unsafe_string(pointer)
    ccall(:jl_generating_output, Cint, ()) == 1 && image !== nothing && isfile(image) ?
        (path=image, stamp=file_stamp(image), sha256=file_hash(image)) : nothing
end

function image_hash(image, build=BUILD_IMAGE)
    image !== nothing && build !== nothing && image == build.path && isfile(image) &&
        file_stamp(image) == build.stamp && return build.sha256
    return file_hash(image)
end

function git_value(args)
    try
        return strip(read(pipeline(`git -C $ENGINE_ROOT $args`, stderr=devnull), String))
    catch
        return nothing
    end
end

function engine_identity()
    if IDENTITY[] === nothing
        files = Dict{String,String}()
        for (label, directory) in (("julia_local", ENGINE_ROOT), ("beat_contract", joinpath(ENGINE_ROOT, "..", "beat_contract")))
            for (root, _, names) in walkdir(directory)
                relative = replace(relpath(root, directory), '\\' => '/')
                (relative == "." || label == "julia_local" && (relative == "src" || startswith(relative, "src/"))) || continue
                for name in names
                    (endswith(name, ".jl") || label == "beat_contract" && (endswith(name, ".json") || endswith(name, ".py"))) || continue
                    path = joinpath(root, name)
                    files["$label/" * replace(relpath(path, directory), '\\' => '/')] = file_hash(path)
                end
            end
        end
        digest_input = join(["$name\0$(files[name])\n" for name in sort!(collect(keys(files)))])
        # Installed wheels may live below an unrelated application checkout.
        # Only claim a Git revision if this engine source is tracked there.
        tracked = git_value(["ls-files", "--error-unmatch", "--", "src/BeatEngineProvenance.jl"])
        revision = tracked === nothing ? nothing : git_value(["rev-parse", "HEAD"])
        status = revision === nothing ? nothing : git_value(["status", "--porcelain", "--untracked-files=all", "--", ".", "../beat_contract"])
        IDENTITY[] = Dict("repository_revision" => revision,
            "repository_dirty" => status === nothing ? nothing : !isempty(status),
            "source_sha256" => bytes2hex(sha256(digest_input)), "source_files_sha256" => files)
    end
    return deepcopy(IDENTITY[])
end

function runtime_identity()
    if RUNTIME[] !== nothing
        return merge(deepcopy(RUNTIME[]), Dict("blas_threads" => BLAS.get_num_threads()))
    end
    project = Base.active_project()
    directory = project === nothing ? nothing : dirname(project)
    manifest = directory === nothing ? nothing : joinpath(directory, "Manifest-v$(VERSION.major).$(VERSION.minor).toml")
    if manifest !== nothing && !isfile(manifest)
        manifest = joinpath(directory, "Manifest.toml")
    end
    image_pointer = Base.JLOptions().image_file
    image = image_pointer == C_NULL ? nothing : unsafe_string(image_pointer)
    RUNTIME[] = Dict{String,Any}(
        "julia_version" => string(VERSION), "executable" => joinpath(Sys.BINDIR, Base.julia_exename()),
        "project_file" => project, "project_sha256" => file_hash(project),
        "manifest_file" => manifest !== nothing && isfile(manifest) ? manifest : nothing,
        "manifest_sha256" => file_hash(manifest), "julia_threads" => Threads.nthreads(),
        "sysimage_file" => image, "sysimage_sha256" => image_hash(image),
        "blas_threads" => BLAS.get_num_threads(), "blas_config" => string(BLAS.get_config()),
        "machine" => Sys.MACHINE, "kernel" => string(Sys.KERNEL), "cpu_name" => Sys.CPU_NAME,
    )
    return deepcopy(RUNTIME[])
end

end
