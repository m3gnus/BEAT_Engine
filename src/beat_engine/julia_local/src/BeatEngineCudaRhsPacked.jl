# Research-only packed block operator. One launch for all low-rank right
# factors, one for all dense/left-factor rows. No per-block launch or gather.
function _rhs_packed_rank!(values, data, q, columns, column_starts, ns, ranks,
                           v_offsets, task_blocks, task_local)
    task = (blockIdx().x-1)*blockDim().x + threadIdx().x
    task > length(values) && return
    b = task_blocks[task]; r = task_local[task]
    value = zero(eltype(q))
    for j in 1:ns[b]
        value += data[v_offsets[b] + (j-1)*ranks[b] + r] * q[columns[column_starts[b]+j]]
    end
    values[task] = value
    return
end

function _rhs_packed_rows!(yr, yi, data, q, columns, column_starts, ms, ns, ranks,
                           offsets, rank_offsets, rank_values, task_blocks, task_local, task_rows)
    task = (blockIdx().x-1)*blockDim().x + threadIdx().x
    task > length(task_rows) && return
    b = task_blocks[task]; row = task_local[task]
    value = zero(eltype(q))
    if ranks[b] < 0
        for j in 1:ns[b]
            value += data[offsets[b]+(j-1)*ms[b]+row] * q[columns[column_starts[b]+j]]
        end
    else
        for j in 1:ranks[b]
            value += data[offsets[b]+(j-1)*ms[b]+row] * rank_values[rank_offsets[b]+j]
        end
    end
    _cuda_atomic_add!(yr, task_rows[task], real(value))
    _cuda_atomic_add!(yi, task_rows[task], imag(value))
    return
end

function apply_cuda_packed_rhs(p, q, row_count)
    values = CUDA.zeros(eltype(q), length(p.rank_blocks))
    yr = CUDA.zeros(real(eltype(q)), row_count)
    yi = CUDA.zeros(real(eltype(q)), row_count)
    try
        if !isempty(values)
            CUDA.@cuda threads=256 blocks=cld(length(values),256) _rhs_packed_rank!(
                values,p.data,q,p.columns,p.column_starts,p.ns,p.ranks,p.v_offsets,p.rank_blocks,p.rank_local)
        end
        CUDA.@cuda threads=256 blocks=cld(length(p.task_rows),256) _rhs_packed_rows!(
            yr,yi,p.data,q,p.columns,p.column_starts,p.ms,p.ns,p.ranks,p.offsets,p.rank_offsets,
            values,p.task_blocks,p.task_local,p.task_rows)
        result = complex.(yr,yi)
        CUDA.synchronize()
        return result
    finally
        CUDA.unsafe_free!(values); CUDA.unsafe_free!(yr); CUDA.unsafe_free!(yi)
    end
end
