# mts_labels_create
# mts_labels_free

# mts_labels_position
# mts_labels_union
# mts_labels_intersection

# Not required initially
# mts_labels_set_user_data
# mts_labels_user_data


mutable struct Labels
    const __raw :: lib.mts_labels_t

    function Labels(raw::lib.mts_labels_t)
        @assert Int(raw.internal_ptr_) != 0 "Labels internal pointer must be non-NULL"
        this = new(raw)

        function finalize_labels(labels::Labels)
            ptr = pointer_from_objref(labels.__raw)
            lib.mts_labels_free(Base.convert(Ptr{lib.mts_labels_t}, ptr))
        end

        finalizer(finalize_labels, this)
        return this
    end
end

function Labels(names::Vector{String}, values::Array{T, 2}) where T <: Integer
    if length(names) != size(values)[1]
        error("expected the same number of names as there are columns in values")
    end

    c_names = Vector{Ptr{Cchar}}()
    for name in names
        push!(c_names, pointer(name))
    end

    values = convert(Array{Int32, 2}, values)

    raw = lib.mts_labels_t(
        Base.C_NULL,       # internal_ptr_
        pointer(c_names),  # names
        pointer(values),   # values
        length(names),     # size
        size(values)[2],   # count
    )

    check(lib.mts_labels_create(
        Base.convert(Ptr{lib.mts_labels_t}, pointer_from_objref(raw)))
    )

    return Labels(raw)
end


function Labels(name::String, values::Array{T, 2}) where T <: Integer
end

# TODO: Labels.range, Labels.empty, Labels.single

function Base.names(labels::Labels)
    c_names = unsafe_wrap(
        Vector{Ptr{Cchar}},
        labels.__raw.names,
        labels.__raw.size;
        own=false
    )

    return map(unsafe_string, c_names)
end

function Base.values(labels::Labels)
    return unsafe_wrap(
        Array{Int32, 2},
        labels.__raw.values,
        size(labels);
        own=false
    )
end

function Base.size(labels::Labels)
    return (
        labels.__raw.size,
        labels.__raw.count,
    )
end

# TODO: iteration over labels + LabelsEntry
