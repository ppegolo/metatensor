module Metatensor
    module lib
        using Preferences
        using ..Metatensor

        if has_preference(Metatensor, "libmetatensor")
            # if the users told us to use a custom libmetatensor (e.g. built
            # from the local checkout), load it!
            const libmetatensor = load_preference(Metatensor, "libmetatensor")
            const custom_build = true
        else
            using Metatensor_jll
            const custom_build = false
        end

        include("generated/_c_api.jl")

        function version()
            unsafe_string(lib.mts_version())
        end

        function __init__()
            if custom_build
                @info "using custom libmetatensor v$(version()) from $(libmetatensor)"
            end
        end
    end

    function last_error()
        error("TODO")
    end

    function check(status ::lib.mts_status_t)
        if status != lib.MTS_SUCCESS
            error(last_error())
        end
    end


    include("labels.jl")
    include("array.jl")
    include("block.jl")
    include("tensor.jl")
    include("io.jl")

    function __init__()
        lib.mts_disable_panic_printing()
    end

    export Labels, TensorBlock, TensorMap

end # module Metatensor
