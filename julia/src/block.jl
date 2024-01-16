# mts_block
# mts_block_copy
# mts_block_free

# mts_block_labels
# mts_block_gradient
# mts_block_data
# mts_block_add_gradient
# mts_block_gradients_list


struct TensorBlock
    __ptr::Ptr{lib.mts_block_t}
end
