#pragma once

#include <cstddef>
#include <utility>

#include <socl/Buffer.hpp>

namespace soclblas{
    struct BufferView{
        socl::Buffer buffer;
        std::size_t offset;
        std::size_t size;

        BufferView(socl::Buffer buffer)
            : buffer(std::move(buffer)),
              offset(0),
              size(this->buffer.size()){
        }

        BufferView(socl::Buffer buffer, std::size_t offset, std::size_t size)
            : buffer(std::move(buffer)), offset(offset), size(size){
        }
    };
}
