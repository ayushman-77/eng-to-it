# Build Stage
FROM ubuntu:22.04 AS build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY CMakeLists.txt .
COPY src/ ./src/

# Compile the C++ worker
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Production Stage
FROM ubuntu:22.04

WORKDIR /app
COPY --from=build /build/build/hpc_worker .

# Run the HPC C++ Worker
CMD ["./hpc_worker"]
