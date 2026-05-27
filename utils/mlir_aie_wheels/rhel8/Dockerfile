FROM centos:8

USER root
WORKDIR /workspace

# Fix CentOS 8 EOL mirrors
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-* && \
    sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*

RUN yum -y update && \
    yum -y install epel-release && \
    yum -y install \
        gcc gcc-c++ make clang lld \
        python3 python3-devel \
        openssl-devel bzip2-devel libffi-devel zlib-devel xz-devel \
        elfutils elfutils-libelf-devel boost-devel \
        git git-lfs wget unzip \
        ccache \
        patchelf && \
    yum -y install --enablerepo=powertools glibc-static libstdc++-static

# Build Python 3.11 from source
RUN wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz && \
    tar xzf Python-3.11.9.tgz && \
    cd Python-3.11.9 && \
    ./configure --enable-optimizations --enable-shared --prefix=/usr \
        LDFLAGS="-Wl,-rpath /usr/lib" && \
    make -j $(nproc) && \
    make altinstall && \
    cd /workspace && \
    rm -rf Python-3.11.9*

# Make python3.11 the default python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3.11 /usr/bin/pip3

# Install Python build dependencies
RUN python3.11 -m pip install --upgrade pip setuptools wheel && \
    python3.11 -m pip install \
        ninja cmake numpy \
        pybind11>=2.10.4 nanobind \
        auditwheel pylint

# CentOS 8 Boost 1.66 lacks cmake config files; create them for cmake 4.x
RUN mkdir -p /usr/lib64/cmake/Boost-1.66.0 && \
    printf 'set(Boost_FOUND TRUE)\nset(Boost_VERSION_STRING "1.66.0")\nset(Boost_VERSION "1.66.0")\nset(Boost_VERSION_MAJOR 1)\nset(Boost_VERSION_MINOR 66)\nset(Boost_VERSION_PATCH 0)\nset(Boost_INCLUDE_DIRS "/usr/include")\nset(Boost_LIBRARIES "")\n' \
        > /usr/lib64/cmake/Boost-1.66.0/BoostConfig.cmake && \
    printf 'set(PACKAGE_VERSION "1.66.0")\nif("${PACKAGE_FIND_VERSION}" VERSION_GREATER "${PACKAGE_VERSION}")\n  set(PACKAGE_VERSION_COMPATIBLE FALSE)\nelse()\n  set(PACKAGE_VERSION_COMPATIBLE TRUE)\n  if("${PACKAGE_FIND_VERSION}" VERSION_EQUAL "${PACKAGE_VERSION}")\n    set(PACKAGE_VERSION_EXACT TRUE)\n  endif()\nendif()\n' \
        > /usr/lib64/cmake/Boost-1.66.0/BoostConfigVersion.cmake

# Set up ccache symlinks
RUN mkdir -p /usr/lib64/ccache && \
    for tool in cc c++ gcc g++; do \
        ln -sf /usr/bin/ccache /usr/lib64/ccache/$tool; \
    done

ENV PATH="/usr/lib64/ccache:${PATH}"

# Raise open file limit for large link jobs
RUN echo "* soft nofile 65536" >> /etc/security/limits.conf && \
    echo "* hard nofile 65536" >> /etc/security/limits.conf

COPY build-wheels-rhel8.sh /workspace/
RUN chmod +x /workspace/build-wheels-rhel8.sh

ENTRYPOINT ["/bin/bash"]
